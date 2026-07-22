import types

import pytest
import torch

from artist.features.gaussian_blur import (
    apply_gaussian_blur,
    denoise_flux,
    optimal_blur_sigma,
)

# Blur widths measured by direct grid search per ray count in
# personal/Gauss_filter/fit_sigma_law.py: for each rays-per-surface-point count N
# (at P = 10 000 surface points, 256x256 bitmap), the sigma minimising the mean MSE
# against the converged reference. These are the ground truth the closed form must
# reproduce.
SURFACE_POINTS = 10_000
MEASURED_OPTIMA = {
    10: 5.8259,
    30: 4.9308,
    70: 4.2356,
    100: 4.0168,
    300: 3.2785,
    700: 2.8512,
    1000: 2.6852,
    3000: 2.1894,
    5000: 1.9942,
    10000: 1.7513,
    20000: 1.5201,
}


@pytest.mark.parametrize("number_of_rays, measured", MEASURED_OPTIMA.items())
def test_reproduces_measured_optima(number_of_rays: int, measured: float) -> None:
    """The law lands within 8% of every measured optimum.

    8% is the leave-one-out envelope from the fit: the endpoints (N=10, N=20000) are
    the loosest, and even there the MSE cost of the sigma error was under 0.6%.
    """
    predicted = optimal_blur_sigma(number_of_rays, SURFACE_POINTS)
    assert predicted == pytest.approx(measured, rel=0.08)


def test_depends_only_on_total_rays() -> None:
    """Splitting the same total ray budget differently must give the same sigma.

    The width depends on N * P, and multiplication is commutative, so swapping the
    two arguments -- the most likely caller mistake -- cannot change the result.
    """
    assert optimal_blur_sigma(100, 10_000) == optimal_blur_sigma(10_000, 100)


def test_scales_with_resolution() -> None:
    """A fixed physical blur spans twice the pixels at twice the resolution."""
    base = optimal_blur_sigma(100, SURFACE_POINTS, bitmap_resolution=256)
    doubled = optimal_blur_sigma(100, SURFACE_POINTS, bitmap_resolution=512)
    assert doubled == pytest.approx(2.0 * base)


def test_decreases_with_more_rays() -> None:
    """More rays mean less noise, so the optimal blur width shrinks monotonically."""
    counts = sorted(MEASURED_OPTIMA)
    sigmas = [optimal_blur_sigma(n, SURFACE_POINTS) for n in counts]
    assert all(earlier > later for earlier, later in zip(sigmas, sigmas[1:]))


# --- apply_gaussian_blur ---------------------------------------------------
# A batch of flux bitmaps: [number_of_heliostats, resolution_e, resolution_u].
BATCH, RESOLUTION = 3, 32


def test_blur_preserves_shape() -> None:
    """The blur returns bitmaps of the same shape, dtype and device as the input."""
    image = torch.rand(BATCH, RESOLUTION, RESOLUTION)
    blurred = apply_gaussian_blur(image, sigma=2.5)
    assert blurred.shape == image.shape
    assert blurred.dtype == image.dtype
    assert blurred.device == image.device


def test_blur_preserves_autograd_graph() -> None:
    """Gradients flow from the blurred output back to the input bitmaps.

    sigma is a constant, so the blur is not differentiable in sigma, but it must stay
    differentiable in the flux -- that is the gradient path the optimizer relies on.
    """
    image = torch.rand(BATCH, RESOLUTION, RESOLUTION, requires_grad=True)
    blurred = apply_gaussian_blur(image, sigma=2.5)
    assert blurred.requires_grad
    blurred.mean().backward()
    assert image.grad is not None


def test_blur_is_independent_per_heliostat() -> None:
    """Each bitmap is blurred on its own; changing one leaves the others untouched."""
    image = torch.rand(BATCH, RESOLUTION, RESOLUTION)
    perturbed = image.clone()
    perturbed[0] += 5.0
    original = apply_gaussian_blur(image, sigma=2.5)
    changed = apply_gaussian_blur(perturbed, sigma=2.5)
    assert not torch.allclose(original[0], changed[0])
    assert torch.allclose(original[1:], changed[1:])


def test_blur_conserves_flux() -> None:
    """A normalized-kernel blur redistributes flux without changing each bitmap's sum."""
    image = torch.zeros(BATCH, 51, 51)
    image[:, 25, 25] = 1.0  # a single hot pixel per bitmap
    blurred = apply_gaussian_blur(image, sigma=3.0)
    assert torch.allclose(image.sum(dim=(1, 2)), blurred.sum(dim=(1, 2)), atol=1e-5)


# --- denoise_flux ----------------------------------------------------------
@pytest.fixture
def mock_ray_tracer() -> types.SimpleNamespace:
    """Provide a stand-in tracer exposing only the attributes denoise_flux reads.

    denoise_flux reaches ``ray_tracer.light_source.number_of_rays`` and
    ``ray_tracer.heliostat_group.active_surface_points.shape[1]``. Supplying exactly
    those with nested SimpleNamespaces avoids building a real scenario; if the function
    ever reads a further attribute, this mock lacks it and the test fails loudly.
    """
    return types.SimpleNamespace(
        light_source=types.SimpleNamespace(number_of_rays=100),
        heliostat_group=types.SimpleNamespace(
            active_surface_points=torch.empty(1, SURFACE_POINTS, 4)
        ),
    )


def test_denoise_flux_preserves_shape_and_graph(
    mock_ray_tracer: types.SimpleNamespace,
) -> None:
    """A square batch is returned unchanged in shape, with the autograd graph intact."""
    image = torch.rand(BATCH, 256, 256, requires_grad=True)
    out = denoise_flux(image, mock_ray_tracer)
    assert out.shape == image.shape
    assert out.requires_grad
    out.mean().backward()
    assert image.grad is not None


def test_denoise_flux_rejects_non_square(
    mock_ray_tracer: types.SimpleNamespace,
) -> None:
    """Non-square bitmaps raise, since the blur width is calibrated for square only."""
    image = torch.rand(BATCH, 128, 64)
    with pytest.raises(ValueError):
        denoise_flux(image, mock_ray_tracer)


def test_denoise_flux_uses_predicted_sigma(
    mock_ray_tracer: types.SimpleNamespace,
) -> None:
    """The wrapper equals blurring by hand with the sigma the law predicts for N, P."""
    image = torch.rand(BATCH, 256, 256)
    sigma = optimal_blur_sigma(100, SURFACE_POINTS, bitmap_resolution=256)
    expected = apply_gaussian_blur(image, sigma)
    assert torch.allclose(denoise_flux(image, mock_ray_tracer), expected)
