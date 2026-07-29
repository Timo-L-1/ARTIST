import math
from typing import TYPE_CHECKING

import torch

from artist.util import indices

if TYPE_CHECKING:
    from artist.raytracing import HeliostatRayTracer
from typing import TYPE_CHECKING

# Optimal blur width fitted in personal/Gauss_filter/fit_sigma_law.py on a sweep of
# 11 ray counts (one focal spot, 10 000 surface points, 256x256 bitmap). The free
# power-law fit was sigma = 8.9749 * N^(-0.1769) in rays-per-surface-point N;
# rewritten below in the total ray count R = N * P to sigma = 45.77 * R^(-0.1769).
_AXIS_INTERCEPT = 45.77
_POWER_LAW_EXPONENT = -0.1769
_CALIBRATION_RESOLUTION = 256


def optimal_blur_sigma(
    number_of_rays: int,
    number_of_surface_points: int,
    bitmap_resolution: int = 256,
) -> float:
    """
    Predict the Gaussian-blur width that best denoises a low-ray flux bitmap.

    Monte-Carlo ray tracing leaves shot noise on the flux bitmap that shrinks only
    slowly with the ray count. Blurring the bitmap with the right width removes most
    of that noise, so a cheap low-ray render can approach the accuracy of a far more
    expensive one. This returns that width for a given ray budget.

    The width follows a power law in the total number of traced rays,
    ``R = number_of_rays * number_of_surface_points``, fitted on a ray-count sweep
    (see module header for the fit and its scope). The result is a length in bitmap
    pixels, so it scales with the bitmap resolution.

    Parameters
    ----------
    number_of_rays : int
        Rays sampled per surface point (the light source's rays-per-point setting,
        ``N``). This is *not* the total ray count.
    number_of_surface_points : int
        Discrete surface points per heliostat (``P``), e.g.
        ``group.active_surface_points.shape[1]``. Required, and easy to omit: the
        blur width depends on the total rays ``N * P``, so leaving ``P`` out (or
        passing the total ray count as ``number_of_rays``) silently returns a width
        wrong by a factor of ``P`` -- there is no error, only a bad result.
    bitmap_resolution : int
        Side length in pixels of the (square) flux bitmap the blur will be applied
        to (default is 256, the resolution the law was calibrated at). The returned
        width scales linearly with this.

    Returns
    -------
    float
        Optimal Gaussian standard deviation in bitmap pixels.

    Notes
    -----
    Calibrated on a single focal spot at one surface-point count and one bitmap
    resolution. Two extrapolations are therefore untested:

    - Only the product ``N * P`` was varied, at a fixed ``P = 10 000``. Predicting
      from ``N * P`` assumes the width depends on the total ray count alone; a
      different ``P`` may need recalibration of the constants above.
    - The pixel-to-resolution scaling is a geometric argument (a fixed physical blur
      spans proportionally more pixels at higher resolution), not a measured result.

    The constants are module-level so they can be recalibrated for other geometries.
    """
    total_rays = number_of_rays * number_of_surface_points
    resolution_scale = bitmap_resolution / _CALIBRATION_RESOLUTION
    return _AXIS_INTERCEPT * total_rays**_POWER_LAW_EXPONENT * resolution_scale


def apply_gaussian_blur(
    image: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Blur a batch of flux bitmaps with a fixed-width Gaussian, preserving autograd.

    Applies an isotropic Gaussian blur of standard deviation ``sigma`` (in pixels) to
    each bitmap in the batch independently. The width is a constant, so no gradient
    flows with respect to ``sigma``; the operation is differentiable in the input
    bitmaps, so gradients propagate from the blurred output back through the flux to
    the physics parameters that produced it. Pair with :func:`optimal_blur_sigma` to
    choose ``sigma`` from the ray count.

    The Gaussian kernel is separable and built as the outer product of two 1D
    kernels, normalized to sum to one so the total flux of each bitmap is conserved.
    Its radius follows ``sigma`` (``ceil(4 * sigma)``), wide enough that the truncated
    tail is negligible. The kernel is created on the input's device and dtype, so the
    input may live on CPU or GPU.

    Parameters
    ----------
    image : torch.Tensor
        Batch of flux bitmaps to blur.
        Shape is ``[number_of_heliostats, bitmap_resolution_e, bitmap_resolution_u]``.
    sigma : float
        Gaussian standard deviation in bitmap pixels, e.g. from
        :func:`optimal_blur_sigma`.

    Returns
    -------
    torch.Tensor
        The blurred bitmaps, same shape, dtype and device as ``image`` and still
        attached to the autograd graph.

    See Also
    --------
    optimal_blur_sigma : Predict ``sigma`` from the ray count.
    """
    kernel_radius = math.ceil(4 * sigma)
    offsets_grid = torch.arange(
        -kernel_radius, kernel_radius + 1, device=image.device, dtype=image.dtype
    )
    bell = torch.exp(-(offsets_grid**2) / (2 * sigma**2))
    kernel_2d = torch.outer(bell, bell)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return torch.nn.functional.conv2d(
        image[:, None], kernel_2d[None, None], padding=kernel_radius
    )[:, 0]


def denoise_flux(
    flux_distributions: torch.Tensor,
    ray_tracer: "HeliostatRayTracer",
) -> torch.Tensor:
    """
    Denoise a batch of flux bitmaps with the ray-count-optimal Gaussian blur.

    Convenience wrapper that reads the ray budget off ``ray_tracer``, predicts the
    optimal blur width with :func:`optimal_blur_sigma`, and applies it with
    :func:`apply_gaussian_blur`. Deriving the ray count from the tracer means the
    caller cannot mismatch the rays-per-point and surface-point counts. Intended as an
    optional post-processing step on the tracer's output::

        bitmaps, *_ = ray_tracer.trace_rays(...)
        bitmaps = denoise_flux(bitmaps, ray_tracer)

    Parameters
    ----------
    flux_distributions : torch.Tensor
        Batch of flux bitmaps produced by the ray tracer. Must be square.
        Shape is ``[number_of_active_heliostats, bitmap_resolution, bitmap_resolution]``.
    ray_tracer : HeliostatRayTracer
        The tracer that produced the bitmaps. Its light source and heliostat group
        supply the rays-per-point and surface-point counts; both persist after tracing,
        so the tracer can be queried afterwards.

    Returns
    -------
    torch.Tensor
        The denoised bitmaps, same shape, dtype and device as the input and still
        attached to the autograd graph.

    Raises
    ------
    ValueError
        If the bitmaps are not square. The blur width is calibrated for square bitmaps
        (isotropic blur); non-square bitmaps would need per-axis (anisotropic) widths.

    See Also
    --------
    optimal_blur_sigma : Predict the blur width from the ray count.
    apply_gaussian_blur : Apply the blur to the bitmaps.
    """
    if flux_distributions.shape[-2] != flux_distributions.shape[-1]:
        raise ValueError(
            f"Bitmaps are not square. Blur width is calibrated for square bitmaps only. Bitmap e-dim: {flux_distributions.shape[-2]}; u-dim: {flux_distributions.shape[-1]}"
        )
    number_of_rays = ray_tracer.light_source.number_of_rays
    number_of_surface_points = ray_tracer.heliostat_group.active_surface_points.shape[
        indices.number_of_surface_points_dimension
    ]
    bitmap_resolution = flux_distributions.shape[-1]
    sigma = optimal_blur_sigma(
        number_of_rays, number_of_surface_points, bitmap_resolution
    )
    blurred_flux_distributions = apply_gaussian_blur(flux_distributions, sigma)
    return blurred_flux_distributions
