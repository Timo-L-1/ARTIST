"""
Validate that gradients flow correctly to a heliostat's ENU position.

This is the "gradient plumbing" test for the differentiable-heliostat-position
feature: it checks that the autograd gradient of a flux-based loss with respect to
a heliostat's position matches an independent central finite-difference estimate.

What is validated
-----------------
The loss is the row centroid (vertical center of flux density, in pixels) of the
rendered flux image. We differentiate it with respect to the heliostat's up (u)
coordinate and compare:
  - the analytic gradient from ``autograd``, and
  - a central finite difference ``(L(u+e) - L(u-e)) / (2e)``.

Why u, and why u = 1 m
----------------------
``translate_enu`` consumes the e, n and u components identically, so a correct u
gradient exercises the same code path that carries e and n. u also has the
cleanest signal in this toy geometry. The base height is 1 m (not 0) to stay off
the u = 0 kinematics discontinuity, where the ideal kinematics switches between its
two discrete motor-position solutions and the loss is not differentiable.

Tolerance
---------
Agreement is asserted at ``rel_tol = 5%``. This is the intrinsic accuracy floor of
a float32 central finite difference (truncation error at large epsilon vs.
catastrophic cancellation at small epsilon), not a value chosen to make the test
pass. Tightening it would require running the pipeline in float64.
"""

import math
import pathlib

import h5py
import torch

from artist import ARTIST_ROOT
from artist.raytracing import HeliostatRayTracer
from artist.scenario import Scenario
from artist.util import indices

SCENARIO_PATH = (
    pathlib.Path(ARTIST_ROOT) / "tutorials/data/scenarios/single_heliostat_scenario.h5"
)
BASE_HEIGHT_U = 1.0  # m; off the ground to avoid the u = 0 kinematics branch flip
EPSILON = 1e-1  # m; center of the float32 central-difference plateau (~1% error;
# a sweep showed eps<=1e-2 falls into cancellation, eps in [2e-2, 2e-1] is stable)
REL_TOL = 0.05  # float32 finite-difference agreement floor (see module docstring)
RANDOM_SEED = 7  # pin ray distortions so only EPSILON differs across renders


def flux_row_centroid(flux: torch.Tensor) -> torch.Tensor:
    """
    Compute the row centroid (vertical center of flux density) of a flux image.

    This is a center of mass with flux as mass: a flux-weighted average of the row
    indices, normalized by the total flux so it reports location, not brightness.

    Parameters
    ----------
    flux : torch.Tensor
        The flux image. Shape is ``[1, H, W]`` (one target area).

    Returns
    -------
    torch.Tensor
        The row centroid, in pixels. Scalar tensor.
    """
    _, height, _ = flux.shape
    rows = torch.arange(height, dtype=flux.dtype, device=flux.device)  # (H,)
    flux_per_row = flux.sum(dim=2)  # (1, H) — collapse the column axis
    return (rows * flux_per_row).sum() / flux.sum()


def test_heliostat_position_gradient_matches_finite_difference(
    device: torch.device,
) -> None:
    """Assert the autograd position gradient matches a central finite difference."""
    with h5py.File(SCENARIO_PATH) as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    group = scenario.heliostat_field.heliostat_groups[indices.first_heliostat_group]
    active_mask = torch.tensor([1], dtype=torch.int32, device=device)
    target_area_indices = torch.tensor([0], device=device)
    aim_points = scenario.solar_tower.get_centers_of_target_areas(
        target_area_indices=target_area_indices, device=device
    )
    # Sun in the south -> rays travel north; single heliostat, single direction.
    incident_ray_directions = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)

    def render_loss(u_value: float) -> torch.Tensor:
        """
        Run the full position -> flux -> row-centroid pipeline at a given height.

        Rebuilds the active snapshots (``activate_heliostats`` — ``align_*`` is not
        idempotent) and the tracer from scratch each call, with the ray seed pinned,
        so the only thing that changes between calls is ``u_value``. No gradient is
        tracked here: this is a pure numerical evaluation for the finite difference.
        """
        with torch.no_grad():
            group.positions[0, indices.u] = u_value
            group.activate_heliostats(
                active_heliostats_mask=active_mask, device=device
            )
            group.align_surfaces_with_incident_ray_directions(
                aim_points=aim_points,
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_mask,
                device=device,
            )
            tracer = HeliostatRayTracer(
                scenario=scenario, heliostat_group=group, random_seed=RANDOM_SEED
            )
            bitmaps, *_ = tracer.trace_rays(
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_mask,
                target_area_indices=target_area_indices,
                device=device,
            )
        return flux_row_centroid(bitmaps)

    # --- analytic gradient d(row centroid)/d(u) at u = BASE_HEIGHT_U ---
    # Set the value while positions is still a plain tensor, then make it a grad leaf.
    group.positions[0, indices.u] = BASE_HEIGHT_U
    group.positions.requires_grad_()

    group.activate_heliostats(active_heliostats_mask=active_mask, device=device)
    group.align_surfaces_with_incident_ray_directions(
        aim_points=aim_points,
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_mask,
        device=device,
    )
    tracer = HeliostatRayTracer(
        scenario=scenario, heliostat_group=group, random_seed=RANDOM_SEED
    )
    bitmaps, *_ = tracer.trace_rays(
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_mask,
        target_area_indices=target_area_indices,
        device=device,
    )
    loss = flux_row_centroid(bitmaps)
    (position_gradient,) = torch.autograd.grad(loss, group.positions)
    analytic = position_gradient[0, indices.u]

    # --- central finite difference at the same point (gradient-free renders) ---
    loss_plus = render_loss(BASE_HEIGHT_U + EPSILON)
    loss_minus = render_loss(BASE_HEIGHT_U - EPSILON)
    central = (loss_plus - loss_minus) / (2 * EPSILON)

    assert math.isclose(analytic.item(), central.item(), rel_tol=REL_TOL), (
        f"Position gradient {analytic.item():.6f} does not match finite difference "
        f"{central.item():.6f} within rel_tol={REL_TOL}."
    )
