import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from artist.features.gaussian_blur import (
    denoise_flux,
)
from artist.raytracing import HeliostatRayTracer
from artist.scenario import Scenario
from artist.util import indices, set_logger_config
from artist.util.env import get_device

# Specify the path to your scenario.h5 file.
scenario_path = pathlib.Path(__file__).parent / "toy_model_gauss_heliostat.h5"

# Set up logger.
set_logger_config()

# Set the device.
device = get_device()

# Load the scenario.
with h5py.File(scenario_path) as scenario_path:
    scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_path, device=device
    )

# Inspect the scenario.
print(scenario)
print(
    f"The light source is a {scenario.light_sources.light_source_list[indices.first_light_source].__class__.__name__}."
)
print(
    f"The target areas have the following index mapping: {scenario.solar_tower.target_name_to_index}."
)
print(
    f"The first heliostat in the first group in the field is {scenario.heliostat_field.heliostat_groups[indices.first_heliostat_group].names[indices.first_heliostat]}."
)
print(
    f"The location of {scenario.heliostat_field.heliostat_groups[indices.first_heliostat_group].names[indices.first_heliostat]} is: {scenario.heliostat_field.heliostat_groups[indices.first_heliostat_group].positions[indices.first_heliostat].tolist()}."
)

# We only consider one heliostat for the beginning.
# There is only one heliostat in the scenario. That is why the active_heliostat_mask has only one element.
# To activate a heliostat once, you write a 1 at the index of the heliostat you want to activate.
# In our case we write a 1 at index 0. To activate this heliostat twice (this would duplicate the heliostat) you would write a 2 at index 0.

number_heliostats = 1
for heliostat_index in range(number_heliostats):
    active_heliostats_mask = torch.zeros(
        number_heliostats, dtype=torch.int32, device=device
    )
    active_heliostats_mask[heliostat_index] = 1

    # Activate the heliostat. Only activated heliostats will be aligned or ray-traced.
    scenario.heliostat_field.heliostat_groups[
        indices.first_heliostat_group
    ].activate_heliostats(
        active_heliostats_mask=active_heliostats_mask,
        device=device,
    )

    # Each heliostat has an aim point. We choose an aim point on one of the target areas.
    # Select the first target area as the designated target for this heliostat.
    target_area_indices = torch.tensor([3], device=device)

    # Use the center of the selected target area as the aim point.
    aim_point = scenario.solar_tower.get_centers_of_target_areas(
        target_area_indices=target_area_indices, device=device
    )
    print(f"The initial aim point used for this raytracing is {aim_point.tolist()}.")

    # Since we only have one heliostat we need to define a single incident ray direction.
    # When the sun is directly in the south, the rays point directly to the north.
    # Incident ray directions need to be normalized.
    incident_ray_directions = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)

    # Save the original surface points of the one active heliostat.
    original_surface_points = scenario.heliostat_field.heliostat_groups[
        0
    ].surface_points
    number_of_surface_points = scenario.heliostat_field.heliostat_groups[
        0
    ].active_surface_points.shape[indices.number_of_surface_points_dimension]

    # Align the heliostat(s).
    scenario.heliostat_field.heliostat_groups[
        indices.first_heliostat_group
    ].align_surfaces_with_incident_ray_directions(
        aim_points=aim_point,
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_heliostats_mask,
        device=device,
    )

    # Save the aligned surface points of the one active heliostat.
    # The original surface points are saved for all heliostats, active or not.
    # The aligned surface points are saved only for the active/current/aligned heliostats.
    # That is why we do not need to select specific indices here.
    aligned_surface_points = scenario.heliostat_field.heliostat_groups[
        indices.first_heliostat_group
    ].active_surface_points

    # N is rays per surface point; the true traced-ray count is N * P.
    print(f"surface points per heliostat: {number_of_surface_points}")

    # --- demonstration configuration ---------------------------------------
    ray_counts = [10, 30, 70, 100, 300, 700, 1000, 3000, 5000]
    reference_ray_count = 5000
    data_reps = 1  # noisy renders per ray count (1 is enough to illustrate)
    reference_reps = 10  # renders averaged into the reference (must beat the data)
    feature_example_n = 10  # ray count shown in the raw/blurred/reference triptych
    output_dir = pathlib.Path(__file__).parent

    # One counter across every render, so no two share a seed and the reference
    # renders stay disjoint from the data renders. Setting data_reps == reference_reps
    # reproduces the fully symmetric sampling of the original study.
    seed_counter = 100

    def render(ray_count, seed):
        """Render one flux bitmap, returning the raw and feature-blurred [H, W] images.

        The blur uses the shipped feature: denoise_flux reads the ray count and
        surface-point count off the tracer, predicts the optimal width, and applies
        it. Both images are normalized per traced ray so ray counts are comparable.
        """
        scenario.set_number_of_rays(ray_count)
        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=scenario.heliostat_field.heliostat_groups[
                indices.first_heliostat_group
            ],
            random_seed=seed,
        )
        image_south, *_ = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            target_area_indices=target_area_indices,
            device=device,
        )
        raw = image_south / (ray_count * number_of_surface_points)  # [1, H, W]
        blurred = denoise_flux(raw, ray_tracer)  # shipped feature does the blur
        return raw[0].detach().cpu(), blurred[0].detach().cpu()

    # Data: raw and feature-blurred renders at each ray count.
    raw_by_n, blurred_by_n = {}, {}
    for ray_count in ray_counts:
        pairs = [render(ray_count, seed_counter + i) for i in range(data_reps)]
        seed_counter += data_reps
        raw_by_n[ray_count] = [raw for raw, _ in pairs]
        blurred_by_n[ray_count] = [blurred for _, blurred in pairs]
        print(f"rendered N={ray_count} ({ray_count * number_of_surface_points} rays)")

    # Reference: the mean of several high-ray renders, cleaner than any data image.
    reference_raws = [
        render(reference_ray_count, seed_counter + i)[0] for i in range(reference_reps)
    ]
    seed_counter += reference_reps
    reference = torch.stack(reference_raws).mean(dim=0)  # [H, W]
    print(f"rendered reference: N={reference_ray_count} x {reference_reps} (mean)")

    # --- MSE vs reference, raw and blurred ---------------------------------
    def mean_mse(image_list):
        """Return the mean MSE of a list of images against the reference."""
        return float(
            np.mean(
                [
                    torch.nn.functional.mse_loss(img, reference).item()
                    for img in image_list
                ]
            )
        )

    raw_mse = np.array([mean_mse(raw_by_n[n]) for n in ray_counts])
    blurred_mse = np.array([mean_mse(blurred_by_n[n]) for n in ray_counts])
    true_rays = np.array(ray_counts, dtype=float) * number_of_surface_points

    print(f"\n{'N':>6} | {'raw MSE':>11} | {'blurred MSE':>11} | {'factor':>6}")
    for n, r, b in zip(ray_counts, raw_mse, blurred_mse):
        print(f"{n:>6} | {r:>11.3e} | {b:>11.3e} | {r / b:>5.1f}x")

    # --- plot 1: error before and after the shipped blur -------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(true_rays, raw_mse, "o-", color="#2a78d6", lw=2, label="raw (no blur)")
    ax.loglog(
        true_rays,
        blurred_mse,
        "o-",
        color="#008300",
        lw=2,
        label="after optimal_blur_sigma (shipped)",
    )
    ax.set_xlabel("true rays traced per image")
    ax.set_ylabel("MSE vs reference")
    ax.set_title("Gaussian-blur feature: flux error before and after")
    ax.grid(color="#e1e0d9", lw=0.8, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_mse_vs_rays.png", dpi=150)
    print("Saved feature_mse_vs_rays.png")

    # --- plot 2: raw vs blurred vs reference at one ray count --------------
    raw_example = raw_by_n[feature_example_n][0]
    blurred_example = blurred_by_n[feature_example_n][0]
    vmax = reference.max()
    panels = [
        (
            raw_example,
            f"raw (N={feature_example_n})",
            torch.nn.functional.mse_loss(raw_example, reference).item(),
        ),
        (
            blurred_example,
            "blurred (shipped feature)",
            torch.nn.functional.mse_loss(blurred_example, reference).item(),
        ),
        (reference, f"reference ({reference_ray_count} x {reference_reps})", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (img, title, mse) in zip(axes, panels):
        ax.imshow(img, cmap="inferno", vmin=0.0, vmax=vmax)
        ax.set_title(title, fontsize=11)
        if mse is not None:
            ax.set_xlabel(f"MSE {mse:.2e}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "A low-ray render, the same render denoised, and the reference", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(output_dir / "feature_example_flux.png", dpi=150)
    print("Saved feature_example_flux.png")
