from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from forward_modeling.matlab_fwd3d.forward_model import (
    FWD3DGravityForwardModel,
)
from forward_modeling.matlab_fwd3d.grid_adapter import (
    model_grid_from_grid_spec,
)
from forward_modeling.matlab_fwd3d.receivers import (
    ReceiverGrid,
)
from synthetic_models.common.bodies import (
    RectangularBodySpec,
    build_density_model,
)
from synthetic_models.common.grid import GridSpec


def define_case(
    grid: GridSpec,
) -> RectangularBodySpec:
    """
    Define the initial repository-integrated FWD3D test body.

    The body is centered horizontally and occupies five vertical layers.
    """

    body_width_x = 12
    body_width_y = 12
    body_thickness_z = 5

    x_start = (
        grid.nx - body_width_x
    ) // 2

    y_start = (
        grid.ny - body_width_y
    ) // 2

    z_start = 6

    return RectangularBodySpec(
        name="centered_rectangular_body",
        x_start=x_start,
        x_end=x_start + body_width_x,
        y_start=y_start,
        y_end=y_start + body_width_y,
        z_start=z_start,
        z_end=z_start + body_thickness_z,
        density_contrast=1.0,
    )


def build_receiver_grid(
    grid: GridSpec,
) -> ReceiverGrid:
    """
    Build a regular 3D receiver grid.

    The X and Y receiver coordinates match the repository model grid.
    Eight receiver planes are placed above the model surface.

    Z is positive downward, so negative receiver Z values are above
    the surface.
    """

    receiver_z = -grid.dz * np.arange(
        1,
        9,
        dtype=np.float64,
    )

    receiver_x = np.linspace(
        grid.x_min,
        grid.x_max,
        grid.nx,
        dtype=np.float64,
    )

    receiver_y = np.linspace(
        grid.y_min,
        grid.y_max,
        grid.ny,
        dtype=np.float64,
    )

    return ReceiverGrid(
        x=receiver_x,
        y=receiver_y,
        z=receiver_z,
    )


def save_density_slices(
    *,
    density_model: np.ndarray,
    grid: GridSpec,
    output_directory: Path,
) -> None:
    """
    Save horizontal density slices containing nonzero cells.
    """

    density_figure_directory = (
        output_directory
        / "density_slices"
    )

    density_figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for z_index in range(
        density_model.shape[0]
    ):
        density_slice = density_model[
            z_index
        ]

        if not np.any(
            density_slice != 0.0
        ):
            continue

        figure, axis = plt.subplots(
            figsize=(7.0, 6.0)
        )

        image = axis.imshow(
            density_slice,
            origin="lower",
            extent=(
                grid.x_min,
                grid.x_max,
                grid.y_min,
                grid.y_max,
            ),
            aspect="equal",
        )

        physical_z = (
            grid.z_min
            + z_index * grid.dz
        )

        axis.set_xlabel("X [m]")
        axis.set_ylabel("Y [m]")
        axis.set_title(
            f"Density model at Z = {physical_z:g} m"
        )

        colorbar = figure.colorbar(
            image,
            ax=axis,
        )

        colorbar.set_label(
            "Density contrast [g/cm³]"
        )

        figure.tight_layout()

        figure.savefig(
            density_figure_directory
            / f"density_z_index_{z_index:02d}.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )


def save_gravity_slices(
    *,
    gravity_volume: np.ndarray,
    receiver_grid: ReceiverGrid,
    output_directory: Path,
) -> None:
    """
    Save one horizontal Gz map for every receiver elevation.
    """

    gravity_figure_directory = (
        output_directory
        / "gravity_slices"
    )

    gravity_figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_minimum = float(
        np.min(
            gravity_volume
        )
    )

    global_maximum = float(
        np.max(
            gravity_volume
        )
    )

    for z_index, receiver_z in enumerate(
        receiver_grid.z
    ):
        gravity_slice = gravity_volume[
            z_index
        ]

        figure, axis = plt.subplots(
            figsize=(7.0, 6.0)
        )

        image = axis.imshow(
            gravity_slice,
            origin="lower",
            extent=(
                float(
                    receiver_grid.x.min()
                ),
                float(
                    receiver_grid.x.max()
                ),
                float(
                    receiver_grid.y.min()
                ),
                float(
                    receiver_grid.y.max()
                ),
            ),
            aspect="equal",
            vmin=global_minimum,
            vmax=global_maximum,
        )

        axis.set_xlabel("X [m]")
        axis.set_ylabel("Y [m]")
        axis.set_title(
            f"Gz at receiver Z = {receiver_z:g} m"
        )

        colorbar = figure.colorbar(
            image,
            ax=axis,
        )

        colorbar.set_label(
            "Gz [mGal]"
        )

        figure.tight_layout()

        figure.savefig(
            gravity_figure_directory
            / f"gz_receiver_z_{receiver_z:g}.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )


def save_center_profiles(
    *,
    gravity_volume: np.ndarray,
    receiver_grid: ReceiverGrid,
    output_path: Path,
) -> None:
    """
    Save centerline Gz profiles for every receiver elevation.
    """

    center_y_index = (
        gravity_volume.shape[1]
        // 2
    )

    figure, axis = plt.subplots(
        figsize=(9.0, 6.0)
    )

    for z_index, receiver_z in enumerate(
        receiver_grid.z
    ):
        axis.plot(
            receiver_grid.x,
            gravity_volume[
                z_index,
                center_y_index,
                :,
            ],
            marker="o",
            markersize=3,
            label=f"Z = {receiver_z:g} m",
        )

    axis.set_xlabel("X [m]")
    axis.set_ylabel("Gz [mGal]")
    axis.set_title(
        "FWD3D center gravity profiles"
    )

    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main() -> None:
    """
    Run the first repository-integrated FWD3D gravity-volume experiment.
    """

    experiment_directory = (
        Path(__file__).resolve().parent
    )

    output_directory = (
        experiment_directory
        / "output"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    repository_grid = GridSpec()

    model_grid = model_grid_from_grid_spec(
        repository_grid
    )

    receiver_grid = build_receiver_grid(
        repository_grid
    )

    case = define_case(
        repository_grid
    )

    case.validate(
        repository_grid
    )

    density_model = build_density_model(
        grid=repository_grid,
        case=case,
    )

    forward_model = FWD3DGravityForwardModel(
        model_grid=model_grid,
        receiver_grid=receiver_grid,
        channel=4,
        receiver_chunk_size=128,
    )

    gravity_volume = forward_model.calculate(
        density_model
    )

    expected_density_shape = (
        repository_grid.nz,
        repository_grid.ny,
        repository_grid.nx,
    )

    expected_gravity_shape = (
        receiver_grid.nz,
        receiver_grid.ny,
        receiver_grid.nx,
    )

    if density_model.shape != expected_density_shape:
        raise RuntimeError(
            "Unexpected density-model shape. "
            f"Expected {expected_density_shape}, "
            f"received {density_model.shape}."
        )

    if gravity_volume.shape != expected_gravity_shape:
        raise RuntimeError(
            "Unexpected gravity-volume shape. "
            f"Expected {expected_gravity_shape}, "
            f"received {gravity_volume.shape}."
        )

    density_path = (
        output_directory
        / "true_density.npy"
    )

    gravity_path = (
        output_directory
        / "gravity_volume_gz.npy"
    )

    np.save(
        density_path,
        density_model,
    )

    np.save(
        gravity_path,
        gravity_volume,
    )

    layer_maximums = np.max(
        gravity_volume,
        axis=(1, 2),
    )

    metadata = {
        "case_name": case.name,
        "density_contrast_g_per_cm3": (
            case.density_contrast
        ),
        "density_shape": list(
            density_model.shape
        ),
        "gravity_shape": list(
            gravity_volume.shape
        ),
        "gravity_channel": 4,
        "gravity_component": "Gz",
        "gravity_unit": "mGal",
        "density_array_order": (
            "density[z, y, x]"
        ),
        "gravity_array_order": (
            "gravity[z_receiver, y_receiver, x_receiver]"
        ),
        "receiver_x": (
            receiver_grid.x.tolist()
        ),
        "receiver_y": (
            receiver_grid.y.tolist()
        ),
        "receiver_z": (
            receiver_grid.z.tolist()
        ),
        "layer_maximum_gz": (
            layer_maximums.tolist()
        ),
        "body_indices": {
            "x_start": case.x_start,
            "x_end": case.x_end,
            "y_start": case.y_start,
            "y_end": case.y_end,
            "z_start": case.z_start,
            "z_end": case.z_end,
        },
        "nonzero_density_cells": int(
            np.count_nonzero(
                density_model
            )
        ),
    }

    metadata_path = (
        output_directory
        / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=2,
        )

    save_density_slices(
        density_model=density_model,
        grid=repository_grid,
        output_directory=output_directory,
    )

    save_gravity_slices(
        gravity_volume=gravity_volume,
        receiver_grid=receiver_grid,
        output_directory=output_directory,
    )

    save_center_profiles(
        gravity_volume=gravity_volume,
        receiver_grid=receiver_grid,
        output_path=(
            output_directory
            / "center_profiles.png"
        ),
    )

    print()
    print("Repository FWD3D gravity-volume experiment")
    print("=" * 45)
    print(
        f"Case: {case.name}"
    )
    print(
        f"Density shape: "
        f"{density_model.shape}"
    )
    print(
        f"Gravity shape: "
        f"{gravity_volume.shape}"
    )
    print(
        f"Nonzero density cells: "
        f"{np.count_nonzero(density_model)}"
    )
    print()
    print("Receiver-layer maximum Gz:")

    for receiver_z, maximum in zip(
        receiver_grid.z,
        layer_maximums,
        strict=True,
    ):
        print(
            f"Z = {receiver_z:7.2f} m | "
            f"maximum Gz = {maximum:.12e} mGal"
        )

    print()
    print(
        f"Density model: {density_path}"
    )
    print(
        f"Gravity volume: {gravity_path}"
    )
    print(
        f"Metadata: {metadata_path}"
    )
    print(
        f"Figures: {output_directory}"
    )


if __name__ == "__main__":
    main()