from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from forward_modeling.matlab_fwd3d.example_config import (
    build_matlab_example_density,
    build_matlab_example_grid,
)
from forward_modeling.matlab_fwd3d.gravity import (
    calculate_gravity_volume,
)
from forward_modeling.matlab_fwd3d.receivers import (
    ReceiverGrid,
)


def build_receiver_volume() -> ReceiverGrid:
    """
    Build the multi-height receiver grid.

    Z is positive downward, so negative Z values are above the surface.
    """

    return ReceiverGrid.from_ranges(
        x_min=-250.0,
        x_max=250.0,
        x_step=50.0,
        y_min=-250.0,
        y_max=250.0,
        y_step=50.0,
        z_values=(
            -10.0,
            -20.0,
            -30.0,
            -40.0,
            -50.0,
            -60.0,
            -70.0,
            -80.0,
        ),
    )


def save_gravity_layer(
    *,
    gravity_layer: np.ndarray,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
    receiver_z: float,
    output_path: Path,
) -> None:
    """
    Save one horizontal Gz map.
    """

    figure, axis = plt.subplots(
        figsize=(7.0, 6.0)
    )

    image = axis.imshow(
        gravity_layer,
        origin="lower",
        extent=(
            float(receiver_x.min()),
            float(receiver_x.max()),
            float(receiver_y.min()),
            float(receiver_y.max()),
        ),
        aspect="equal",
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
    colorbar.set_label("Gz [mGal]")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    """
    Generate a multi-height 3D Gz gravity volume.
    """

    package_directory = Path(__file__).resolve().parent

    output_directory = (
        package_directory
        / "gravity_volume_output"
    )

    figure_directory = (
        output_directory
        / "figures"
    )

    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_grid = build_matlab_example_grid()

    density_model = build_matlab_example_density(
        model_grid
    )

    receiver_grid = build_receiver_volume()

    gravity_volume = calculate_gravity_volume(
        model=density_model,
        model_grid=model_grid,
        receiver_grid=receiver_grid,
        channel=4,
    )

    density_path = (
        output_directory
        / "density_model.npy"
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

    metadata = {
        "density_model_shape": list(
            density_model.shape
        ),
        "gravity_volume_shape": list(
            gravity_volume.shape
        ),
        "gravity_channel": 4,
        "gravity_component": "Gz",
        "gravity_unit": "mGal",
        "receiver_x": receiver_grid.x.tolist(),
        "receiver_y": receiver_grid.y.tolist(),
        "receiver_z": receiver_grid.z.tolist(),
        "array_order": {
            "density": "density[z, y, x]",
            "gravity": (
                "gravity[z_receiver, y_receiver, x_receiver]"
            ),
        },
        "coordinate_convention": {
            "x": "positive east",
            "y": "positive north",
            "z": "positive downward",
        },
    }

    metadata_path = (
        output_directory
        / "gravity_volume_metadata.json"
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

    for z_index, receiver_z in enumerate(
        receiver_grid.z
    ):
        save_gravity_layer(
            gravity_layer=gravity_volume[z_index],
            receiver_x=receiver_grid.x,
            receiver_y=receiver_grid.y,
            receiver_z=float(receiver_z),
            output_path=(
                figure_directory
                / f"gz_z_{receiver_z:g}.png"
            ),
        )

    print()
    print("3D Gz gravity volume generated")
    print("=" * 31)
    print(
        f"Density model shape: "
        f"{density_model.shape}"
    )
    print(
        f"Gravity volume shape: "
        f"{gravity_volume.shape}"
    )
    print(
        f"Density model saved to: "
        f"{density_path}"
    )
    print(
        f"Gravity volume saved to: "
        f"{gravity_path}"
    )
    print(
        f"Metadata saved to: "
        f"{metadata_path}"
    )
    print(
        f"Figures saved to: "
        f"{figure_directory}"
    )


if __name__ == "__main__":
    main()