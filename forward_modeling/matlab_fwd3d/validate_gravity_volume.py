from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_outputs(
    output_directory: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load the generated density model, gravity volume, and metadata.
    """

    density_path = (
        output_directory
        / "density_model.npy"
    )

    gravity_path = (
        output_directory
        / "gravity_volume_gz.npy"
    )

    metadata_path = (
        output_directory
        / "gravity_volume_metadata.json"
    )

    for path in (
        density_path,
        gravity_path,
        metadata_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required output file was not found:\n{path}"
            )

    density_model = np.load(
        density_path
    )

    gravity_volume = np.load(
        gravity_path
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as metadata_file:
        metadata = json.load(
            metadata_file
        )

    return (
        density_model,
        gravity_volume,
        metadata,
    )


def validate_shapes(
    *,
    density_model: np.ndarray,
    gravity_volume: np.ndarray,
    metadata: dict,
) -> None:
    """
    Confirm that saved array shapes match the metadata.
    """

    expected_density_shape = tuple(
        metadata["density_model_shape"]
    )

    expected_gravity_shape = tuple(
        metadata["gravity_volume_shape"]
    )

    if density_model.shape != expected_density_shape:
        raise ValueError(
            "Density shape does not match metadata: "
            f"{density_model.shape} != "
            f"{expected_density_shape}."
        )

    if gravity_volume.shape != expected_gravity_shape:
        raise ValueError(
            "Gravity shape does not match metadata: "
            f"{gravity_volume.shape} != "
            f"{expected_gravity_shape}."
        )

    receiver_z = metadata["receiver_z"]

    if gravity_volume.shape[0] != len(receiver_z):
        raise ValueError(
            "The gravity Z dimension does not match the "
            "number of receiver elevations."
        )


def validate_values(
    *,
    density_model: np.ndarray,
    gravity_volume: np.ndarray,
) -> None:
    """
    Check for invalid or unexpected values.
    """

    if not np.all(
        np.isfinite(density_model)
    ):
        raise ValueError(
            "Density model contains NaN or infinite values."
        )

    if not np.all(
        np.isfinite(gravity_volume)
    ):
        raise ValueError(
            "Gravity volume contains NaN or infinite values."
        )

    if np.all(
        gravity_volume == 0.0
    ):
        raise ValueError(
            "Gravity volume contains only zeros."
        )

    if np.any(
        gravity_volume < 0.0
    ):
        raise ValueError(
            "This positive-density example unexpectedly produced "
            "negative Gz values."
        )


def calculate_layer_statistics(
    *,
    gravity_volume: np.ndarray,
    receiver_z: np.ndarray,
) -> list[dict[str, float]]:
    """
    Calculate summary statistics for every receiver elevation.
    """

    statistics: list[dict[str, float]] = []

    for z_index, z_value in enumerate(
        receiver_z
    ):
        gravity_layer = gravity_volume[
            z_index
        ]

        statistics.append(
            {
                "receiver_z": float(z_value),
                "minimum": float(
                    np.min(gravity_layer)
                ),
                "maximum": float(
                    np.max(gravity_layer)
                ),
                "mean": float(
                    np.mean(gravity_layer)
                ),
                "standard_deviation": float(
                    np.std(gravity_layer)
                ),
            }
        )

    return statistics


def validate_peak_behavior(
    *,
    statistics: list[dict[str, float]],
) -> None:
    """
    Confirm that the gravity peak decreases as receivers move farther away.

    The receiver list is ordered from Z = -10 m to Z = -80 m.
    Because the model uses positive-downward Z, increasingly negative
    receiver Z values are farther above the density body.
    """

    maximum_values = np.asarray(
        [
            item["maximum"]
            for item in statistics
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.diff(maximum_values) < 0.0
    ):
        raise ValueError(
            "Maximum Gz does not decrease monotonically as "
            "receiver elevation moves farther from the body."
        )


def validate_horizontal_symmetry(
    *,
    gravity_volume: np.ndarray,
    absolute_tolerance: float = 1.0e-12,
) -> float:
    """
    Check symmetry about the X and Y centerlines.

    Returns
    -------
    float
        Maximum symmetry mismatch.
    """

    flipped_x = gravity_volume[
        :,
        :,
        ::-1,
    ]

    flipped_y = gravity_volume[
        :,
        ::-1,
        :,
    ]

    x_difference = np.max(
        np.abs(
            gravity_volume
            - flipped_x
        )
    )

    y_difference = np.max(
        np.abs(
            gravity_volume
            - flipped_y
        )
    )

    maximum_difference = float(
        max(
            x_difference,
            y_difference,
        )
    )

    if maximum_difference > absolute_tolerance:
        raise ValueError(
            "The gravity volume is not symmetric within tolerance. "
            f"Maximum mismatch: {maximum_difference:.12e}"
        )

    return maximum_difference


def save_peak_plot(
    *,
    statistics: list[dict[str, float]],
    output_path: Path,
) -> None:
    """
    Plot peak Gz against receiver elevation.
    """

    receiver_z = np.asarray(
        [
            item["receiver_z"]
            for item in statistics
        ],
        dtype=np.float64,
    )

    maximum_gz = np.asarray(
        [
            item["maximum"]
            for item in statistics
        ],
        dtype=np.float64,
    )

    figure, axis = plt.subplots(
        figsize=(7.0, 5.0)
    )

    axis.plot(
        receiver_z,
        maximum_gz,
        marker="o",
    )

    axis.set_xlabel(
        "Receiver Z [m]"
    )

    axis.set_ylabel(
        "Maximum Gz [mGal]"
    )

    axis.set_title(
        "Gravity peak versus receiver elevation"
    )

    axis.grid(
        True
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_center_profile_plot(
    *,
    gravity_volume: np.ndarray,
    receiver_z: np.ndarray,
    receiver_x: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot the center Y profile for every receiver elevation.
    """

    center_y_index = (
        gravity_volume.shape[1]
        // 2
    )

    figure, axis = plt.subplots(
        figsize=(8.0, 6.0)
    )

    for z_index, z_value in enumerate(
        receiver_z
    ):
        axis.plot(
            receiver_x,
            gravity_volume[
                z_index,
                center_y_index,
                :,
            ],
            marker="o",
            label=f"Z = {z_value:g} m",
        )

    axis.set_xlabel(
        "X [m]"
    )

    axis.set_ylabel(
        "Gz [mGal]"
    )

    axis.set_title(
        "Center gravity profiles at multiple elevations"
    )

    axis.legend()

    axis.grid(
        True
    )

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
    Validate the generated three-dimensional gravity volume.
    """

    package_directory = (
        Path(__file__).resolve().parent
    )

    output_directory = (
        package_directory
        / "gravity_volume_output"
    )

    validation_directory = (
        output_directory
        / "validation"
    )

    validation_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        density_model,
        gravity_volume,
        metadata,
    ) = load_outputs(
        output_directory
    )

    receiver_x = np.asarray(
        metadata["receiver_x"],
        dtype=np.float64,
    )

    receiver_z = np.asarray(
        metadata["receiver_z"],
        dtype=np.float64,
    )

    validate_shapes(
        density_model=density_model,
        gravity_volume=gravity_volume,
        metadata=metadata,
    )

    validate_values(
        density_model=density_model,
        gravity_volume=gravity_volume,
    )

    statistics = (
        calculate_layer_statistics(
            gravity_volume=gravity_volume,
            receiver_z=receiver_z,
        )
    )

    validate_peak_behavior(
        statistics=statistics
    )

    symmetry_error = (
        validate_horizontal_symmetry(
            gravity_volume=gravity_volume
        )
    )

    save_peak_plot(
        statistics=statistics,
        output_path=(
            validation_directory
            / "peak_gz_vs_receiver_z.png"
        ),
    )

    save_center_profile_plot(
        gravity_volume=gravity_volume,
        receiver_z=receiver_z,
        receiver_x=receiver_x,
        output_path=(
            validation_directory
            / "center_profiles.png"
        ),
    )

    print()
    print("3D gravity-volume validation")
    print("=" * 31)
    print(
        f"Density shape: "
        f"{density_model.shape}"
    )
    print(
        f"Gravity shape: "
        f"{gravity_volume.shape}"
    )
    print()
    print("Layer statistics")
    print("-" * 65)

    for item in statistics:
        print(
            f"Z = {item['receiver_z']:7.1f} m | "
            f"min = {item['minimum']:.8f} | "
            f"max = {item['maximum']:.8f} | "
            f"mean = {item['mean']:.8f}"
        )

    print()
    print(
        "Maximum horizontal symmetry error: "
        f"{symmetry_error:.12e}"
    )
    print(
        "Peak decreases with receiver distance: PASSED"
    )
    print(
        "Finite-value validation: PASSED"
    )
    print(
        "Shape validation: PASSED"
    )
    print()
    print(
        f"Validation figures: "
        f"{validation_directory}"
    )


if __name__ == "__main__":
    main()