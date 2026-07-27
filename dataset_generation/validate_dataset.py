from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


EXPECTED_DENSITY_SHAPE = (
    24,
    64,
    64,
)

EXPECTED_GRAVITY_SHAPE = (
    8,
    64,
    64,
)


def find_repository_root() -> Path:
    """
    Return the repository root.
    """

    return Path(__file__).resolve().parents[1]


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Validate a generated FWD3D gravity-volume dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "datasets/fwd3d_smoke_test"
        ),
        help=(
            "Dataset directory. Relative paths are interpreted "
            "from the repository root."
        ),
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help=(
            "Number of dataset examples to visualize."
        ),
    )

    return parser


def resolve_dataset_directory(
    *,
    repository_root: Path,
    dataset_argument: Path,
) -> Path:
    """
    Resolve the dataset path from the repository root.
    """

    dataset_directory = dataset_argument

    if not dataset_directory.is_absolute():
        dataset_directory = (
            repository_root
            / dataset_directory
        )

    dataset_directory = (
        dataset_directory.resolve()
    )

    if not dataset_directory.exists():
        raise FileNotFoundError(
            "Dataset directory does not exist:\n"
            f"{dataset_directory}"
        )

    return dataset_directory


def load_metadata(
    metadata_path: Path,
) -> dict[str, Any]:
    """
    Load dataset metadata.
    """

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as metadata_file:
        metadata: dict[str, Any] = (
            json.load(
                metadata_file
            )
        )

    return metadata


def load_manifest(
    manifest_path: Path,
) -> list[dict[str, str]]:
    """
    Load the CSV sample manifest.
    """

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest file not found:\n{manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        reader = csv.DictReader(
            manifest_file
        )

        rows = list(
            reader
        )

    if not rows:
        raise ValueError(
            "Manifest contains no samples."
        )

    return rows


def validate_metadata(
    *,
    metadata: dict[str, Any],
    manifest_rows: list[dict[str, str]],
) -> None:
    """
    Validate dataset-level metadata.
    """

    required_metadata = {
        "dataset_type",
        "gravity_component",
        "gravity_channel",
        "gravity_unit",
        "density_unit",
        "density_shape",
        "gravity_shape",
        "number_of_samples",
        "noise",
        "regularization",
    }

    missing_keys = (
        required_metadata
        - metadata.keys()
    )

    if missing_keys:
        raise ValueError(
            "Metadata is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    if (
        metadata["dataset_type"]
        != "single_positive_rectangular_body"
    ):
        raise ValueError(
            "Unexpected dataset type: "
            f"{metadata['dataset_type']}"
        )

    if metadata["gravity_component"] != "Gz":
        raise ValueError(
            "Baseline dataset must use Gz."
        )

    if int(
        metadata["gravity_channel"]
    ) != 4:
        raise ValueError(
            "Baseline dataset must use gravity channel 4."
        )

    if metadata["noise"] is not None:
        raise ValueError(
            "Baseline dataset unexpectedly contains noise."
        )

    if metadata["regularization"] is not None:
        raise ValueError(
            "Baseline dataset unexpectedly records regularization."
        )

    density_shape = tuple(
        int(value)
        for value in metadata["density_shape"]
    )

    gravity_shape = tuple(
        int(value)
        for value in metadata["gravity_shape"]
    )

    if density_shape != EXPECTED_DENSITY_SHAPE:
        raise ValueError(
            "Unexpected density shape in metadata: "
            f"{density_shape}"
        )

    if gravity_shape != EXPECTED_GRAVITY_SHAPE:
        raise ValueError(
            "Unexpected gravity shape in metadata: "
            f"{gravity_shape}"
        )

    expected_sample_count = int(
        metadata["number_of_samples"]
    )

    if len(manifest_rows) != expected_sample_count:
        raise ValueError(
            "Manifest sample count does not match metadata: "
            f"{len(manifest_rows)} != "
            f"{expected_sample_count}"
        )


def load_sample(
    sample_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one saved gravity-density pair.
    """

    if not sample_path.exists():
        raise FileNotFoundError(
            f"Sample file not found:\n{sample_path}"
        )

    with np.load(
        sample_path
    ) as sample:
        available_keys = set(
            sample.files
        )

        expected_keys = {
            "gravity",
            "density",
        }

        if available_keys != expected_keys:
            raise ValueError(
                f"{sample_path.name} contains keys "
                f"{sorted(available_keys)}, expected "
                f"{sorted(expected_keys)}."
            )

        gravity = np.asarray(
            sample["gravity"],
            dtype=np.float32,
        )

        density = np.asarray(
            sample["density"],
            dtype=np.float32,
        )

    return gravity, density


def validate_sample(
    *,
    sample_path: Path,
    gravity: np.ndarray,
    density: np.ndarray,
    manifest_row: dict[str, str],
) -> dict[str, float | int]:
    """
    Validate one generated sample.
    """

    if gravity.shape != EXPECTED_GRAVITY_SHAPE:
        raise ValueError(
            f"{sample_path.name}: unexpected gravity shape "
            f"{gravity.shape}."
        )

    if density.shape != EXPECTED_DENSITY_SHAPE:
        raise ValueError(
            f"{sample_path.name}: unexpected density shape "
            f"{density.shape}."
        )

    if gravity.dtype != np.float32:
        raise ValueError(
            f"{sample_path.name}: gravity must be float32."
        )

    if density.dtype != np.float32:
        raise ValueError(
            f"{sample_path.name}: density must be float32."
        )

    if not np.all(
        np.isfinite(gravity)
    ):
        raise ValueError(
            f"{sample_path.name}: gravity contains invalid values."
        )

    if not np.all(
        np.isfinite(density)
    ):
        raise ValueError(
            f"{sample_path.name}: density contains invalid values."
        )

    nonzero_density = density[
        density != 0.0
    ]

    if nonzero_density.size == 0:
        raise ValueError(
            f"{sample_path.name}: density contains no body."
        )

    if np.any(
        nonzero_density <= 0.0
    ):
        raise ValueError(
            f"{sample_path.name}: density contrast is not "
            "strictly positive."
        )

    unique_nonzero_density = np.unique(
        nonzero_density
    )

    if unique_nonzero_density.size != 1:
        raise ValueError(
            f"{sample_path.name}: body does not have one "
            "constant density contrast."
        )

    density_contrast = float(
        unique_nonzero_density[0]
    )

    expected_density_contrast = float(
        manifest_row["density_contrast"]
    )

    if not np.isclose(
        density_contrast,
        expected_density_contrast,
        rtol=1.0e-6,
        atol=1.0e-7,
    ):
        raise ValueError(
            f"{sample_path.name}: density contrast does not "
            "match the manifest."
        )

    x_start = int(
        manifest_row["x_start"]
    )
    x_end = int(
        manifest_row["x_end"]
    )
    y_start = int(
        manifest_row["y_start"]
    )
    y_end = int(
        manifest_row["y_end"]
    )
    z_start = int(
        manifest_row["z_start"]
    )
    z_end = int(
        manifest_row["z_end"]
    )

    expected_mask = np.zeros(
        EXPECTED_DENSITY_SHAPE,
        dtype=bool,
    )

    expected_mask[
        z_start:z_end,
        y_start:y_end,
        x_start:x_end,
    ] = True

    actual_mask = (
        density != 0.0
    )

    if not np.array_equal(
        actual_mask,
        expected_mask,
    ):
        raise ValueError(
            f"{sample_path.name}: occupied density cells do not "
            "match the rectangular body recorded in the manifest."
        )

    expected_nonzero_cells = (
        (x_end - x_start)
        * (y_end - y_start)
        * (z_end - z_start)
    )

    actual_nonzero_cells = int(
        np.count_nonzero(
            density
        )
    )

    manifest_nonzero_cells = int(
        manifest_row[
            "nonzero_density_cells"
        ]
    )

    if (
        actual_nonzero_cells
        != expected_nonzero_cells
    ):
        raise ValueError(
            f"{sample_path.name}: nonzero-cell count does not "
            "match body dimensions."
        )

    if (
        actual_nonzero_cells
        != manifest_nonzero_cells
    ):
        raise ValueError(
            f"{sample_path.name}: nonzero-cell count does not "
            "match manifest."
        )

    if np.all(
        gravity == 0.0
    ):
        raise ValueError(
            f"{sample_path.name}: gravity contains only zeros."
        )

    layer_maximums = np.max(
        gravity,
        axis=(1, 2),
    )

    if not np.all(
        np.diff(layer_maximums) < 0.0
    ):
        raise ValueError(
            f"{sample_path.name}: peak gravity does not decrease "
            "as receiver elevation increases."
        )

    return {
        "nonzero_density_cells": (
            actual_nonzero_cells
        ),
        "density_contrast": (
            density_contrast
        ),
        "gravity_minimum": float(
            np.min(
                gravity
            )
        ),
        "gravity_maximum": float(
            np.max(
                gravity
            )
        ),
        "gravity_mean": float(
            np.mean(
                gravity
            )
        ),
    }


def save_sample_figure(
    *,
    gravity: np.ndarray,
    density: np.ndarray,
    manifest_row: dict[str, str],
    output_path: Path,
) -> None:
    """
    Save a summary figure for one dataset sample.
    """

    z_start = int(
        manifest_row["z_start"]
    )
    z_end = int(
        manifest_row["z_end"]
    )

    center_density_z = (
        z_start
        + (
            z_end
            - z_start
        )
        // 2
    )

    density_slice = density[
        center_density_z
    ]

    nearest_gravity = gravity[0]
    farthest_gravity = gravity[-1]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16.0, 5.0),
    )

    density_image = axes[0].imshow(
        density_slice,
        origin="lower",
        aspect="equal",
    )

    axes[0].set_title(
        f"Density at Z index {center_density_z}"
    )
    axes[0].set_xlabel("X index")
    axes[0].set_ylabel("Y index")

    figure.colorbar(
        density_image,
        ax=axes[0],
        label="Density contrast [g/cm³]",
    )

    gravity_minimum = float(
        np.min(
            gravity
        )
    )

    gravity_maximum = float(
        np.max(
            gravity
        )
    )

    near_image = axes[1].imshow(
        nearest_gravity,
        origin="lower",
        aspect="equal",
        vmin=gravity_minimum,
        vmax=gravity_maximum,
    )

    axes[1].set_title(
        "Nearest receiver plane"
    )
    axes[1].set_xlabel("X index")
    axes[1].set_ylabel("Y index")

    figure.colorbar(
        near_image,
        ax=axes[1],
        label="Gz [mGal]",
    )

    far_image = axes[2].imshow(
        farthest_gravity,
        origin="lower",
        aspect="equal",
        vmin=gravity_minimum,
        vmax=gravity_maximum,
    )

    axes[2].set_title(
        "Farthest receiver plane"
    )
    axes[2].set_xlabel("X index")
    axes[2].set_ylabel("Y index")

    figure.colorbar(
        far_image,
        ax=axes[2],
        label="Gz [mGal]",
    )

    figure.suptitle(
        f"Sample {manifest_row['sample_index']} | "
        f"density = "
        f"{float(manifest_row['density_contrast']):.3f} g/cm³"
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
    Validate every sample in a generated dataset.
    """

    parser = build_argument_parser()
    arguments = parser.parse_args()

    if arguments.examples < 0:
        raise ValueError(
            "--examples must not be negative."
        )

    repository_root = (
        find_repository_root()
    )

    dataset_directory = (
        resolve_dataset_directory(
            repository_root=repository_root,
            dataset_argument=arguments.dataset,
        )
    )

    metadata_path = (
        dataset_directory
        / "metadata.json"
    )

    manifest_path = (
        dataset_directory
        / "manifest.csv"
    )

    metadata = load_metadata(
        metadata_path
    )

    manifest_rows = load_manifest(
        manifest_path
    )

    validate_metadata(
        metadata=metadata,
        manifest_rows=manifest_rows,
    )

    validation_directory = (
        dataset_directory
        / "validation"
    )

    validation_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_nonzero_cells = 0

    density_contrasts: list[
        float
    ] = []

    gravity_minimum = np.inf
    gravity_maximum = -np.inf

    print()
    print("Validating FWD3D dataset")
    print("=" * 26)
    print(
        f"Dataset: {dataset_directory}"
    )
    print(
        f"Samples: {len(manifest_rows):,}"
    )
    print()

    for row_index, manifest_row in enumerate(
        manifest_rows
    ):
        relative_path = Path(
            manifest_row["relative_path"]
        )

        sample_path = (
            dataset_directory
            / relative_path
        )

        gravity, density = load_sample(
            sample_path
        )

        statistics = validate_sample(
            sample_path=sample_path,
            gravity=gravity,
            density=density,
            manifest_row=manifest_row,
        )

        total_nonzero_cells += int(
            statistics[
                "nonzero_density_cells"
            ]
        )

        density_contrasts.append(
            float(
                statistics[
                    "density_contrast"
                ]
            )
        )

        gravity_minimum = min(
            gravity_minimum,
            float(
                statistics[
                    "gravity_minimum"
                ]
            ),
        )

        gravity_maximum = max(
            gravity_maximum,
            float(
                statistics[
                    "gravity_maximum"
                ]
            ),
        )

        if row_index < arguments.examples:
            output_path = (
                validation_directory
                / (
                    f"sample_"
                    f"{row_index:06d}.png"
                )
            )

            save_sample_figure(
                gravity=gravity,
                density=density,
                manifest_row=manifest_row,
                output_path=output_path,
            )

        completed = row_index + 1

        if (
            completed == 1
            or completed % 100 == 0
            or completed == len(
                manifest_rows
            )
        ):
            print(
                f"Validated "
                f"{completed:,}/"
                f"{len(manifest_rows):,}"
            )

    density_contrast_array = np.asarray(
        density_contrasts,
        dtype=np.float64,
    )

    print()
    print("Dataset validation complete")
    print("=" * 27)
    print(
        f"Density shape: "
        f"{EXPECTED_DENSITY_SHAPE}"
    )
    print(
        f"Gravity shape: "
        f"{EXPECTED_GRAVITY_SHAPE}"
    )
    print(
        f"Density contrast range: "
        f"{density_contrast_array.min():.4f} to "
        f"{density_contrast_array.max():.4f} g/cm³"
    )
    print(
        f"Gravity range: "
        f"{gravity_minimum:.8e} to "
        f"{gravity_maximum:.8e} mGal"
    )
    print(
        f"Total nonzero density cells: "
        f"{total_nonzero_cells:,}"
    )
    print()
    print(
        "One rectangular body per sample: PASSED"
    )
    print(
        "Positive density contrast only: PASSED"
    )
    print(
        "No observational noise recorded: PASSED"
    )
    print(
        "No regularization recorded: PASSED"
    )
    print(
        "Array shape and finite-value checks: PASSED"
    )
    print(
        "Gravity elevation behavior: PASSED"
    )
    print()
    print(
        f"Validation figures: "
        f"{validation_directory}"
    )


if __name__ == "__main__":
    main()