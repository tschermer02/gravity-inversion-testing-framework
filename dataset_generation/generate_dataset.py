from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from dataset_generation.case_sampler import (
    RectangularBodySampler,
)
from dataset_generation.config import (
    DatasetGenerationConfig,
)
from forward_modeling.matlab_fwd3d.forward_model import (
    FWD3DGravityForwardModel,
)
from forward_modeling.matlab_fwd3d.grid_adapter import (
    model_grid_from_grid_spec,
)
from synthetic_models.common.bodies import (
    build_density_model,
)
from dataset_generation.matlab_grid import MatlabCompatibleGridSpec


MANIFEST_FIELD_NAMES = (
    "sample_index",
    "relative_path",
    "body_name",
    "x_start",
    "x_end",
    "y_start",
    "y_end",
    "z_start",
    "z_end",
    "width_x",
    "width_y",
    "thickness_z",
    "top_depth_index",
    "bottom_depth_index",
    "center_x_index",
    "center_y_index",
    "center_z_index",
    "top_depth_m",
    "bottom_depth_m",
    "center_depth_m",
    "width_x_m",
    "width_y_m",
    "thickness_z_m",
    "center_x_m",
    "center_y_m",
    "density_contrast",
    "nonzero_density_cells",
    "gravity_minimum_mgal",
    "gravity_maximum_mgal",
    "gravity_mean_mgal",
    "gravity_std_mgal",
)


def find_repository_root() -> Path:
    """
    Return the repository root containing this package.

    ``generate_dataset.py`` is expected at:

        repository_root/dataset_generation/generate_dataset.py
    """

    return Path(__file__).resolve().parents[1]


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a baseline FWD3D gravity-volume dataset "
            "containing one positive rectangular body per sample."
        )
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help=(
            "Number of examples to generate. "
            "Overrides DatasetGenerationConfig."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed. Overrides DatasetGenerationConfig."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Dataset output directory. Relative paths are interpreted "
            "from the repository root."
        ),
    )

    parser.add_argument(
        "--maximum-top-depth-index",
        type=int,
        default=None,
        help=(
            "Optional maximum body top-depth edge index. "
            "With the MATLAB-compatible 10 m grid, index 8 "
            "corresponds to 80 m. The configured default is unchanged "
            "when this option is omitted."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete and recreate an existing output directory."
        ),
    )

    parser.add_argument(
        "--uncompressed",
        action="store_true",
        help=(
            "Use np.savez instead of np.savez_compressed."
        ),
    )

    return parser


def apply_command_line_overrides(
    *,
    config: DatasetGenerationConfig,
    arguments: argparse.Namespace,
) -> DatasetGenerationConfig:
    """
    Return a configuration updated from command-line values.
    """

    updated = config

    if arguments.samples is not None:
        updated = replace(
            updated,
            number_of_samples=arguments.samples,
        )

    if arguments.seed is not None:
        updated = replace(
            updated,
            random_seed=arguments.seed,
        )

    if arguments.output is not None:
        updated = replace(
            updated,
            output_directory=arguments.output,
        )

    if arguments.maximum_top_depth_index is not None:
        updated = replace(
            updated,
            body=replace(
                updated.body,
                maximum_top_depth_index=(
                    arguments.maximum_top_depth_index
                ),
            ),
        )

    if arguments.overwrite:
        updated = replace(
            updated,
            overwrite=True,
        )

    if arguments.uncompressed:
        updated = replace(
            updated,
            compressed=False,
        )

    return updated


def prepare_output_directory(
    *,
    output_directory: Path,
    overwrite: bool,
) -> Path:
    """
    Create an empty dataset output directory.
    """

    if output_directory.exists():
        if not overwrite:
            raise FileExistsError(
                "Dataset output directory already exists:\n"
                f"{output_directory}\n"
                "Use --overwrite to replace it."
            )

        shutil.rmtree(
            output_directory
        )

    samples_directory = (
        output_directory
        / "samples"
    )

    samples_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return samples_directory


def save_sample(
    *,
    output_path: Path,
    gravity_volume: np.ndarray,
    density_model: np.ndarray,
    compressed: bool,
) -> None:
    """
    Save one gravity-density pair.

    Arrays are saved as float32 to reduce storage and to match typical
    TensorFlow training input.
    """

    gravity_float32 = np.ascontiguousarray(
        gravity_volume,
        dtype=np.float32,
    )

    density_float32 = np.ascontiguousarray(
        density_model,
        dtype=np.float32,
    )

    if compressed:
        np.savez_compressed(
            output_path,
            gravity=gravity_float32,
            density=density_float32,
        )
    else:
        np.savez(
            output_path,
            gravity=gravity_float32,
            density=density_float32,
        )


def write_manifest(
    *,
    manifest_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """
    Save the sample manifest as CSV.
    """

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        writer: csv.DictWriter[str] = csv.DictWriter(
            manifest_file,
            fieldnames=list(MANIFEST_FIELD_NAMES),
        )

        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    *,
    metadata_path: Path,
    config: DatasetGenerationConfig,
    grid: MatlabCompatibleGridSpec,
    gravity_shape: tuple[int, int, int],
    generation_seconds: float,
    total_nonzero_density_cells: int,
    gravity_global_minimum: float,
    gravity_global_maximum: float,
) -> None:
    """
    Save dataset-level metadata as JSON.
    """

    metadata = {
        "dataset_type": (
            "single_positive_rectangular_body"
        ),
        "gravity_component": "Gz",
        "gravity_channel": 4,
        "gravity_unit": "mGal",
        "density_unit": "g/cm3",
        "density_array_order": (
            "density[z, y, x]"
        ),
        "gravity_array_order": (
            "gravity[z_receiver, y_receiver, x_receiver]"
        ),
        "density_shape": [
            grid.nz,
            grid.ny,
            grid.nx,
        ],
        "gravity_shape": list(
            gravity_shape
        ),
        "number_of_samples": (
            config.number_of_samples
        ),
        "random_seed": config.random_seed,
        "noise": None,
        "regularization": None,
        "configuration": config.to_dict(),
        "grid": {
            "nx": grid.nx,
            "ny": grid.ny,
            "nz": grid.nz,
            "x_min": grid.x_min,
            "x_max": grid.x_max,
            "y_min": grid.y_min,
            "y_max": grid.y_max,
            "z_min": grid.z_min,
            "z_max": grid.z_max,
            "dx": grid.dx,
            "dy": grid.dy,
            "dz": grid.dz,
        },
        "generation_seconds": (
            generation_seconds
        ),
        "mean_seconds_per_sample": (
            generation_seconds
            / config.number_of_samples
        ),
        "total_nonzero_density_cells": (
            total_nonzero_density_cells
        ),
        "gravity_global_minimum_mgal": (
            gravity_global_minimum
        ),
        "gravity_global_maximum_mgal": (
            gravity_global_maximum
        ),
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=2,
        )


def main() -> None:
    """
    Generate the baseline FWD3D gravity-volume dataset.
    """

    parser = build_argument_parser()
    arguments = parser.parse_args()

    repository_root = find_repository_root()

    config = apply_command_line_overrides(
        config=DatasetGenerationConfig(),
        arguments=arguments,
    )

    grid = MatlabCompatibleGridSpec()
    config.validate(grid)

    output_directory = (
        config.resolved_output_directory(
            repository_root
        )
    )

    samples_directory = (
        prepare_output_directory(
            output_directory=output_directory,
            overwrite=config.overwrite,
        )
    )

    random_generator = (
        np.random.default_rng(
            config.random_seed
        )
    )

    body_sampler = RectangularBodySampler(
        grid=grid,
        config=config.body,
        random_generator=random_generator,
    )

    receiver_grid = (
        config.receivers.build_receiver_grid(
            grid
        )
    )

    model_grid = (
        model_grid_from_grid_spec(
            grid
        )
    )

    forward_model = (
        FWD3DGravityForwardModel(
            model_grid=model_grid,
            receiver_grid=receiver_grid,
            channel=4,
            receiver_chunk_size=(
                config.receiver_chunk_size
            ),
        )
    )

    expected_density_shape = (
        grid.nz,
        grid.ny,
        grid.nx,
    )

    expected_gravity_shape = (
        receiver_grid.nz,
        receiver_grid.ny,
        receiver_grid.nx,
    )

    manifest_rows: list[
        dict[str, Any]
    ] = []

    total_nonzero_density_cells = 0
    gravity_global_minimum = np.inf
    gravity_global_maximum = -np.inf

    generation_start = perf_counter()

    print()
    print("Generating FWD3D baseline dataset")
    print("=" * 35)
    print(
        f"Samples: {config.number_of_samples:,}"
    )
    print(
        f"Density shape: "
        f"{expected_density_shape}"
    )
    print(
        f"Gravity shape: "
        f"{expected_gravity_shape}"
    )
    print(
        f"Output: {output_directory}"
    )
    print()

    for sample_index in range(
        config.number_of_samples
    ):
        sampled_body = body_sampler.sample(
            sample_index
        )

        density_model = build_density_model(
            grid=grid,
            case=sampled_body.specification,
        )

        if (
            density_model.shape
            != expected_density_shape
        ):
            raise RuntimeError(
                "Generated density shape is incorrect: "
                f"{density_model.shape}."
            )

        gravity_volume = (
            forward_model.calculate(
                density_model
            )
        )

        if (
            gravity_volume.shape
            != expected_gravity_shape
        ):
            raise RuntimeError(
                "Generated gravity shape is incorrect: "
                f"{gravity_volume.shape}."
            )

        if not np.all(
            np.isfinite(gravity_volume)
        ):
            raise RuntimeError(
                f"Sample {sample_index} contains invalid gravity values."
            )

        if not np.all(
            np.isfinite(density_model)
        ):
            raise RuntimeError(
                f"Sample {sample_index} contains invalid density values."
            )

        if np.count_nonzero(
            density_model
        ) == 0:
            raise RuntimeError(
                f"Sample {sample_index} contains no anomalous body."
            )

        sample_filename = (
            f"sample_{sample_index:06d}.npz"
        )

        sample_path = (
            samples_directory
            / sample_filename
        )

        save_sample(
            output_path=sample_path,
            gravity_volume=gravity_volume,
            density_model=density_model,
            compressed=config.compressed,
        )

        relative_path = sample_path.relative_to(
            output_directory
        ).as_posix()

        nonzero_density_cells = int(
            np.count_nonzero(
                density_model
            )
        )

        gravity_minimum = float(
            np.min(
                gravity_volume
            )
        )

        gravity_maximum = float(
            np.max(
                gravity_volume
            )
        )

        gravity_mean = float(
            np.mean(
                gravity_volume
            )
        )

        gravity_standard_deviation = float(
            np.std(
                gravity_volume
            )
        )

        total_nonzero_density_cells += (
            nonzero_density_cells
        )

        gravity_global_minimum = min(
            gravity_global_minimum,
            gravity_minimum,
        )

        gravity_global_maximum = max(
            gravity_global_maximum,
            gravity_maximum,
        )

        manifest_rows.append(
            sampled_body.to_manifest_row(
                sample_index=sample_index,
                relative_path=relative_path,
                gravity_minimum=gravity_minimum,
                gravity_maximum=gravity_maximum,
                gravity_mean=gravity_mean,
                gravity_standard_deviation=(
                    gravity_standard_deviation
                ),
                nonzero_density_cells=(
                    nonzero_density_cells
                ),
            )
        )

        completed = sample_index + 1

        if (
            completed == 1
            or completed % 10 == 0
            or completed
            == config.number_of_samples
        ):
            elapsed = (
                perf_counter()
                - generation_start
            )

            seconds_per_sample = (
                elapsed / completed
            )

            remaining = (
                config.number_of_samples
                - completed
            )

            estimated_remaining = (
                seconds_per_sample
                * remaining
            )

            print(
                f"[{completed:>6,}/"
                f"{config.number_of_samples:,}] "
                f"{seconds_per_sample:.3f} s/sample | "
                f"estimated remaining "
                f"{estimated_remaining:.1f} s"
            )

    generation_seconds = (
        perf_counter()
        - generation_start
    )

    manifest_path = (
        output_directory
        / "manifest.csv"
    )

    metadata_path = (
        output_directory
        / "metadata.json"
    )

    write_manifest(
        manifest_path=manifest_path,
        rows=manifest_rows,
    )

    write_metadata(
        metadata_path=metadata_path,
        config=config,
        grid=grid,
        gravity_shape=expected_gravity_shape,
        generation_seconds=(
            generation_seconds
        ),
        total_nonzero_density_cells=(
            total_nonzero_density_cells
        ),
        gravity_global_minimum=(
            gravity_global_minimum
        ),
        gravity_global_maximum=(
            gravity_global_maximum
        ),
    )

    print()
    print("Dataset generation complete")
    print("=" * 27)
    print(
        f"Generated samples: "
        f"{config.number_of_samples:,}"
    )
    print(
        f"Total time: "
        f"{generation_seconds:.2f} seconds"
    )
    print(
        f"Mean time per sample: "
        f"{generation_seconds / config.number_of_samples:.3f} seconds"
    )
    print(
        f"Samples directory: "
        f"{samples_directory}"
    )
    print(
        f"Manifest: {manifest_path}"
    )
    print(
        f"Metadata: {metadata_path}"
    )


if __name__ == "__main__":
    main()
