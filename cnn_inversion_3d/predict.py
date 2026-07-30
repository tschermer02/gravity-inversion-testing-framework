from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cnn_inversion_3d.dataset import (
    DENSITY_SHAPE,
    GRAVITY_SHAPE,
    find_repository_root,
    load_npz_sample,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run the trained 3D CNN on samples from a dataset split."
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
        "--gravity-scale",
        type=float,
        default=1.0,
        help=(
            "Gravity normalization scale used during training. "
            "Prediction must use the same value."
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "training_outputs/fwd3d_smoke_test/best_model.keras"
        ),
        help=(
            "Trained Keras model. Relative paths are interpreted "
            "from the repository root."
        ),
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default="test_manifest.csv",
        help="Manifest used for prediction.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "prediction_outputs/fwd3d_smoke_test"
        ),
        help=(
            "Prediction-output directory. Relative paths are "
            "interpreted from the repository root."
        ),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Maximum number of samples to predict.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help=(
            "Density threshold used for occupied-cell comparisons."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing prediction-output directory.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted prediction run, reusing complete "
            "per-sample outputs in the existing directory."
        ),
    )

    return parser


def resolve_path(
    *,
    repository_root: Path,
    path: Path,
) -> Path:
    """
    Resolve a path relative to the repository root.
    """

    if not path.is_absolute():
        path = repository_root / path

    return path.resolve()


def prepare_output_directory(
    *,
    output_directory: Path,
    overwrite: bool,
) -> None:
    """
    Create an empty prediction-output directory.
    """

    if output_directory.exists():
        if not overwrite:
            raise FileExistsError(
                "Prediction-output directory already exists:\n"
                f"{output_directory}\n"
                "Use --overwrite to replace it."
            )

        import shutil

        shutil.rmtree(
            output_directory
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )


def load_manifest(
    *,
    dataset_directory: Path,
    manifest_name: str,
) -> list[dict[str, str]]:
    """
    Load one dataset split manifest.
    """

    manifest_path = (
        dataset_directory
        / manifest_name
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        reader = csv.DictReader(
            manifest_file
        )

        rows = list(reader)

    if not rows:
        raise ValueError(
            f"{manifest_name} contains no samples."
        )

    return rows


def calculate_metrics(
    *,
    true_density: np.ndarray,
    predicted_density: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """
    Calculate basic density-reconstruction metrics.
    """

    difference = (
        predicted_density
        - true_density
    )

    mse = float(
        np.mean(
            difference**2
        )
    )

    mae = float(
        np.mean(
            np.abs(difference)
        )
    )

    true_mask = (
        true_density >= threshold
    )

    predicted_mask = (
        predicted_density >= threshold
    )

    intersection = int(
        np.count_nonzero(
            true_mask
            & predicted_mask
        )
    )

    union = int(
        np.count_nonzero(
            true_mask
            | predicted_mask
        )
    )

    true_count = int(
        np.count_nonzero(
            true_mask
        )
    )

    predicted_count = int(
        np.count_nonzero(
            predicted_mask
        )
    )

    iou = (
        intersection / union
        if union > 0
        else 1.0
    )

    dice_denominator = (
        true_count
        + predicted_count
    )

    dice = (
        2.0 * intersection
        / dice_denominator
        if dice_denominator > 0
        else 1.0
    )

    true_flat = true_density.ravel()
    predicted_flat = predicted_density.ravel()

    if (
        np.std(true_flat) > 0.0
        and np.std(predicted_flat) > 0.0
    ):
        correlation = float(
            np.corrcoef(
                true_flat,
                predicted_flat,
            )[0, 1]
        )
    else:
        correlation = 0.0

    return {
        "mse": mse,
        "mae": mae,
        "correlation": correlation,
        "threshold": threshold,
        "true_occupied_cells": true_count,
        "predicted_occupied_cells": predicted_count,
        "intersection_cells": intersection,
        "union_cells": union,
        "iou": float(iou),
        "dice": float(dice),
        "prediction_minimum": float(
            np.min(predicted_density)
        ),
        "prediction_maximum": float(
            np.max(predicted_density)
        ),
        "prediction_mean": float(
            np.mean(predicted_density)
        ),
    }


def density_center_indices(
    density: np.ndarray,
) -> tuple[int, int, int]:
    """
    Return the center indices of the true occupied body.
    """

    occupied = np.argwhere(
        density > 0.0
    )

    if occupied.size == 0:
        raise ValueError(
            "True density model contains no occupied cells."
        )

    minimum = occupied.min(
        axis=0
    )

    maximum = occupied.max(
        axis=0
    )

    center = (
        minimum + maximum
    ) // 2

    return (
        int(center[0]),
        int(center[1]),
        int(center[2]),
    )


def save_prediction_figure(
    *,
    gravity: np.ndarray,
    true_density: np.ndarray,
    predicted_density: np.ndarray,
    output_path: Path,
    sample_name: str,
) -> None:
    """
    Save orthogonal true and predicted density sections.
    """

    center_z, center_y, center_x = (
        density_center_indices(
            true_density
        )
    )

    density_maximum = max(
        float(
            np.max(true_density)
        ),
        float(
            np.max(predicted_density)
        ),
    )

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(17.0, 15.0),
    )

    true_xy = true_density[
        center_z,
        :,
        :,
    ]

    predicted_xy = predicted_density[
        center_z,
        :,
        :,
    ]

    difference_xy = (
        predicted_xy
        - true_xy
    )

    true_xz = true_density[
        :,
        center_y,
        :,
    ]

    predicted_xz = predicted_density[
        :,
        center_y,
        :,
    ]

    difference_xz = (
        predicted_xz
        - true_xz
    )

    true_yz = true_density[
        :,
        :,
        center_x,
    ]

    predicted_yz = predicted_density[
        :,
        :,
        center_x,
    ]

    difference_yz = (
        predicted_yz
        - true_yz
    )

    density_sections = (
        (
            true_xy,
            predicted_xy,
            difference_xy,
            "Horizontal X-Y",
        ),
        (
            true_xz,
            predicted_xz,
            difference_xz,
            "Vertical X-Z",
        ),
        (
            true_yz,
            predicted_yz,
            difference_yz,
            "Vertical Y-Z",
        ),
    )

    difference_limit = max(
        abs(
            float(
                np.min(
                    predicted_density
                    - true_density
                )
            )
        ),
        abs(
            float(
                np.max(
                    predicted_density
                    - true_density
                )
            )
        ),
    )

    for row_index, (
        true_section,
        predicted_section,
        difference_section,
        section_name,
    ) in enumerate(
        density_sections
    ):
        true_image = axes[
            row_index,
            0,
        ].imshow(
            true_section,
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=density_maximum,
        )

        axes[row_index, 0].set_title(
            f"True density\n{section_name}"
        )

        figure.colorbar(
            true_image,
            ax=axes[row_index, 0],
            label="Density [g/cm³]",
        )

        predicted_image = axes[
            row_index,
            1,
        ].imshow(
            predicted_section,
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=density_maximum,
        )

        axes[row_index, 1].set_title(
            f"Predicted density\n{section_name}"
        )

        figure.colorbar(
            predicted_image,
            ax=axes[row_index, 1],
            label="Density [g/cm³]",
        )

        difference_image = axes[
            row_index,
            2,
        ].imshow(
            difference_section,
            origin="lower",
            aspect="auto",
            vmin=-difference_limit,
            vmax=difference_limit,
            cmap="coolwarm",
        )

        axes[row_index, 2].set_title(
            f"Prediction minus truth\n{section_name}"
        )

        figure.colorbar(
            difference_image,
            ax=axes[row_index, 2],
            label="Density difference [g/cm³]",
        )

    figure.suptitle(
        f"3D density reconstruction: {sample_name}",
        fontsize=17,
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
    Predict test density models and save results.
    """

    parser = build_argument_parser()
    arguments = parser.parse_args()

    if arguments.samples < 1:
        raise ValueError(
            "--samples must be at least one."
        )

    if arguments.threshold <= 0.0:
        raise ValueError(
            "--threshold must be greater than zero."
        )

    if arguments.gravity_scale <= 0.0:
        raise ValueError(
            "--gravity-scale must be greater than zero."
        )

    if arguments.overwrite and arguments.resume:
        raise ValueError(
            "--overwrite and --resume cannot be used together."
        )

    repository_root = (
        find_repository_root()
    )

    dataset_directory = resolve_path(
        repository_root=repository_root,
        path=arguments.dataset,
    )

    model_path = resolve_path(
        repository_root=repository_root,
        path=arguments.model,
    )

    output_directory = resolve_path(
        repository_root=repository_root,
        path=arguments.output,
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    if arguments.resume:
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
    else:
        prepare_output_directory(
            output_directory=output_directory,
            overwrite=arguments.overwrite,
        )

    manifest_rows = load_manifest(
        dataset_directory=dataset_directory,
        manifest_name=arguments.manifest,
    )

    selected_rows = manifest_rows[
        :arguments.samples
    ]

    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
    )

    results: list[dict[str, Any]] = []
    results_path = (
        output_directory
        / "prediction_metrics.json"
    )

    print()
    print("Predicting 3D density models")
    print("=" * 28)
    print(
        f"Model: {model_path}"
    )
    print(
        f"Manifest: {arguments.manifest}"
    )
    print(
        f"Samples: {len(selected_rows)}"
    )
    print()

    for row_index, row in enumerate(
        selected_rows
    ):
        sample_path = (
            dataset_directory
            / row["relative_path"]
        )

        gravity, true_density = (
            load_npz_sample(
                sample_path
            )
        )

        if gravity.shape != GRAVITY_SHAPE:
            raise RuntimeError(
                f"Unexpected gravity shape: "
                f"{gravity.shape}"
            )

        if true_density.shape != DENSITY_SHAPE:
            raise RuntimeError(
                f"Unexpected density shape: "
                f"{true_density.shape}"
            )

        true_density_3d = (
            true_density[..., 0]
        )

        sample_name = sample_path.stem

        prediction_path = (
            output_directory
            / f"{sample_name}_prediction.npz"
        )

        if arguments.resume and prediction_path.exists():
            with np.load(
                prediction_path,
                allow_pickle=False,
            ) as prediction_file:
                predicted_density_3d = np.asarray(
                    prediction_file[
                        "predicted_density"
                    ],
                    dtype=np.float32,
                )
        else:
            normalized_gravity = (
                gravity
                / np.float32(
                    arguments.gravity_scale
                )
            )
            gravity_batch = normalized_gravity[
                np.newaxis,
                ...,
            ]
            predicted_batch = model(
                gravity_batch,
                training=False,
            )
            predicted_density = np.asarray(
                predicted_batch[0],
                dtype=np.float32,
            )
            predicted_density_3d = (
                predicted_density[..., 0]
            )

            np.savez_compressed(
                prediction_path,
                gravity=np.asarray(
                    gravity[..., 0],
                    dtype=np.float32,
                ),
                true_density=np.asarray(
                    true_density_3d,
                    dtype=np.float32,
                ),
                predicted_density=np.asarray(
                    predicted_density_3d,
                    dtype=np.float32,
                ),
            )

        metrics = calculate_metrics(
            true_density=true_density_3d,
            predicted_density=(
                predicted_density_3d
            ),
            threshold=arguments.threshold,
        )

        figure_path = (
            output_directory
            / f"{sample_name}_comparison.png"
        )

        figure_was_cached = (
            arguments.resume
            and figure_path.exists()
        )

        if not figure_was_cached:
            save_prediction_figure(
                gravity=gravity[..., 0],
                true_density=true_density_3d,
                predicted_density=(
                    predicted_density_3d
                ),
                output_path=figure_path,
                sample_name=sample_name,
            )

        result = {
            "sample_index": row_index,
            "sample_path": (
                row["relative_path"]
            ),
            "prediction_path": (
                prediction_path.name
            ),
            "figure_path": figure_path.name,
            **metrics,
        }

        results.append(result)

        with results_path.open(
            "w",
            encoding="utf-8",
        ) as results_file:
            json.dump(
                results,
                results_file,
                indent=2,
            )

        print(
            f"{sample_name}"
            f"{' [cached]' if figure_was_cached else ''}: "
            f"MSE={metrics['mse']:.6e}, "
            f"IoU={metrics['iou']:.4f}, "
            f"Dice={metrics['dice']:.4f}, "
            f"predicted max="
            f"{metrics['prediction_maximum']:.4f}"
        )

    print()
    print(
        f"Prediction outputs: "
        f"{output_directory}"
    )
    print(
        f"Metrics: {results_path}"
    )


if __name__ == "__main__":
    main()
