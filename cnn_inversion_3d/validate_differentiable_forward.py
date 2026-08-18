"""Validate differentiable E07 Gz against the source FWD3D solver."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from cnn_inversion_3d.differentiable_gravity import DifferentiableSinglePlaneGz
from cnn_inversion_3d.single_plane_review import (
    SinglePlaneReviewConfig,
    forward_model_single_plane,
)


def comparison_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate required numerical-agreement metrics."""

    residual = predicted - true
    return {
        "rmse_mgal": float(np.sqrt(np.mean(residual**2))),
        "maximum_absolute_error_mgal": float(np.max(np.abs(residual))),
        "relative_l2": float(
            np.linalg.norm(residual.ravel()) / np.linalg.norm(true.ravel())
        ),
        "correlation": float(np.corrcoef(true.ravel(), predicted.ravel())[0, 1]),
    }


def validation_cases(
    config: SinglePlaneReviewConfig,
    dataset: Path,
    *,
    real_samples: int,
) -> list[tuple[str, np.ndarray]]:
    """Build required synthetic cases and selected real training densities."""

    cases: list[tuple[str, np.ndarray]] = []
    for name, index in (
        ("center_shallow_voxel", (0, 32, 32)),
        ("center_deep_voxel", (23, 32, 32)),
        ("edge_shallow_voxel", (0, 1, 1)),
        ("edge_deep_voxel", (23, 62, 60)),
    ):
        density = np.zeros(config.density_shape, dtype=np.float64)
        density[index] = 0.7
        cases.append((name, density))
    body = np.zeros(config.density_shape, dtype=np.float64)
    body[2:7, 20:30, 35:43] = 0.6
    cases.append(("rectangular_body", body))
    random_density = np.zeros(config.density_shape, dtype=np.float64)
    rng = np.random.default_rng(20260727)
    indices = rng.choice(random_density.size, size=40, replace=False)
    random_density.flat[indices] = rng.uniform(0.2, 1.0, size=indices.size)
    cases.append(("random_sparse_density", random_density))
    with (dataset / "train_manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))[:real_samples]
    for row in rows:
        with np.load(dataset / row["relative_path"], allow_pickle=False) as sample:
            density = np.asarray(sample["density"], dtype=np.float64)
        cases.append((f"training_{row['sample_id']}", density))
    return cases


def main() -> None:
    """Run validation and fail closed if the required tolerance is exceeded."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--real-samples", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    config = SinglePlaneReviewConfig()
    operator = DifferentiableSinglePlaneGz(
        config, calculation_dtype=tf.float64
    )
    results: list[dict[str, object]] = []
    for name, density in validation_cases(
        config, arguments.dataset.resolve(), real_samples=arguments.real_samples
    ):
        true = forward_model_single_plane(density, config=config)
        predicted = operator(
            tf.constant(density[None, ..., None], dtype=tf.float64)
        ).numpy()[0, ..., 0]
        metrics = comparison_metrics(true, predicted)
        results.append({"case": name, **metrics})
        print(f"{name}: relative L2={metrics['relative_l2']:.3e}")
    variable = tf.Variable(
        np.ones((1, *config.density_shape, 1), dtype=np.float64)
    )
    with tf.GradientTape() as tape:
        scalar = tf.reduce_sum(tf.square(operator(variable)))
    gradient = tape.gradient(scalar, variable)
    gradient_valid = bool(
        gradient is not None
        and tuple(gradient.shape) == tuple(variable.shape)
        and np.all(np.isfinite(gradient.numpy()))
    )
    maximum_relative_l2 = max(float(row["relative_l2"]) for row in results)
    report = {
        "tolerance": arguments.tolerance,
        "maximum_relative_l2": maximum_relative_l2,
        "gradient_valid": gradient_valid,
        "passed": maximum_relative_l2 <= arguments.tolerance and gradient_valid,
        "cases": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(
            "Differentiable forward validation failed; E07 training is blocked. "
            f"See {arguments.output.resolve()}"
        )
    print(f"Validation passed: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
