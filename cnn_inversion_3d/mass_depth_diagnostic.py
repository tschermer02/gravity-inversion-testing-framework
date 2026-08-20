"""Mass/depth diagnostics and E05-versus-E06 comparison reporting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def occupied_depth_extent(
    density: np.ndarray,
    *,
    threshold: float,
) -> int:
    """Return the number of occupied Z slices at the evaluation threshold."""

    if density.ndim != 3:
        raise ValueError(f"Density must be three-dimensional: {density.shape}")
    occupied_by_depth = np.any(density >= threshold, axis=(1, 2))
    return int(np.count_nonzero(occupied_by_depth))


def calculate_mass_depth_row(
    true_density: np.ndarray,
    predicted_density: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate one prediction's anomalous-mass and depth diagnostics."""

    if true_density.shape != predicted_density.shape:
        raise ValueError("True and predicted density shapes must match.")
    true_sum = float(np.sum(true_density))
    predicted_sum = float(np.sum(predicted_density))
    true_depth = occupied_depth_extent(true_density, threshold=threshold)
    predicted_depth = occupied_depth_extent(
        predicted_density, threshold=threshold
    )
    return {
        "true_density_sum": true_sum,
        "predicted_density_sum": predicted_sum,
        "mass_ratio": predicted_sum / true_sum if true_sum > 0.0 else float("nan"),
        "true_occupied_depth_extent_cells": true_depth,
        "predicted_occupied_depth_extent_cells": predicted_depth,
        "depth_extent_ratio": (
            predicted_depth / true_depth if true_depth > 0 else float("nan")
        ),
    }


def diagnose_predictions(
    prediction_directory: Path,
    *,
    label: str,
    threshold: float,
) -> tuple[Path, Path]:
    """Diagnose every saved prediction NPZ and write CSV/JSON outputs."""

    rows: list[dict[str, Any]] = []
    for path in sorted(prediction_directory.glob("*_prediction.npz")):
        with np.load(path, allow_pickle=False) as sample:
            row = calculate_mass_depth_row(
                np.asarray(sample["true_density"], dtype=np.float64),
                np.asarray(sample["predicted_density"], dtype=np.float64),
                threshold=threshold,
            )
        rows.append({"sample_id": path.name.removesuffix("_prediction.npz"), **row})
    if not rows:
        raise FileNotFoundError(
            f"No *_prediction.npz files found in {prediction_directory}"
        )
    csv_path = prediction_directory / f"{label}_mass_depth_diagnostic.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mass_ratios = np.asarray([row["mass_ratio"] for row in rows], dtype=float)
    depth_ratios = np.asarray(
        [row["depth_extent_ratio"] for row in rows], dtype=float
    )
    summary = {
        "number_of_samples": len(rows),
        "occupancy_threshold": threshold,
        "mass_proxy_definition": "sum of density values over all voxels",
        "occupied_depth_extent_definition": (
            "number of z slices containing at least one voxel at or above threshold"
        ),
        "mean_mass_ratio": float(np.nanmean(mass_ratios)),
        "median_mass_ratio": float(np.nanmedian(mass_ratios)),
        "mean_depth_extent_ratio": float(np.nanmean(depth_ratios)),
        "median_depth_extent_ratio": float(np.nanmedian(depth_ratios)),
    }
    summary_path = (
        prediction_directory / f"{label}_mass_depth_diagnostic_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, summary_path


def _mean_metric(analysis: dict[str, Any], section: str, name: str) -> float:
    """Read one mean metric from an analysis JSON document."""

    if section == "density":
        return float(analysis["overall"][name]["mean"])
    return float(
        analysis["gravity_consistency"]["metrics"][name]["mean"]
    )


def compare_experiments(
    e05_directory: Path,
    e06_directory: Path,
    output_path: Path,
    *,
    baseline_label: str = "E05",
    candidate_label: str = "E06",
) -> Path:
    """Write a neutral side-by-side controlled-experiment comparison."""

    results: dict[str, Any] = {
        "scientific_question": (
            f"How does {candidate_label} change density reconstruction, "
            f"gravity consistency, mass, and depth relative to {baseline_label}?"
        ),
        "experiments": {},
        "interpretation_guidance": [
            "Assess overlap, gravity consistency, mass ratio, and depth ratio together.",
            "Do not infer success from a single metric.",
        ],
    }
    for label, directory in (
        (baseline_label, e05_directory),
        (candidate_label, e06_directory),
    ):
        analysis = json.loads((directory / "test_analysis.json").read_text())
        diagnostic = json.loads(
            (directory / f"{label.lower()}_mass_depth_diagnostic_summary.json").read_text()
        )
        density_count = int(analysis["overall"]["number_of_samples"])
        diagnostic_count = int(diagnostic["number_of_samples"])
        if density_count != diagnostic_count:
            raise ValueError(
                f"{label} outputs are incomplete or stale: density analysis "
                f"contains {density_count} samples but the mass/depth "
                f"diagnostic contains {diagnostic_count}. Rerun prediction "
                "to completion, then rerun density analysis and the "
                "mass/depth diagnostic."
            )
        if "gravity_consistency" not in analysis:
            completed = len(
                list(
                    (directory / "gravity_consistency").glob(
                        "*/gravity_consistency_metrics.json"
                    )
                )
            )
            raise ValueError(
                f"{label} has no aggregate gravity-consistency analysis "
                f"({completed} per-sample results currently cached). "
                "Resume gravity-consistency evaluation before comparing."
            )
        gravity_count = int(
            analysis["gravity_consistency"]["number_of_samples"]
        )
        if gravity_count != density_count:
            raise ValueError(
                f"{label} gravity consistency contains {gravity_count} "
                f"samples but density analysis contains {density_count}."
            )
        results["experiments"][label] = {
            "number_of_samples": density_count,
            "density_mean_dice": _mean_metric(analysis, "density", "dice"),
            "density_mean_iou": _mean_metric(analysis, "density", "iou"),
            "density_mean_mse": _mean_metric(analysis, "density", "mse"),
            "gravity_mean_rmse": _mean_metric(
                analysis, "gravity", "gravity_rmse"
            ),
            "gravity_mean_relative_l2": _mean_metric(
                analysis, "gravity", "gravity_relative_l2"
            ),
            "gravity_mean_correlation": _mean_metric(
                analysis, "gravity", "gravity_correlation"
            ),
            **{
                key: diagnostic[key]
                for key in (
                    "mean_mass_ratio",
                    "median_mass_ratio",
                    "mean_depth_extent_ratio",
                    "median_depth_extent_ratio",
                )
            },
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return output_path


def build_argument_parser() -> argparse.ArgumentParser:
    """Build diagnostic and comparison subcommands."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("--predictions", type=Path, required=True)
    diagnose.add_argument(
        "--label", choices=("e05", "e06", "e07", "e08"), required=True
    )
    diagnose.add_argument("--threshold", type=float, default=0.1)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--e05", type=Path, required=True)
    compare.add_argument("--e06", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--baseline-label", default="E05")
    compare.add_argument("--candidate-label", default="E06")
    return parser


def main() -> None:
    """Run the selected diagnostic workflow."""

    arguments = build_argument_parser().parse_args()
    if arguments.command == "diagnose":
        csv_path, summary_path = diagnose_predictions(
            arguments.predictions.resolve(),
            label=arguments.label,
            threshold=arguments.threshold,
        )
        print(f"Diagnostic CSV: {csv_path}")
        print(f"Diagnostic summary: {summary_path}")
    else:
        output = compare_experiments(
            arguments.e05.resolve(),
            arguments.e06.resolve(),
            arguments.output.resolve(),
            baseline_label=arguments.baseline_label,
            candidate_label=arguments.candidate_label,
        )
        print(f"Comparison summary: {output}")


if __name__ == "__main__":
    main()
