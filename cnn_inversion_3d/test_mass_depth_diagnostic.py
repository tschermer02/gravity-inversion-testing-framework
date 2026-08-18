"""Tests for E05/E06 mass and vertical-extent diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cnn_inversion_3d.mass_depth_diagnostic import (
    calculate_mass_depth_row,
    compare_experiments,
    diagnose_predictions,
)


def test_mass_and_depth_ratios_use_requested_threshold() -> None:
    """Verify diagnostic definitions on an exact synthetic example."""

    true = np.zeros((4, 2, 2))
    predicted = np.zeros_like(true)
    true[1:3] = 0.5
    predicted[:] = 0.5
    row = calculate_mass_depth_row(true, predicted, threshold=0.1)
    assert row["mass_ratio"] == 2.0
    assert row["true_occupied_depth_extent_cells"] == 2
    assert row["predicted_occupied_depth_extent_cells"] == 4
    assert row["depth_extent_ratio"] == 2.0


def test_prediction_diagnostic_writes_csv_and_summary(tmp_path: Path) -> None:
    """Verify E05/E06 diagnostic deliverables are generated."""

    true = np.zeros((3, 2, 2))
    true[1] = 0.5
    np.savez_compressed(
        tmp_path / "sample_000001_prediction.npz",
        true_density=true,
        predicted_density=true,
    )
    csv_path, summary_path = diagnose_predictions(
        tmp_path, label="e06", threshold=0.1
    )
    summary = json.loads(summary_path.read_text())
    assert csv_path.exists()
    assert summary["mean_mass_ratio"] == 1.0
    assert summary["mean_depth_extent_ratio"] == 1.0


def test_e05_e06_comparison_summary(tmp_path: Path) -> None:
    """Verify density, gravity, mass, and depth metrics are compared."""

    def write_experiment(directory: Path, label: str, value: float) -> None:
        directory.mkdir()
        analysis = {
            "overall": {
                "number_of_samples": 1,
                **{
                    name: {"mean": value}
                    for name in ("dice", "iou", "mse")
                },
            },
            "gravity_consistency": {
                "number_of_samples": 1,
                "metrics": {
                    name: {"mean": value}
                    for name in (
                        "gravity_rmse",
                        "gravity_relative_l2",
                        "gravity_correlation",
                    )
                }
            },
        }
        diagnostic = {
            "number_of_samples": 1,
            **{
                name: value
                for name in (
                    "mean_mass_ratio",
                    "median_mass_ratio",
                    "mean_depth_extent_ratio",
                    "median_depth_extent_ratio",
                )
            },
        }
        (directory / "test_analysis.json").write_text(json.dumps(analysis))
        (directory / f"{label}_mass_depth_diagnostic_summary.json").write_text(
            json.dumps(diagnostic)
        )

    e05 = tmp_path / "e05"
    e06 = tmp_path / "e06"
    write_experiment(e05, "e05", 1.0)
    write_experiment(e06, "e06", 2.0)
    output = compare_experiments(e05, e06, tmp_path / "comparison.json")
    comparison = json.loads(output.read_text())
    assert comparison["experiments"]["E05"]["density_mean_dice"] == 1.0
    assert comparison["experiments"]["E06"]["gravity_mean_rmse"] == 2.0
