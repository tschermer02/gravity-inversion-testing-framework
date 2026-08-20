"""Paired analysis-only comparison of canonical E05, E06, and E07."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from cnn_inversion_3d.dataset import find_repository_root
from cnn_inversion_3d.gravity_consistency import build_cnn_forward_model_context
from cnn_inversion_3d.single_plane_review import (
    SinglePlaneReviewConfig,
)


EXPERIMENTS = {
    "E05": "E05_canonical_single_plane_baseline_bf8",
    "E06": "E06_canonical_single_plane_learned_depth_seed_bf8",
    "E07": "E07_canonical_single_plane_learned_depth_seed_physics_bf8",
}
THRESHOLD = 0.1


def extract_occupied_geometry(
    density: np.ndarray,
    *,
    threshold: float = THRESHOLD,
    config: SinglePlaneReviewConfig | None = None,
) -> dict[str, float]:
    """Extract occupied extents from ``density[z,y,x]`` in physical meters.

    Parameters
    ----------
    density
        Three-dimensional density array.
    threshold
        Inclusive occupancy threshold.
    config
        Canonical geometry configuration.

    Returns
    -------
    dict
        Physical top, bottom, thickness, and horizontal widths. Values are
        NaN when the thresholded prediction is empty.
    """

    geometry = config or SinglePlaneReviewConfig()
    if density.shape != geometry.density_shape:
        raise ValueError(
            f"Expected density shape {geometry.density_shape}, got {density.shape}."
        )
    occupied = np.argwhere(density >= threshold)
    if occupied.size == 0:
        return {
            "predicted_top_depth_m": float("nan"),
            "predicted_bottom_depth_m": float("nan"),
            "predicted_thickness_m": float("nan"),
            "predicted_width_x_m": float("nan"),
            "predicted_width_y_m": float("nan"),
        }
    minimum = occupied.min(axis=0)
    maximum = occupied.max(axis=0) + 1
    return {
        "predicted_top_depth_m": float(minimum[0] * geometry.dz_m),
        "predicted_bottom_depth_m": float(maximum[0] * geometry.dz_m),
        "predicted_thickness_m": float((maximum[0] - minimum[0]) * geometry.dz_m),
        "predicted_width_x_m": float((maximum[2] - minimum[2]) * geometry.dx_m),
        "predicted_width_y_m": float((maximum[1] - minimum[1]) * geometry.dy_m),
    }


def validate_sample_alignment(
    manifest_ids: Iterable[str], experiment_ids: dict[str, Iterable[str]]
) -> list[str]:
    """Return canonical paired order or fail on missing, extra, or duplicate IDs."""

    expected = list(manifest_ids)
    if len(expected) != 100 or len(set(expected)) != 100:
        raise ValueError("Canonical test manifest must contain 100 unique sample IDs.")
    expected_set = set(expected)
    for label, values in experiment_ids.items():
        ids = list(values)
        if len(ids) != 100 or len(set(ids)) != 100:
            raise ValueError(f"{label} must contain 100 unique sample IDs.")
        if set(ids) != expected_set:
            missing = sorted(expected_set - set(ids))
            extra = sorted(set(ids) - expected_set)
            raise ValueError(
                f"{label} sample alignment differs: missing={missing}, extra={extra}."
            )
    return expected


def _sample_id_from_metric(row: dict[str, Any]) -> str:
    return Path(str(row["sample_path"])).stem


def _pearson(x: Iterable[float], y: Iterable[float]) -> float:
    xv = np.asarray(list(x), dtype=float)
    yv = np.asarray(list(y), dtype=float)
    valid = np.isfinite(xv) & np.isfinite(yv)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    if np.std(xv[valid]) == 0.0 or np.std(yv[valid]) == 0.0:
        return float("nan")
    return float(np.corrcoef(xv[valid], yv[valid])[0, 1])


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.nanmean([float(row[key]) for row in rows]))


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.nanmedian([float(row[key]) for row in rows]))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2), encoding="utf-8")


def _gravity_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - true
    true_flat = true.ravel()
    predicted_flat = predicted.ravel()
    return {
        "gravity_rmse_mgal": float(np.sqrt(np.mean(residual**2))),
        "gravity_mae_mgal": float(np.mean(np.abs(residual))),
        "gravity_relative_l2": float(
            np.linalg.norm(residual) / max(np.linalg.norm(true), 1.0e-12)
        ),
        "gravity_correlation": _pearson(true_flat, predicted_flat),
        "gravity_maximum_absolute_residual_mgal": float(np.max(np.abs(residual))),
    }


def load_experiment_rows(
    *,
    label: str,
    prediction_directory: Path,
    manifest: dict[str, dict[str, str]],
    ordered_ids: list[str],
    config: SinglePlaneReviewConfig,
    forward_model: Any,
) -> list[dict[str, Any]]:
    """Load paired prediction arrays and calculate all requested metrics."""

    metrics = json.loads(
        (prediction_directory / "prediction_metrics.json").read_text(encoding="utf-8")
    )
    metric_map = {_sample_id_from_metric(row): row for row in metrics}
    rows: list[dict[str, Any]] = []
    for sample_id in ordered_ids:
        source = manifest[sample_id]
        metric = metric_map[sample_id]
        prediction_path = prediction_directory / str(metric["prediction_path"])
        with np.load(prediction_path, allow_pickle=False) as sample:
            true_density = np.asarray(sample["true_density"], dtype=np.float64)
            predicted_density = np.asarray(sample["predicted_density"], dtype=np.float64)
            true_gravity = np.asarray(sample["gravity"], dtype=np.float64)
        predicted_gravity_volume = np.asarray(
            forward_model.calculate(predicted_density), dtype=np.float64
        )
        predicted_gravity = predicted_gravity_volume[0]
        occupied = predicted_density >= THRESHOLD
        true_occupied = true_density > 0.0
        predicted_cells = int(np.count_nonzero(occupied))
        true_cells = int(np.count_nonzero(true_occupied))
        true_mass = float(np.sum(true_density))
        predicted_mass = float(np.sum(predicted_density))
        geometry = extract_occupied_geometry(
            predicted_density, threshold=THRESHOLD, config=config
        )
        top_error = abs(geometry["predicted_top_depth_m"] - float(source["top_depth_m"]))
        bottom_error = abs(
            geometry["predicted_bottom_depth_m"] - float(source["bottom_depth_m"])
        )
        thickness_error = abs(
            geometry["predicted_thickness_m"] - float(source["thickness_z_m"])
        )
        row: dict[str, Any] = {
            "experiment": label,
            "sample_id": sample_id,
            "mse": float(metric["mse"]),
            "mae": float(metric["mae"]),
            "iou": float(metric["iou"]),
            "dice": float(metric["dice"]),
            "prediction_maximum": float(metric["prediction_maximum"]),
            "prediction_mean": float(metric["prediction_mean"]),
            "body_prediction_mean": float(np.mean(predicted_density[true_occupied])),
            "background_prediction_mean": float(np.mean(predicted_density[~true_occupied])),
            "true_occupied_cells": true_cells,
            "predicted_occupied_cells": predicted_cells,
            "occupied_volume_ratio": predicted_cells / true_cells,
            "occupied_volume_ratio_error": abs(predicted_cells / true_cells - 1.0),
            "true_mass_proxy": true_mass,
            "predicted_mass_proxy": predicted_mass,
            "mass_ratio": predicted_mass / true_mass,
            "mass_ratio_error": abs(predicted_mass / true_mass - 1.0),
            "empty_prediction": predicted_cells == 0,
            **geometry,
            "top_depth_absolute_error_m": top_error,
            "bottom_depth_absolute_error_m": bottom_error,
            "width_x_absolute_error_m": abs(
                geometry["predicted_width_x_m"] - float(source["width_x_m"])
            ),
            "width_y_absolute_error_m": abs(
                geometry["predicted_width_y_m"] - float(source["width_y_m"])
            ),
            "thickness_absolute_error_m": thickness_error,
            "true_top_depth_m": float(source["top_depth_m"]),
            "true_bottom_depth_m": float(source["bottom_depth_m"]),
            "true_thickness_m": float(source["thickness_z_m"]),
            "true_width_x_m": float(source["width_x_m"]),
            "true_width_y_m": float(source["width_y_m"]),
            "true_body_volume_cells": true_cells,
            "density_contrast_g_cm3": float(source["density_contrast"]),
            "horizontal_distance_from_center_m": float(
                np.hypot(
                    float(source["center_x_m"]) - 320.0,
                    float(source["center_y_m"]) - 320.0,
                )
            ),
            "true_gravity_maximum_mgal": float(np.max(true_gravity)),
            "true_gravity_standard_deviation_mgal": float(np.std(true_gravity)),
            "comparison_figure_path": str(
                prediction_directory / str(metric["figure_path"])
            ),
            **_gravity_metrics(true_gravity, predicted_gravity),
        }
        rows.append(row)
    return rows


def overall_row(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one experiment over all held-out samples."""

    return {
        "experiment": label,
        "samples": len(rows),
        "empty_predictions": sum(bool(row["empty_prediction"]) for row in rows),
        **{
            f"{stat}_{metric}": function(rows, metric)
            for metric in ("mse", "mae", "iou", "dice")
            for stat, function in (("mean", _mean), ("median", _median))
        },
        "mean_prediction_maximum": _mean(rows, "prediction_maximum"),
        "mean_prediction_mean": _mean(rows, "prediction_mean"),
        "mean_body_prediction_mean": _mean(rows, "body_prediction_mean"),
        "mean_background_prediction_mean": _mean(rows, "background_prediction_mean"),
        "mean_predicted_occupied_cells": _mean(rows, "predicted_occupied_cells"),
        "mean_true_occupied_cells": _mean(rows, "true_occupied_cells"),
        "mean_occupied_volume_ratio": _mean(rows, "occupied_volume_ratio"),
        "median_occupied_volume_ratio": _median(rows, "occupied_volume_ratio"),
        "mean_mass_ratio": _mean(rows, "mass_ratio"),
        "median_mass_ratio": _median(rows, "mass_ratio"),
        "mean_absolute_top_depth_error_m": _mean(rows, "top_depth_absolute_error_m"),
        "median_absolute_top_depth_error_m": _median(rows, "top_depth_absolute_error_m"),
        "mean_absolute_bottom_depth_error_m": _mean(rows, "bottom_depth_absolute_error_m"),
        "median_absolute_bottom_depth_error_m": _median(rows, "bottom_depth_absolute_error_m"),
        "mean_absolute_width_x_error_m": _mean(rows, "width_x_absolute_error_m"),
        "mean_absolute_width_y_error_m": _mean(rows, "width_y_absolute_error_m"),
        "mean_absolute_thickness_error_m": _mean(rows, "thickness_absolute_error_m"),
        "mean_gravity_rmse_mgal": _mean(rows, "gravity_rmse_mgal"),
        "median_gravity_rmse_mgal": _median(rows, "gravity_rmse_mgal"),
        "mean_gravity_correlation": _mean(rows, "gravity_correlation"),
        "median_gravity_correlation": _median(rows, "gravity_correlation"),
        "mean_gravity_relative_l2": _mean(rows, "gravity_relative_l2"),
        "median_gravity_relative_l2": _median(rows, "gravity_relative_l2"),
    }


GROUP_METRICS = (
    "dice", "iou", "mse", "occupied_volume_ratio", "mass_ratio",
    "top_depth_absolute_error_m", "bottom_depth_absolute_error_m",
    "thickness_absolute_error_m", "gravity_rmse_mgal",
)


def grouped_rows(
    all_rows: dict[str, list[dict[str, Any]]],
    group_name: str,
    classify: Any,
) -> list[dict[str, Any]]:
    """Calculate requested means for one true-property grouping."""

    output: list[dict[str, Any]] = []
    for label, rows in all_rows.items():
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(classify(row)), []).append(row)
        for subgroup, values in groups.items():
            output.append({
                "experiment": label,
                "grouping": group_name,
                "subgroup": subgroup,
                "samples": len(values),
                **{f"mean_{metric}": _mean(values, metric) for metric in GROUP_METRICS},
            })
    return output


def paired_rows(all_rows: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build wide paired sample table and improvement percentages."""

    maps = {label: {row["sample_id"]: row for row in rows} for label, rows in all_rows.items()}
    metrics = {
        "dice": True, "iou": True, "mse": False,
        "occupied_volume_ratio_error": False, "mass_ratio_error": False,
        "top_depth_absolute_error_m": False,
        "bottom_depth_absolute_error_m": False,
        "thickness_absolute_error_m": False, "gravity_rmse_mgal": False,
    }
    output: list[dict[str, Any]] = []
    for sample_id in maps["E05"]:
        row: dict[str, Any] = {"sample_id": sample_id}
        for label in EXPERIMENTS:
            for metric in metrics:
                row[f"{label}_{metric}"] = maps[label][sample_id][metric]
        for before, after in (("E05", "E06"), ("E06", "E07")):
            for metric, higher_better in metrics.items():
                delta = maps[after][sample_id][metric] - maps[before][sample_id][metric]
                row[f"{after}_minus_{before}_{metric}"] = delta
                row[f"{after}_beats_{before}_{metric}"] = (
                    delta > 0.0 if higher_better else delta < 0.0
                )
        output.append(row)
    summary: dict[str, Any] = {}
    for before, after in (("E05", "E06"), ("E06", "E07")):
        transition = f"{before}_to_{after}"
        summary[transition] = {}
        for metric, higher_better in metrics.items():
            deltas = np.asarray(
                [row[f"{after}_minus_{before}_{metric}"] for row in output], dtype=float
            )
            beats = deltas > 0.0 if higher_better else deltas < 0.0
            summary[transition][metric] = {
                "delta_definition": f"{after} minus {before}",
                "higher_is_better": higher_better,
                "mean_delta": float(np.nanmean(deltas)),
                "median_delta": float(np.nanmedian(deltas)),
                "percent_samples_improved": float(100.0 * np.mean(beats)),
            }
    return output, summary


def correlation_rows(all_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Calculate property-performance and property-error correlations."""

    pairs = (
        ("dice", "true_top_depth_m"), ("dice", "true_thickness_m"),
        ("dice", "true_body_volume_cells"), ("dice", "density_contrast_g_cm3"),
        ("dice", "horizontal_distance_from_center_m"),
        ("dice", "true_gravity_maximum_mgal"),
        ("dice", "true_gravity_standard_deviation_mgal"),
        ("top_depth_absolute_error_m", "true_top_depth_m"),
        ("thickness_absolute_error_m", "true_thickness_m"),
        ("occupied_volume_ratio", "true_body_volume_cells"),
        ("mass_ratio", "true_body_volume_cells"),
    )
    return [
        {
            "experiment": label,
            "outcome": outcome,
            "true_property": property_name,
            "pearson_r": _pearson(
                (row[outcome] for row in rows),
                (row[property_name] for row in rows),
            ),
            "samples": len(rows),
        }
        for label, rows in all_rows.items()
        for outcome, property_name in pairs
    ]


def _plot_outputs(
    all_rows: dict[str, list[dict[str, Any]]], figures: Path
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"E05": "#4c78a8", "E06": "#f58518", "E07": "#54a24b"}
    metrics = (
        ("mse", "MSE"), ("dice", "Dice"), ("iou", "IoU"),
        ("occupied_volume_ratio", "Occupied-volume ratio"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(13, 4), constrained_layout=True)
    for axis, (key, title) in zip(axes.ravel(), metrics):
        axis.boxplot(
            [[row[key] for row in all_rows[label]] for label in EXPERIMENTS],
            tick_labels=list(EXPERIMENTS), showfliers=False,
        )
        axis.set_title(title)
    figure.savefig(figures / "overall_metrics.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for label, rows in all_rows.items():
        axis.scatter(
            [row["true_top_depth_m"] for row in rows],
            [row["dice"] for row in rows], s=18, alpha=0.6,
            label=label, color=colors[label],
        )
    axis.set(xlabel="True top depth (m)", ylabel="Dice")
    axis.legend()
    figure.savefig(figures / "dice_vs_depth.png", dpi=200)
    plt.close(figure)

    for true_key, predicted_key, xlabel, ylabel, filename in (
        ("true_occupied_cells", "predicted_occupied_cells", "True occupied cells", "Predicted occupied cells", "volume_recovery.png"),
    ):
        figure, axis = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
        maximum = 0.0
        for label, rows in all_rows.items():
            x = [row[true_key] for row in rows]
            y = [row[predicted_key] for row in rows]
            maximum = max(maximum, max(x), max(y))
            axis.scatter(x, y, s=18, alpha=0.55, label=label, color=colors[label])
        axis.plot((0, maximum), (0, maximum), "k--", linewidth=1)
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.legend()
        figure.savefig(figures / filename, dpi=200)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    depth_metrics = (
        ("true_top_depth_m", "predicted_top_depth_m", "Top depth (m)"),
        ("true_thickness_m", "predicted_thickness_m", "Thickness (m)"),
    )
    for axis, (true_key, predicted_key, title) in zip(axes, depth_metrics):
        for label, rows in all_rows.items():
            axis.scatter([row[true_key] for row in rows], [row[predicted_key] for row in rows],
                         s=16, alpha=0.55, label=label, color=colors[label])
        limit = 240.0 if "depth" in true_key else 100.0
        axis.plot((0, limit), (0, limit), "k--", linewidth=1)
        axis.set(xlabel=f"True {title}", ylabel=f"Predicted {title}")
    axes[0].legend()
    figure.savefig(figures / "depth_recovery.png", dpi=200)
    plt.close(figure)


def _representatives(
    all_rows: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    maps = {label: {row["sample_id"]: row for row in rows} for label, rows in all_rows.items()}
    ids = list(maps["E07"])
    e06_gain = {sid: maps["E06"][sid]["dice"] - maps["E05"][sid]["dice"] for sid in ids}
    e07_gain = {sid: maps["E07"][sid]["dice"] - maps["E06"][sid]["dice"] for sid in ids}
    sorted_e07 = sorted(ids, key=lambda sid: maps["E07"][sid]["dice"])
    choices = (
        ("largest_E06_over_E05_Dice_improvement", max(e06_gain, key=e06_gain.get)),
        ("largest_E07_over_E06_Dice_improvement", max(e07_gain, key=e07_gain.get)),
        ("largest_E07_Dice_worsening", min(e07_gain, key=e07_gain.get)),
        ("median_E07_Dice", sorted_e07[len(sorted_e07) // 2]),
        ("best_E07_Dice", sorted_e07[-1]),
        ("worst_E07_Dice", sorted_e07[0]),
    )
    return [{
        "category": category, "sample_id": sid,
        "E05_dice": maps["E05"][sid]["dice"],
        "E06_dice": maps["E06"][sid]["dice"],
        "E07_dice": maps["E07"][sid]["dice"],
        "E06_minus_E05_dice": e06_gain[sid],
        "E07_minus_E06_dice": e07_gain[sid],
        "E05_figure": maps["E05"][sid]["comparison_figure_path"],
        "E06_figure": maps["E06"][sid]["comparison_figure_path"],
        "E07_figure": maps["E07"][sid]["comparison_figure_path"],
    } for category, sid in choices]


def _percent_change(new: float, old: float) -> float:
    return 100.0 * (new - old) / old if old != 0.0 else float("nan")


def build_summary(
    overall: list[dict[str, Any]], paired: dict[str, Any], correlations: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build quantitative answers and one E08 recommendation."""

    values = {row["experiment"]: row for row in overall}
    transitions: dict[str, Any] = {}
    for before, after in (("E05", "E06"), ("E06", "E07")):
        transitions[f"{before}_to_{after}"] = {
            metric: {
                "absolute_change": values[after][metric] - values[before][metric],
                "percent_change": _percent_change(values[after][metric], values[before][metric]),
            }
            for metric in (
                "mean_dice", "mean_iou", "mean_mse",
                "mean_occupied_volume_ratio", "mean_mass_ratio",
                "mean_absolute_top_depth_error_m",
                "mean_absolute_bottom_depth_error_m",
                "mean_absolute_thickness_error_m", "mean_gravity_rmse_mgal",
                "mean_body_prediction_mean", "mean_background_prediction_mean",
            )
        }
    e07_dice_correlations = [
        row for row in correlations if row["experiment"] == "E07" and row["outcome"] == "dice"
    ]
    strongest_dice = max(e07_dice_correlations, key=lambda row: abs(row["pearson_r"]))
    ratio_correlations = [
        row for row in correlations
        if row["experiment"] == "E07" and row["outcome"] in {"occupied_volume_ratio", "mass_ratio"}
    ]
    strongest_ratio = max(ratio_correlations, key=lambda row: abs(row["pearson_r"]))
    remaining = {
        "occupied_volume_ratio_excess": values["E07"]["mean_occupied_volume_ratio"] - 1.0,
        "mass_ratio_excess": values["E07"]["mean_mass_ratio"] - 1.0,
        "top_depth_error_m": values["E07"]["mean_absolute_top_depth_error_m"],
        "bottom_depth_error_m": values["E07"]["mean_absolute_bottom_depth_error_m"],
        "thickness_error_m": values["E07"]["mean_absolute_thickness_error_m"],
    }
    dominant = max(remaining, key=lambda key: abs(remaining[key]))
    return {
        "controls": {
            "E05": "repeated identical depth seed; BalancedDensityMSE",
            "E06": "learned depth-specific seed; BalancedDensityMSE",
            "E07": "exact E06 architecture plus global-normalized gravity MSE, weight 0.001",
        },
        "transitions": transitions,
        "paired_improvement": paired,
        "questions": {
            "learned_depth_seeding_E06_vs_E05": transitions["E05_to_E06"],
            "physics_loss_E07_vs_E06": transitions["E06_to_E07"],
            "E07_body_vs_background": {
                "body_prediction_mean_change": transitions["E06_to_E07"]["mean_body_prediction_mean"],
                "background_prediction_mean_change": transitions["E06_to_E07"]["mean_background_prediction_mean"],
            },
            "E07_reduces_occupied_volume_overprediction": values["E07"]["mean_occupied_volume_ratio"] < values["E06"]["mean_occupied_volume_ratio"],
            "E07_reduces_mass_overprediction": values["E07"]["mean_mass_ratio"] < values["E06"]["mean_mass_ratio"],
            "E07_improves_top_depth": values["E07"]["mean_absolute_top_depth_error_m"] < values["E06"]["mean_absolute_top_depth_error_m"],
            "E07_improves_bottom_depth": values["E07"]["mean_absolute_bottom_depth_error_m"] < values["E06"]["mean_absolute_bottom_depth_error_m"],
            "E07_improves_thickness": values["E07"]["mean_absolute_thickness_error_m"] < values["E06"]["mean_absolute_thickness_error_m"],
            "strongest_true_property_predicting_poor_Dice": strongest_dice,
            "strongest_true_property_predicting_volume_or_mass_ratio": strongest_ratio,
            "dominant_remaining_failure_indicator": {dominant: remaining[dominant]},
        },
        "E08_recommendation": (
            "Test one explicit occupied-volume regularization variable while keeping "
            "the E07 architecture, data, and gravity term fixed."
            if "ratio" in dominant
            else "Test one vertical-extent constraint variable while keeping the E07 "
            "architecture, data, and gravity term fixed."
        ),
    }


def _readme(overall: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    fields = (
        "mean_dice", "mean_iou", "mean_mse", "mean_occupied_volume_ratio",
        "mean_mass_ratio", "mean_absolute_top_depth_error_m",
        "mean_absolute_bottom_depth_error_m", "mean_absolute_thickness_error_m",
        "mean_gravity_rmse_mgal",
    )
    lines = [
        "# E05 vs E06 vs E07 Final Comparison", "", "## Overall results", "",
        "| Experiment | " + " | ".join(fields) + " |",
        "|---|" + "---:|" * len(fields),
    ]
    for row in overall:
        lines.append(
            f"| {row['experiment']} | "
            + " | ".join(f"{float(row[field]):.6g}" for field in fields) + " |"
        )
    lines.extend([
        "", "## E05", "",
        "Repeated identical depth seed with BalancedDensityMSE.",
        "", "## E06", "",
        "Uses learned depth-specific latent features. Relative to E05:",
    ])
    for metric, values in summary["transitions"]["E05_to_E06"].items():
        lines.append(f"- {metric}: {values['absolute_change']:+.6g} ({values['percent_change']:+.2f}%).")
    lines.extend([
        "", "## E07", "",
        "Keeps the exact E06 architecture and adds globally normalized gravity consistency with weight 0.001. Relative to E06:",
    ])
    for metric, values in summary["transitions"]["E06_to_E07"].items():
        lines.append(f"- {metric}: {values['absolute_change']:+.6g} ({values['percent_change']:+.2f}%).")
    lines.extend([
        "", "## Main remaining problem", "",
        "The largest remaining quantitative indicator is "
        f"`{next(iter(summary['questions']['dominant_remaining_failure_indicator']))}`. "
        "Interpretation uses density, geometry, mass/volume, and trusted FWD3D gravity metrics together.",
        "", "## Suggested E08", "", summary["E08_recommendation"], "",
    ])
    return "\n".join(lines)


def main() -> None:
    """Run the complete paired comparison without modifying experiment outputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/canonical_single_plane_train2000"))
    parser.add_argument("--output", type=Path, default=Path("analysis_outputs/E05_E06_E07_final_comparison"))
    args = parser.parse_args()
    root = find_repository_root()
    dataset = (root / args.dataset).resolve() if not args.dataset.is_absolute() else args.dataset.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = SinglePlaneReviewConfig()
    forward_context = build_cnn_forward_model_context(dataset / "metadata.json")

    with (dataset / "test_manifest.csv").open(encoding="utf-8", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    manifest = {row["sample_id"]: row for row in manifest_rows}
    prediction_directories = {
        label: root / "prediction_outputs" / name for label, name in EXPERIMENTS.items()
    }
    experiment_ids: dict[str, list[str]] = {}
    for label, directory in prediction_directories.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing prediction directory: {directory}")
        metrics_path = directory / "prediction_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing prediction metrics: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        experiment_ids[label] = [_sample_id_from_metric(row) for row in metrics]
        prediction_files = list(directory.glob("*_prediction.npz"))
        if len(prediction_files) != 100:
            raise ValueError(f"{label} has {len(prediction_files)} prediction files, expected 100.")
    ordered_ids = validate_sample_alignment(manifest, experiment_ids)
    print("Input alignment passed: identical canonical 100-sample test set.")

    all_rows: dict[str, list[dict[str, Any]]] = {}
    cache_paths: list[Path] = []
    for label, directory in prediction_directories.items():
        cache_path = output / f".{label}_rows_cache.json"
        cache_paths.append(cache_path)
        if cache_path.exists():
            all_rows[label] = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"Reused completed trusted-gravity rows for {label}.")
        else:
            all_rows[label] = load_experiment_rows(
                label=label, prediction_directory=directory, manifest=manifest,
                ordered_ids=ordered_ids, config=config,
                forward_model=forward_context.forward_model,
            )
            cache_path.write_text(
                json.dumps(all_rows[label], indent=2), encoding="utf-8"
            )
            print(f"Completed trusted-gravity rows for {label}.")
    overall = [overall_row(label, all_rows[label]) for label in EXPERIMENTS]
    _write_csv(output / "overall_comparison.csv", overall)

    groupings = {
        "depth": lambda row: "shallow_20_40" if row["true_top_depth_m"] <= 40 else ("medium_50_60" if row["true_top_depth_m"] <= 60 else "deep_70_80"),
        "thickness": lambda row: "20_30" if row["true_thickness_m"] <= 30 else ("40_50" if row["true_thickness_m"] <= 50 else "60_80"),
    }
    grouped: list[dict[str, Any]] = []
    for name, classifier in groupings.items():
        grouped.extend(grouped_rows(all_rows, name, classifier))
    _write_csv(output / "grouped_results.csv", grouped)

    paired, paired_summary = paired_rows(all_rows)
    _write_csv(output / "paired_metrics.csv", paired)
    correlations = correlation_rows(all_rows)
    _plot_outputs(all_rows, output / "figures")
    summary = build_summary(overall, paired_summary, correlations)
    (output / "README.md").write_text(_readme(overall, summary), encoding="utf-8")
    for cache_path in cache_paths:
        cache_path.unlink(missing_ok=True)
    for row in overall:
        print(row)
    print(f"Suggested E08: {summary['E08_recommendation']}")
    print(f"Paired analysis complete: {output}")


if __name__ == "__main__":
    main()
