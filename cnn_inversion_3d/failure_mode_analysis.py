"""Fast, analysis-only failure-mode comparison for canonical E05/E06/E07.

This module deliberately has no imports from the model or gravity-forward-model
code.  It reads saved prediction arrays and saved gravity-consistency tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = {
    "E05": "E05_canonical_single_plane_baseline_bf8",
    "E06": "E06_canonical_single_plane_learned_depth_seed_bf8",
    "E07": "E07_canonical_single_plane_learned_depth_seed_physics_bf8",
}
THRESHOLD = 0.1
CELL_M = 10.0


def sample_id(row: dict[str, Any]) -> str:
    return Path(str(row["sample_path"])).stem


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def geometry(density: np.ndarray, threshold: float = THRESHOLD) -> dict[str, float]:
    """Return extents and centers for a thresholded density[z,y,x] array."""
    occupied = np.argwhere(np.asarray(density) >= threshold)
    keys = ("top_depth_m", "bottom_depth_m", "thickness_m", "width_x_m", "width_y_m",
            "center_x_m", "center_y_m")
    if occupied.size == 0:
        return {key: float("nan") for key in keys}
    lo, hi = occupied.min(axis=0), occupied.max(axis=0) + 1
    return {
        "top_depth_m": float(lo[0] * CELL_M),
        "bottom_depth_m": float(hi[0] * CELL_M),
        "thickness_m": float((hi[0] - lo[0]) * CELL_M),
        "width_x_m": float((hi[2] - lo[2]) * CELL_M),
        "width_y_m": float((hi[1] - lo[1]) * CELL_M),
        "center_x_m": float((lo[2] + hi[2]) * CELL_M / 2.0),
        "center_y_m": float((lo[1] + hi[1]) * CELL_M / 2.0),
    }


def validate_alignment(
    manifest_ids: list[str], experiment_ids: dict[str, list[str]]
) -> list[str]:
    if len(manifest_ids) != 100 or len(set(manifest_ids)) != 100:
        raise ValueError("Canonical test manifest must contain 100 unique sample IDs.")
    expected = set(manifest_ids)
    messages = []
    for label, ids in experiment_ids.items():
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        missing, extra = sorted(expected - set(ids)), sorted(set(ids) - expected)
        messages.append(f"{label}: count={len(ids)}, missing={missing}, extra={extra}, duplicates={duplicates}")
        if len(ids) != 100 or missing or extra or duplicates:
            raise ValueError("Test-set mismatch. " + "; ".join(messages))
    print("Sample alignment verified: 100 common held-out samples.")
    for message in messages:
        print("  " + message)
    return manifest_ids


def finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def summary(values: list[float]) -> dict[str, float]:
    array = finite(values)
    if not array.size:
        return {name: float("nan") for name in ("mean", "median", "q25", "q75")}
    return {
        "mean": float(np.mean(array)), "median": float(np.median(array)),
        "q25": float(np.percentile(array, 25)), "q75": float(np.percentile(array, 75)),
    }


def pct_change(old: float, new: float) -> float:
    return float((new - old) / abs(old) * 100.0) if np.isfinite(old) and old != 0 else float("nan")


def collect(root: Path, dataset: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    manifest = read_csv(dataset / "test_manifest.csv")
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    metric_maps: dict[str, dict[str, dict[str, Any]]] = {}
    gravity_maps: dict[str, dict[str, dict[str, str]]] = {}
    diagnostic_maps: dict[str, dict[str, dict[str, str]]] = {}
    experiment_ids: dict[str, list[str]] = {}
    directories: dict[str, Path] = {}
    for label, dirname in EXPERIMENTS.items():
        directory = root / "prediction_outputs" / dirname
        directories[label] = directory
        with (directory / "prediction_metrics.json").open(encoding="utf-8") as stream:
            rows = json.load(stream)
        metric_maps[label] = {sample_id(row): row for row in rows}
        experiment_ids[label] = [sample_id(row) for row in rows]
        gravity_path = directory / "gravity_consistency_metrics.csv"
        gravity_maps[label] = (
            {sample_id(row): row for row in read_csv(gravity_path)} if gravity_path.exists() else {}
        )
        diagnostic_paths = sorted(directory.glob("*_mass_depth_diagnostic.csv"))
        diagnostic_maps[label] = (
            {row["sample_id"]: row for row in read_csv(diagnostic_paths[0])}
            if diagnostic_paths else {}
        )
    ordered_ids = validate_alignment(
        [row["sample_id"] for row in manifest], experiment_ids
    )

    all_rows: list[dict[str, Any]] = []
    by_experiment: dict[str, list[dict[str, Any]]] = {label: [] for label in EXPERIMENTS}
    for label in EXPERIMENTS:
        for sid in ordered_ids:
            metric = metric_maps[label][sid]
            truth_meta = manifest_by_id[sid]
            prediction_path = directories[label] / str(metric["prediction_path"])
            with np.load(prediction_path) as saved:
                truth = np.asarray(saved["true_density"], dtype=float).squeeze()
                prediction = np.asarray(saved["predicted_density"], dtype=float).squeeze()
            truth_geometry = geometry(truth)
            prediction_geometry = geometry(prediction)
            true_cells = int(metric["true_occupied_cells"])
            predicted_cells = int(metric["predicted_occupied_cells"])
            existing_diagnostic = diagnostic_maps[label].get(sid, {})
            true_sum = float(existing_diagnostic.get("true_density_sum", np.sum(truth)))
            predicted_sum = float(existing_diagnostic.get("predicted_density_sum", np.sum(prediction)))
            true_mask = truth >= THRESHOLD
            gravity = gravity_maps[label].get(sid, {})
            row: dict[str, Any] = {
                "experiment": label, "sample_id": sid,
                "sample_path": metric["sample_path"], "prediction_path": metric["prediction_path"],
                "mse": float(metric["mse"]), "mae": float(metric["mae"]),
                "correlation": float(metric["correlation"]), "iou": float(metric["iou"]),
                "dice": float(metric["dice"]), "true_occupied_cells": true_cells,
                "predicted_occupied_cells": predicted_cells,
                "volume_ratio": predicted_cells / true_cells if true_cells else float("nan"),
                "true_mass_proxy": true_sum, "predicted_mass_proxy": predicted_sum,
                "mass_ratio": predicted_sum / true_sum if true_sum else float("nan"),
                "true_body_mean_density": float(np.mean(truth[true_mask])),
                "predicted_mean_density_in_true_body": float(np.mean(prediction[true_mask])),
                "peak_predicted_density": float(np.max(prediction)),
                "true_top_depth_m": truth_geometry["top_depth_m"],
                "true_bottom_depth_m": truth_geometry["bottom_depth_m"],
                "true_thickness_m": truth_geometry["thickness_m"],
                "true_width_x_m": truth_geometry["width_x_m"],
                "true_width_y_m": truth_geometry["width_y_m"],
                "true_center_x_m": truth_geometry["center_x_m"],
                "true_center_y_m": truth_geometry["center_y_m"],
                "gravity_mse": float(gravity.get("gravity_mse", "nan")),
                "gravity_rmse": float(gravity.get("gravity_rmse", "nan")),
                "gravity_mae": float(gravity.get("gravity_mae", "nan")),
                "gravity_relative_l2": float(gravity.get("gravity_relative_l2", "nan")),
                "gravity_correlation": float(gravity.get("gravity_correlation", "nan")),
                "gravity_max_abs_residual": float(gravity.get("gravity_max_abs_residual", "nan")),
                "gravity_mean_residual": float(gravity.get("gravity_mean_residual", "nan")),
                "gravity_residual_std": float(gravity.get("gravity_residual_std", "nan")),
                "manifest_top_depth_m": float(truth_meta["top_depth_m"]),
                "manifest_thickness_m": float(truth_meta["thickness_z_m"]),
            }
            for name in ("top_depth_m", "bottom_depth_m", "thickness_m", "width_x_m", "width_y_m"):
                short = name.removesuffix("_m")
                predicted = prediction_geometry[name]
                true = truth_geometry[name]
                row[f"predicted_{name}"] = predicted
                row[f"{short}_error_m"] = predicted - true
                row[f"absolute_{short}_error_m"] = abs(predicted - true)
            row["predicted_center_x_m"] = prediction_geometry["center_x_m"]
            row["predicted_center_y_m"] = prediction_geometry["center_y_m"]
            row["lateral_center_error_m"] = float(np.hypot(
                prediction_geometry["center_x_m"] - truth_geometry["center_x_m"],
                prediction_geometry["center_y_m"] - truth_geometry["center_y_m"],
            ))
            all_rows.append(row)
            by_experiment[label].append(row)
    return all_rows, by_experiment


AGGREGATES = (
    "mse", "mae", "iou", "dice", "volume_ratio", "mass_ratio",
    "absolute_top_depth_error_m", "absolute_bottom_depth_error_m",
    "absolute_thickness_error_m", "absolute_width_x_error_m", "absolute_width_y_error_m",
    "lateral_center_error_m", "gravity_rmse", "gravity_correlation", "gravity_relative_l2",
)


def aggregate(by_experiment: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    for label, rows in by_experiment.items():
        result: dict[str, Any] = {"experiment": label, "n_samples": len(rows)}
        for metric in AGGREGATES:
            for statistic, value in summary([float(row[metric]) for row in rows]).items():
                result[f"{statistic}_{metric}"] = value
        output.append(result)
    return output


def paired(by_experiment: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    maps = {label: {row["sample_id"]: row for row in rows} for label, rows in by_experiment.items()}
    metrics = ("mse", "mae", "iou", "dice", "volume_ratio", "mass_ratio",
               "absolute_top_depth_error_m", "absolute_bottom_depth_error_m",
               "absolute_thickness_error_m", "gravity_rmse")
    rows = []
    for sid in maps["E05"]:
        row: dict[str, Any] = {"sample_id": sid}
        for label in EXPERIMENTS:
            for metric in metrics:
                row[f"{label}_{metric}"] = maps[label][sid][metric]
        for old, new in (("E05", "E06"), ("E06", "E07"), ("E05", "E07")):
            tag = f"{old}_to_{new}"
            for metric in metrics:
                row[f"{tag}_{metric}_change"] = maps[new][sid][metric] - maps[old][sid][metric]
        rows.append(row)

    comparisons = []
    for old, new in (("E05", "E06"), ("E06", "E07"), ("E05", "E07")):
        result: dict[str, Any] = {"comparison": f"{old} -> {new}"}
        for metric in ("mse", "mae", "iou", "dice", "absolute_top_depth_error_m",
                       "absolute_bottom_depth_error_m", "absolute_thickness_error_m", "gravity_rmse"):
            old_mean = summary([r[metric] for r in by_experiment[old]])["mean"]
            new_mean = summary([r[metric] for r in by_experiment[new]])["mean"]
            result[f"percent_change_mean_{metric}"] = pct_change(old_mean, new_mean)
        for metric in ("volume_ratio", "mass_ratio"):
            old_error = float(np.nanmedian([abs(r[metric] - 1) for r in by_experiment[old]]))
            new_error = float(np.nanmedian([abs(r[metric] - 1) for r in by_experiment[new]]))
            result[f"percent_change_median_{metric}_error"] = pct_change(old_error, new_error)
        rules = {
            "lower_mse": ("mse", lambda a, b: b < a), "higher_dice": ("dice", lambda a, b: b > a),
            "volume_ratio_closer": ("volume_ratio", lambda a, b: abs(b - 1) < abs(a - 1)),
            "mass_ratio_closer": ("mass_ratio", lambda a, b: abs(b - 1) < abs(a - 1)),
            "better_top_depth": ("absolute_top_depth_error_m", lambda a, b: b < a),
            "better_bottom_depth": ("absolute_bottom_depth_error_m", lambda a, b: b < a),
            "better_thickness": ("absolute_thickness_error_m", lambda a, b: b < a),
        }
        for name, (metric, rule) in rules.items():
            result[f"percent_samples_{name}"] = 100.0 * np.mean([
                rule(float(a[metric]), float(b[metric]))
                for a, b in zip(by_experiment[old], by_experiment[new])
            ])
        comparisons.append(result)
    return rows, comparisons


def grouped(by_experiment: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    definitions = {
        "top_depth": [(20, 40), (50, 60), (70, 80)],
        "thickness": [(20, 30), (40, 50), (60, 80)],
    }
    output = []
    for label, rows in by_experiment.items():
        for group_type, ranges in definitions.items():
            field = "true_top_depth_m" if group_type == "top_depth" else "true_thickness_m"
            for low, high in ranges:
                selected = [row for row in rows if low <= float(row[field]) <= high]
                result: dict[str, Any] = {
                    "experiment": label, "group_type": group_type,
                    "range_m": f"{low}-{high}", "n_samples": len(selected),
                }
                for metric in ("dice", "volume_ratio", "mass_ratio", "absolute_top_depth_error_m",
                               "absolute_bottom_depth_error_m", "absolute_thickness_error_m"):
                    result[f"mean_{metric}"] = summary([r[metric] for r in selected])["mean"]
                output.append(result)
    return output


def diagnose(rows: list[dict[str, Any]]) -> tuple[str, str, dict[str, float]]:
    scores = {
        "excess predicted volume": float(np.median(np.abs([r["volume_ratio"] - 1 for r in rows]))),
        "incorrect anomalous mass": float(np.median(np.abs([r["mass_ratio"] - 1 for r in rows]))),
        "incorrect top depth": float(np.mean([r["absolute_top_depth_error_m"] / r["true_top_depth_m"] for r in rows])),
        "incorrect bottom depth": float(np.mean([r["absolute_bottom_depth_error_m"] / r["true_bottom_depth_m"] for r in rows])),
        "incorrect thickness": float(np.mean([r["absolute_thickness_error_m"] / r["true_thickness_m"] for r in rows])),
        "lateral localization": float(np.mean([r["lateral_center_error_m"] / 160.0 for r in rows])),
        "density amplitude": float(np.median(np.abs([
            r["predicted_mean_density_in_true_body"] / r["true_body_mean_density"] - 1 for r in rows
        ]))),
        "gravity consistency": float(np.median([r["gravity_relative_l2"] for r in rows])),
    }
    dominant = max(scores, key=scores.get)
    evidence = {
        "excess predicted volume": "median |volume ratio - 1|",
        "incorrect anomalous mass": "median |mass-proxy ratio - 1|",
        "incorrect top depth": "mean absolute top-depth error relative to true top depth",
        "incorrect bottom depth": "mean absolute bottom-depth error relative to true bottom depth",
        "incorrect thickness": "mean absolute thickness error relative to true thickness",
        "lateral localization": "mean lateral-center error relative to the 160 m maximum body width",
        "density amplitude": "median true-body density-amplitude ratio error",
        "gravity consistency": "median saved gravity relative-L2 error",
    }[dominant]
    recommendations = {
        "excess predicted volume": "Test one explicit occupied-volume/sparsity regularizer while holding E07 fixed.",
        "incorrect anomalous mass": "Test one density-sum (mass-proxy) consistency penalty while holding E07 fixed.",
        "incorrect top depth": "Test one top-depth supervision term while holding E07 fixed.",
        "incorrect bottom depth": "Test one bottom-depth supervision term while holding E07 fixed.",
        "incorrect thickness": "Test one thickness supervision term while holding E07 fixed.",
        "lateral localization": "Test one lateral center-of-mass supervision term while holding E07 fixed.",
        "density amplitude": "Test one true-body density-amplitude supervision term while holding E07 fixed.",
        "gravity consistency": "Test a single adjusted gravity-loss weight while holding the rest of E07 fixed.",
    }
    sentence = f"{dominant.capitalize()} is largest by normalized diagnostic score ({scores[dominant]:.3f}; {evidence})."
    return sentence, recommendations[dominant], scores


def plots(output: Path, overall: list[dict[str, Any]], by: dict[str, list[dict[str, Any]]], scores: dict[str, float]) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = list(EXPERIMENTS)
    colors = ["#6b7280", "#2563eb", "#dc2626"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, metric, title in zip(axes, ("mse", "dice", "iou"), ("MSE (lower is better)", "Dice", "IoU")):
        ax.bar(labels, [r[f"mean_{metric}"] for r in overall], color=colors)
        ax.set_title(title); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(figures / "overall_metrics.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for ax, metric, title in zip(axes, ("volume_ratio", "mass_ratio"), ("Occupied-volume ratio", "Density-sum mass-proxy ratio")):
        ax.boxplot([[r[metric] for r in by[label]] for label in labels], tick_labels=labels, showfliers=False)
        ax.axhline(1, color="black", ls="--", lw=1); ax.set_title(title); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(figures / "volume_mass_recovery.png", dpi=180); plt.close(fig)

    e07 = by["E07"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, name, title in zip(axes, ("top_depth", "bottom_depth", "thickness"), ("Top depth", "Bottom depth", "Thickness")):
        x, y = [r[f"true_{name}_m"] for r in e07], [r[f"predicted_{name}_m"] for r in e07]
        ax.scatter(x, y, s=18, alpha=.65, color="#dc2626")
        limits = [min(x + y), max(x + y)]; ax.plot(limits, limits, "k--", lw=1)
        ax.set(xlabel="True (m)", ylabel="Predicted (m)", title=title); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "depth_recovery.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = list(scores); values = [scores[name] for name in names]
    ax.barh(names, values, color=["#dc2626" if value == max(values) else "#94a3b8" for value in values])
    ax.invert_yaxis(); ax.set_xlabel("Normalized diagnostic error (larger is worse)")
    ax.set_title("E07 remaining failure-mode indicators"); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(figures / "e07_failure_summary.png", dpi=180); plt.close(fig)


def make_readme(overall: list[dict[str, Any]], comparisons: list[dict[str, Any]], diagnosis: str, recommendation: str) -> str:
    comp = {row["comparison"]: row for row in comparisons}
    def transition(name: str) -> str:
        row = comp[name]
        return (f"Mean MSE changed {row['percent_change_mean_mse']:+.1f}%, mean Dice "
                f"{row['percent_change_mean_dice']:+.1f}%, and mean gravity RMSE "
                f"{row['percent_change_mean_gravity_rmse']:+.1f}%. Lower MSE occurred for "
                f"{row['percent_samples_lower_mse']:.1f}% of samples and higher Dice for "
                f"{row['percent_samples_higher_dice']:.1f}%.")
    def paired_table(name: str) -> str:
        row = comp[name]
        entries = (
            ("Mean MSE", "percent_change_mean_mse"), ("Mean MAE", "percent_change_mean_mae"),
            ("Mean IoU", "percent_change_mean_iou"), ("Mean Dice", "percent_change_mean_dice"),
            ("Median volume-ratio error", "percent_change_median_volume_ratio_error"),
            ("Median mass-ratio error", "percent_change_median_mass_ratio_error"),
            ("Mean absolute top-depth error", "percent_change_mean_absolute_top_depth_error_m"),
            ("Mean absolute bottom-depth error", "percent_change_mean_absolute_bottom_depth_error_m"),
            ("Mean absolute thickness error", "percent_change_mean_absolute_thickness_error_m"),
            ("Mean saved gravity RMSE", "percent_change_mean_gravity_rmse"),
        )
        lines = ["| Metric | Percent change |", "|---|---:|"]
        lines.extend(f"| {label} | {row[key]:+.1f}% |" for label, key in entries)
        lines.extend(("", "| Paired sample criterion | Improved |", "|---|---:|"))
        for label, key in (("Lower MSE", "percent_samples_lower_mse"),
                           ("Higher Dice", "percent_samples_higher_dice"),
                           ("Volume ratio closer to 1", "percent_samples_volume_ratio_closer"),
                           ("Mass ratio closer to 1", "percent_samples_mass_ratio_closer"),
                           ("Better top depth", "percent_samples_better_top_depth"),
                           ("Better bottom depth", "percent_samples_better_bottom_depth"),
                           ("Better thickness", "percent_samples_better_thickness")):
            lines.append(f"| {label} | {row[key]:.1f}% |")
        return "\n".join(lines)
    return f"""# E05 / E06 / E07 Failure-Mode Analysis

## What was analyzed

The same 100 canonical held-out samples were matched by sample ID. Saved prediction metrics, saved density arrays, and saved gravity-consistency CSVs were analyzed at density threshold {THRESHOLD}. Density sums are reported only as mass proxies.

No model was loaded, no dataset was regenerated, and no prediction or forward-gravity calculation was run.

## E05 -> E06 findings

{transition('E05 -> E06')}

{paired_table('E05 -> E06')}

## E06 -> E07 findings

{transition('E06 -> E07')}

{paired_table('E06 -> E07')}

## E05 -> E07 paired result

{transition('E05 -> E07')}

{paired_table('E05 -> E07')}

## Current E07 failure mode

{diagnosis}

The ranking uses dimensionless, interpretable error indicators so unlike units are not compared directly. Gravity uses the already-saved relative-L2 result; RMSE and correlation are also preserved in the CSV outputs.

## Recommended target for E08

{recommendation}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/canonical_single_plane_train2000"))
    parser.add_argument("--output", type=Path, default=Path("analysis_outputs/E05_E06_E07_failure_mode_analysis"))
    args = parser.parse_args()
    root = Path.cwd()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    all_rows, by = collect(root, args.dataset.resolve())
    overall = aggregate(by)
    paired_rows, comparisons = paired(by)
    groups = grouped(by)
    diagnosis, recommendation, scores = diagnose(by["E07"])
    write_csv(output / "overall_comparison.csv", overall)
    write_csv(output / "paired_metrics.csv", paired_rows)
    write_csv(output / "geometry_metrics.csv", all_rows)
    write_csv(output / "grouped_results.csv", groups)
    (output / "README.md").write_text(make_readme(overall, comparisons, diagnosis, recommendation), encoding="utf-8")
    plots(output, overall, by, scores)
    e07 = next(row for row in overall if row["experiment"] == "E07")
    print("\n" + "=" * 60)
    print("E05 / E06 / E07 FAILURE-MODE ANALYSIS COMPLETE")
    print("=" * 60)
    for label, key in (("E07 mean MSE", "mean_mse"), ("E07 mean Dice", "mean_dice"),
                       ("E07 mean IoU", "mean_iou"), ("E07 median volume ratio", "median_volume_ratio"),
                       ("E07 median mass ratio", "median_mass_ratio"),
                       ("E07 mean |top-depth error|", "mean_absolute_top_depth_error_m"),
                       ("E07 mean |bottom-depth error|", "mean_absolute_bottom_depth_error_m"),
                       ("E07 mean |thickness error|", "mean_absolute_thickness_error_m"),
                       ("E07 gravity RMSE", "mean_gravity_rmse"),
                       ("E07 gravity correlation", "mean_gravity_correlation")):
        print(f"{label}: {e07[key]:.6g}")
    print(f"\nDOMINANT REMAINING FAILURE MODE:\n{diagnosis}")
    print(f"\nRECOMMENDED E08 TARGET:\n{recommendation}")
    print("=" * 60)


if __name__ == "__main__":
    main()
