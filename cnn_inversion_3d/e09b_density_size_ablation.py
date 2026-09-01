"""Sequential E09B-5--8 workflow and controlled density/body-size analysis."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from cnn_inversion_3d import failure_mode_analysis as analysis

EXPERIMENTS = {
    "E09B": {"slug": "E09B_integrated_sensitivity", "depth": 1.0, "amplitude": 0.0, "small": False},
    "E09B-2": {"slug": "E09B_2_depth2p0", "depth": 2.0, "amplitude": 0.0, "small": False},
    "E09B-3": {"slug": "E09B_3_depth3p0", "depth": 3.0, "amplitude": 0.0, "small": False},
    "E09B-5": {"slug": "E09B_5_depth2p5", "depth": 2.5, "amplitude": 0.0, "small": False},
    "E09B-6": {"slug": "E09B_6_amplitude", "depth": 2.0, "amplitude": 1.0, "small": False},
    "E09B-7": {"slug": "E09B_7_small_body", "depth": 2.0, "amplitude": 0.0, "small": True},
    "E09B-8": {"slug": "E09B_8_amplitude_small_body", "depth": 2.0, "amplitude": 1.0, "small": True},
}
NEW_LABELS = ("E09B-5", "E09B-6", "E09B-7", "E09B-8")


def _run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_training(path: Path, cfg: dict[str, Any]) -> bool:
    metadata = path / "training_metadata.json"
    if not metadata.exists() or not (path / "best_model.keras").exists():
        return False
    data = _json(metadata)
    loss = data.get("loss", {}).get("e09b") or {}
    return (
        data.get("trainable_parameters") == 190592
        and abs(float(loss.get("lambda_depth", -1)) - cfg["depth"]) < 1e-12
        and abs(float(loss.get("lambda_amplitude", -1)) - cfg["amplitude"]) < 1e-12
        and bool(loss.get("small_body_weighting", False)) == cfg["small"]
    )


def _valid_prediction(path: Path) -> bool:
    metrics = path / "prediction_metrics.json"
    return metrics.exists() and len(_json(metrics)) == 100


def _valid_analysis(path: Path) -> bool:
    return (path / "gravity_consistency_metrics.csv").exists()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    analysis.write_csv(path, rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.nanmean([float(row[key]) for row in rows]))


def compare(root: Path, dataset: Path, output: Path) -> None:
    analysis.EXPERIMENTS = {
        label: str(root / "prediction_outputs" / cfg["slug"])
        for label, cfg in EXPERIMENTS.items()
    }
    sample_rows, by_experiment = analysis.collect(root, dataset)
    test_ids = [row["sample_id"] for row in by_experiment["E09B-2"]]
    reference = {row["sample_id"]: row for row in by_experiment["E09B-2"]}
    volumes = np.asarray([reference[sid]["true_occupied_cells"] for sid in test_ids], float)
    q25, q75 = np.quantile(volumes, [0.25, 0.75])
    ranked_ids = sorted(test_ids, key=lambda sid: (reference[sid]["true_occupied_cells"], sid))
    lower_count = len(ranked_ids) // 4
    upper_start = len(ranked_ids) - lower_count
    groups = {sid: ("small" if index < lower_count else
                    "large" if index >= upper_start else "medium")
              for index, sid in enumerate(ranked_ids)}
    for row in sample_rows:
        true_density = float(row["true_body_mean_density"])
        predicted_density = float(row["predicted_mean_density_in_true_body"])
        row.update({
            "true_body_volume_cells": int(row["true_occupied_cells"]),
            "true_body_volume_m3": int(row["true_occupied_cells"]) * 1000.0,
            "true_mean_body_density": true_density,
            "predicted_mean_body_density": predicted_density,
            "density_error": predicted_density - true_density,
            "absolute_density_error": abs(predicted_density - true_density),
            "relative_density_error": abs(predicted_density - true_density) / max(abs(true_density), 1e-8),
            "body_size_group": groups[row["sample_id"]],
            "center_depth_error_m": row["z_center_error_m"],
        })
    by_experiment = {
        label: [row for row in sample_rows if row["experiment"] == label]
        for label in EXPERIMENTS
    }
    overall: list[dict[str, Any]] = []
    density_summary: list[dict[str, Any]] = []
    grouped: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    metrics = ("mse", "iou", "dice", "absolute_top_depth_error_m",
               "absolute_bottom_depth_error_m", "absolute_z_center_error_m",
               "absolute_thickness_error_m", "gravity_rmse", "gravity_relative_l2",
               "gravity_correlation", "absolute_density_error", "relative_density_error")
    for label, rows in by_experiment.items():
        cfg = EXPERIMENTS[label]
        overall.append({"experiment": label, "lambda_depth": cfg["depth"],
                        "lambda_amplitude": cfg["amplitude"],
                        "small_body_weighting": cfg["small"],
                        **{f"mean_{key}": _mean(rows, key) for key in metrics}})
        density_errors = np.asarray([row["density_error"] for row in rows], float)
        density_summary.append({
            "experiment": label,
            "true_mean_body_density": _mean(rows, "true_mean_body_density"),
            "predicted_mean_body_density": _mean(rows, "predicted_mean_body_density"),
            "density_mae": float(np.mean(np.abs(density_errors))),
            "density_rmse": float(np.sqrt(np.mean(np.square(density_errors)))),
            "mean_relative_density_error": _mean(rows, "relative_density_error"),
            "density_bias": float(np.mean(density_errors)),
            "bias_direction": "underprediction" if np.mean(density_errors) < 0 else "overprediction",
            "density_error_gravity_rmse_correlation": float(np.corrcoef(
                [r["absolute_density_error"] for r in rows], [r["gravity_rmse"] for r in rows]
            )[0, 1]),
            "relative_density_gravity_relative_l2_correlation": float(np.corrcoef(
                [r["relative_density_error"] for r in rows], [r["gravity_relative_l2"] for r in rows]
            )[0, 1]),
        })
        for group in ("small", "medium", "large"):
            selected = [row for row in rows if row["body_size_group"] == group]
            group_bias = float(np.mean([row["density_error"] for row in selected]))
            grouped.append({"experiment": label, "body_size_group": group,
                            "sample_count": len(selected),
                            **{f"mean_{key}": _mean(selected, key) for key in metrics},
                            "true_mean_body_density": _mean(selected, "true_mean_body_density"),
                            "predicted_mean_body_density": _mean(selected, "predicted_mean_body_density"),
                            "density_bias": group_bias,
                            "bias_direction": "underprediction" if group_bias < 0 else "overprediction"})
        history_path = root / "training_outputs" / cfg["slug"] / "training_history.csv"
        history = list(csv.DictReader(history_path.open(encoding="utf-8")))
        best = min(history, key=lambda row: float(row["val_loss"]))
        training.append({"experiment": label, "best_epoch": int(best["epoch"]),
                         "final_epoch": int(history[-1]["epoch"]),
                         "best_validation_loss": float(best["val_loss"]),
                         "trainable_parameters": 190592})
    paired: list[dict[str, Any]] = []
    maps = {label: {row["sample_id"]: row for row in rows} for label, rows in by_experiment.items()}
    for old, new in (("E09B-2", "E09B-5"), ("E09B-2", "E09B-6"),
                     ("E09B-2", "E09B-7"), ("E09B-6", "E09B-8"),
                     ("E09B-7", "E09B-8")):
        for key in metrics:
            changes = np.asarray([maps[new][sid][key] - maps[old][sid][key] for sid in test_ids], float)
            paired.append({"comparison": f"{old} -> {new}", "metric": key,
                           "mean_paired_change": float(np.nanmean(changes)),
                           "median_paired_change": float(np.nanmedian(changes))})
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("overall_comparison.csv", overall), ("sample_metrics.csv", sample_rows),
                           ("paired_metrics.csv", paired), ("body_size_group_metrics.csv", grouped),
                           ("density_amplitude_metrics.csv", density_summary),
                           ("training_summary.csv", training)):
        _write_csv(output / filename, rows)
    labels = list(EXPERIMENTS)
    def bars(filename: str, keys: tuple[str, ...], titles: tuple[str, ...]) -> None:
        fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), constrained_layout=True)
        axes = np.atleast_1d(axes)
        for axis, key, title in zip(axes, keys, titles):
            axis.bar(labels, [next(r for r in overall if r["experiment"] == label)[key] for label in labels])
            axis.set_title(title); axis.tick_params(axis="x", rotation=45); axis.grid(axis="y", alpha=0.25)
        fig.savefig(output / filename, dpi=180); plt.close(fig)
    bars("overall_metrics.png", ("mean_iou", "mean_dice", "mean_mse"), ("IoU", "Dice", "MSE"))
    bars("depth_metrics.png", ("mean_absolute_top_depth_error_m", "mean_absolute_bottom_depth_error_m", "mean_absolute_thickness_error_m"), ("Top MAE (m)", "Bottom MAE (m)", "Thickness MAE (m)"))
    bars("density_amplitude_metrics.png", ("mean_absolute_density_error", "mean_relative_density_error"), ("Density MAE", "Relative density error"))
    bars("gravity_metrics.png", ("mean_gravity_rmse", "mean_gravity_relative_l2", "mean_gravity_correlation"), ("Gravity RMSE", "Gravity relative L2", "Gravity correlation"))
    scatter_specs = (
        ("body_size_vs_iou.png", "true_body_volume_cells", "iou"),
        ("body_size_vs_density_error.png", "true_body_volume_cells", "absolute_density_error"),
        ("body_size_vs_gravity_rmse.png", "true_body_volume_cells", "gravity_rmse"),
        ("density_error_vs_gravity_rmse.png", "absolute_density_error", "gravity_rmse"),
    )
    for filename, xkey, ykey in scatter_specs:
        fig, axis = plt.subplots(figsize=(7, 5))
        for label, rows in by_experiment.items():
            axis.scatter([r[xkey] for r in rows], [r[ykey] for r in rows], s=12, alpha=0.45, label=label)
        axis.set(xlabel=xkey, ylabel=ykey); axis.grid(alpha=0.25); axis.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(output / filename, dpi=180); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 5))
    for label, rows in by_experiment.items():
        axis.scatter([r["relative_density_error"] for r in rows],
                     [r["gravity_relative_l2"] for r in rows], s=12, alpha=0.45, label=label)
    axis.set(xlabel="relative_density_error", ylabel="gravity_relative_l2")
    axis.grid(alpha=0.25); axis.legend(fontsize=7); fig.tight_layout()
    fig.savefig(output / "relative_density_error_vs_gravity_relative_l2.png", dpi=180); plt.close(fig)
    overall_map = {row["experiment"]: row for row in overall}
    group_map = {(row["experiment"], row["body_size_group"]): row for row in grouped}
    density_map = {row["experiment"]: row for row in density_summary}
    overlap_winner = max(overall, key=lambda row: row["mean_dice"])["experiment"]
    depth_winner = min(overall, key=lambda row: row["mean_absolute_top_depth_error_m"] +
                       row["mean_absolute_bottom_depth_error_m"] +
                       row["mean_absolute_thickness_error_m"])["experiment"]
    density_winner = min(overall, key=lambda row: row["mean_absolute_density_error"])["experiment"]
    gravity_winner = min(overall, key=lambda row: row["mean_gravity_rmse"])["experiment"]
    small_winner = max(grouped, key=lambda row: row["mean_dice"] if row["body_size_group"] == "small" else -np.inf)["experiment"]
    rank_metrics = (("mean_dice", True), ("mean_absolute_top_depth_error_m", False),
                    ("mean_absolute_density_error", False), ("mean_gravity_rmse", False))
    rank_score = {label: 0 for label in labels}
    for key, higher in rank_metrics:
        ordered = sorted(labels, key=lambda label: overall_map[label][key], reverse=higher)
        for rank, label in enumerate(ordered, 1): rank_score[label] += rank
    compromise = min(rank_score, key=rank_score.get)
    complementary_checks = (
        overall_map["E09B-8"]["mean_dice"] > max(overall_map["E09B-6"]["mean_dice"], overall_map["E09B-7"]["mean_dice"]),
        overall_map["E09B-8"]["mean_absolute_top_depth_error_m"] < min(overall_map["E09B-6"]["mean_absolute_top_depth_error_m"], overall_map["E09B-7"]["mean_absolute_top_depth_error_m"]),
        overall_map["E09B-8"]["mean_absolute_density_error"] < min(overall_map["E09B-6"]["mean_absolute_density_error"], overall_map["E09B-7"]["mean_absolute_density_error"]),
        overall_map["E09B-8"]["mean_gravity_rmse"] < min(overall_map["E09B-6"]["mean_gravity_rmse"], overall_map["E09B-7"]["mean_gravity_rmse"]),
    )
    combined_interpretation = (
        "complementary" if sum(complementary_checks) >= 3
        else "conflicting" if sum(complementary_checks) <= 1
        else "neutral/mixed"
    )
    def delta(old: str, new: str, key: str) -> float:
        return overall_map[new][key] - overall_map[old][key]
    b2_small, b2_large = group_map[("E09B-2", "small")], group_map[("E09B-2", "large")]
    readme = f"""# E09B Density and Body-Size Ablation

E09B-2 is the controlled baseline. The common 100-sample test set is ranked once
by true volume into exactly 25 small, 50 medium, and 25 large bodies (quantile
values: q25={q25:g}, q75={q75:g} cells). Test quantiles are never used in training.

## Direct answers

1. Depth refinement: E09B-5 Dice minus E09B-2 = {delta('E09B-2','E09B-5','mean_dice'):+.5f}; versus E09B-3 = {delta('E09B-3','E09B-5','mean_dice'):+.5f}.
2. Best overlap/depth compromise by the four-objective rank used here: {compromise}.
3. E09B-6 density MAE change from E09B-2: {delta('E09B-2','E09B-6','mean_absolute_density_error'):+.6g} g/cm^3.
4. E09B-6 density-error/gravity-RMSE correlation: {density_map['E09B-6']['density_error_gravity_rmse_correlation']:+.3f}; association does not establish causation.
5. E09B-6 Dice change from E09B-2: {delta('E09B-2','E09B-6','mean_dice'):+.5f}.
6. E09B-2 small-minus-large Dice: {b2_small['mean_dice']-b2_large['mean_dice']:+.5f}.
7. E09B-7 small-body Dice change from E09B-2: {group_map[('E09B-7','small')]['mean_dice']-b2_small['mean_dice']:+.5f}.
8. E09B-7 small-body top-depth MAE change: {group_map[('E09B-7','small')]['mean_absolute_top_depth_error_m']-b2_small['mean_absolute_top_depth_error_m']:+.3f} m.
9. E09B-7 small-body density MAE change: {group_map[('E09B-7','small')]['mean_absolute_density_error']-b2_small['mean_absolute_density_error']:+.6g} g/cm^3.
10. E09B-7 small-body gravity RMSE change: {group_map[('E09B-7','small')]['mean_gravity_rmse']-b2_small['mean_gravity_rmse']:+.6g} mGal.
11. Medium/large Dice changes for E09B-7: {group_map[('E09B-7','medium')]['mean_dice']-group_map[('E09B-2','medium')]['mean_dice']:+.5f} / {group_map[('E09B-7','large')]['mean_dice']-b2_large['mean_dice']:+.5f}.
12. E09B-8 small-body Dice minus E09B-6: {group_map[('E09B-8','small')]['mean_dice']-group_map[('E09B-6','small')]['mean_dice']:+.5f}.
13. E09B-8 density MAE minus E09B-7: {delta('E09B-7','E09B-8','mean_absolute_density_error'):+.6g} g/cm^3.
14. Combined intervention classification: **{combined_interpretation}** ({sum(complementary_checks)}/4 objectives outperform both single interventions).

## Objective leaders

- Best IoU/Dice: {overlap_winner}
- Best combined vertical-depth errors: {depth_winner}
- Best density-amplitude recovery: {density_winner}
- Best gravity consistency by RMSE: {gravity_winner}
- Best small-body Dice: {small_winner}
- Best four-objective rank compromise: {compromise}

Density bias and its under/overprediction direction are reported overall and by
body-size group. Correlations are descriptive associations, not proof of causation.

## Run everything

```bash
python -m cnn_inversion_3d.e09b_density_size_ablation --resume --overwrite
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/canonical_single_plane_train2000"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gravity-scale", type=float, default=0.22938017547130585)
    args = parser.parse_args()
    root = Path.cwd(); dataset = args.dataset.resolve()
    for index, label in enumerate(NEW_LABELS, 1):
        cfg = EXPERIMENTS[label]; output = root / "training_outputs" / cfg["slug"]
        print(f"[TRAIN {index}/4] {label}", flush=True)
        if args.resume and _valid_training(output, cfg): print("  verified; skipping"); continue
        command = [sys.executable, "-m", "cnn_inversion_3d.train", "--dataset", str(dataset),
                   "--output", str(output), "--architecture", "single_plane_asymmetric_2d_unet_sensitivity_loss",
                   "--e09a-lambda-density", "1", "--e09a-lambda-depth", str(cfg["depth"]),
                   "--e09a-alpha-center", "1", "--e09b-lambda-sensitivity", "1",
                   "--e09b-lambda-amplitude", str(cfg["amplitude"]), "--e09b-sensitivity-gamma", "0.5",
                   "--e09b-weight-min", "0.5", "--e09b-weight-max", "5", "--base-filters", "8",
                   "--gravity-scale-summary", str(dataset / "training_distribution.json"),
                   "--gravity-scale-method", "percentile_99", "--learning-rate", "0.001",
                   "--batch-size", "2", "--epochs", "100", "--seed", "20260727"]
        if cfg["small"]:
            command += ["--e09b-small-body-weighting", "--e09b-volume-gamma", "0.5",
                        "--e09b-sample-weight-min", "0.5", "--e09b-sample-weight-max", "2"]
        if args.overwrite: command.append("--overwrite")
        _run(command)
        if not _valid_training(output, cfg): raise RuntimeError(f"Invalid training output for {label}")
    for index, label in enumerate(NEW_LABELS, 1):
        cfg = EXPERIMENTS[label]; output = root / "prediction_outputs" / cfg["slug"]
        print(f"[PREDICT {index}/4] {label}", flush=True)
        if args.resume and _valid_prediction(output): print("  verified; skipping"); continue
        command = [sys.executable, "-m", "cnn_inversion_3d.predict", "--dataset", str(dataset),
                   "--model", str(root / "training_outputs" / cfg["slug"] / "best_model.keras"),
                   "--manifest", "test_manifest.csv", "--output", str(output), "--samples", "100",
                   "--threshold", "0.1", "--gravity-scale", str(args.gravity_scale)]
        if args.overwrite: command.append("--overwrite")
        _run(command)
        if not _valid_prediction(output): raise RuntimeError(f"Invalid predictions for {label}")
    for index, label in enumerate(NEW_LABELS, 1):
        cfg = EXPERIMENTS[label]; output = root / "prediction_outputs" / cfg["slug"]
        print(f"[ANALYZE {index}/4] {label}", flush=True)
        if args.resume and _valid_analysis(output): print("  verified; skipping"); continue
        _run([sys.executable, "-m", "cnn_inversion_3d.analyze_predictions", "--dataset", str(dataset),
              "--predictions", str(output), "--metrics", "prediction_metrics.json",
              "--evaluate-gravity-consistency"])
        if not _valid_analysis(output): raise RuntimeError(f"Invalid analysis for {label}")
    print("[FINAL COMPARISON]", flush=True)
    compare(root, dataset, root / "analysis_outputs" / "E09B_density_size_ablation")


if __name__ == "__main__":
    main()
