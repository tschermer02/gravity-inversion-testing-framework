"""Analysis-only comparison of completed E10A, E10B, and E10C predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from cnn_inversion_3d import failure_mode_analysis as analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/canonical_single_plane_train2000"))
    parser.add_argument("--e10a", type=Path, default=Path("prediction_outputs/E10A_shape_density_conv3d"))
    parser.add_argument("--e10b", type=Path, default=Path("prediction_outputs/E10B_shape_sensitivity_conv3d"))
    parser.add_argument("--e10c", type=Path, default=Path("prediction_outputs/E10C_shape_sensitivity_physics_conv3d"))
    parser.add_argument("--output", type=Path, default=Path("analysis_outputs/E10A_E10B_E10C_comparison"))
    args = parser.parse_args()
    analysis.EXPERIMENTS = {"E10A": str(args.e10a), "E10B": str(args.e10b), "E10C": str(args.e10c)}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, by_experiment = analysis.collect(Path.cwd(), args.dataset.resolve())
    overall = analysis.aggregate(by_experiment)
    grouped = analysis.grouped(by_experiment)
    analysis.write_csv(output / "sample_metrics.csv", rows)
    analysis.write_csv(output / "overall_comparison.csv", overall)
    analysis.write_csv(output / "grouped_results.csv", grouped)

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    metrics = (
        ("mean_iou", "3D IoU"), ("mean_mse", "Density MSE"),
        ("mean_body_mse", "Body MSE"), ("mean_background_mse", "Background MSE"),
        ("mean_absolute_top_depth_error_m", "Top-depth error (m)"),
        ("mean_absolute_thickness_error_m", "Thickness error (m)"),
    )
    labels = [row["experiment"] for row in overall]
    for axis, (key, title) in zip(axes.flat, metrics):
        axis.bar(labels, [row[key] for row in overall], color=("#4c78a8", "#f58518", "#54a24b"))
        axis.set_title(title); axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "e10_ablation_metrics.png", dpi=180)
    plt.close(figure)

    readme = """# E10A / E10B / E10C comparison

All rows use the same canonical 100-sample test manifest and density threshold 0.1.
E10A isolates the revised architecture with uniform 50/50 body/background density loss.
E10B adds inverse-sensitivity weighting. E10C additionally adds data-weighted forward-gravity physics.

See `overall_comparison.csv` for aggregate density, geometry, compactness, lateral-localization,
and gravity metrics; `sample_metrics.csv` for paired sample-level values; and
`grouped_results.csv` for top-depth/thickness breakdowns.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print("E10 ablation comparison complete")
    for row in overall:
        print(row["experiment"], {key: row[key] for key in (
            "mean_iou", "mean_mse", "mean_absolute_top_depth_error_m",
            "mean_absolute_bottom_depth_error_m", "mean_absolute_thickness_error_m",
            "mean_gravity_rmse",
        )})


if __name__ == "__main__":
    main()
