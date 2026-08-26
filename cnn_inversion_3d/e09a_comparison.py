"""Analysis-only paired comparison of E09 and E09A saved predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cnn_inversion_3d import failure_mode_analysis as analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/canonical_single_plane_train2000"))
    parser.add_argument("--e09", type=Path, default=Path("prediction_outputs/E09_canonical_single_plane_asymmetric_2d_unet_bf8"))
    parser.add_argument("--e09a", type=Path, default=Path("prediction_outputs/E09A_depth_supervision"))
    parser.add_argument("--output", type=Path, default=Path("analysis_outputs/E09_vs_E09A_depth_supervision"))
    parser.add_argument("--profile-samples", type=int, default=5)
    args = parser.parse_args()
    analysis.EXPERIMENTS = {"E09": str(args.e09), "E09A": str(args.e09a)}
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    rows, by = analysis.collect(Path.cwd(), args.dataset.resolve())
    overall = analysis.aggregate(by)
    analysis.write_csv(output / "sample_metrics.csv", rows)
    analysis.write_csv(output / "overall_comparison.csv", overall)
    analysis.write_csv(output / "grouped_results.csv", analysis.grouped(by))

    prediction_directories = {"E09": args.e09.resolve(), "E09A": args.e09a.resolve()}
    profile_directory = output / "depth_profiles"; profile_directory.mkdir(exist_ok=True)
    selected = by["E09"][: max(0, args.profile_samples)]
    depth_m = 5.0 + np.arange(24) * 10.0
    for reference in selected:
        sample_id = reference["sample_id"]
        figure, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True, constrained_layout=True)
        for axis, label in zip(axes, ("E09", "E09A")):
            row = next(item for item in by[label] if item["sample_id"] == sample_id)
            with np.load(prediction_directories[label] / row["prediction_path"]) as saved:
                truth = np.asarray(saved["true_density"]).squeeze()
                prediction = np.asarray(saved["predicted_density"]).squeeze()
            true_profile = np.sum(truth, axis=(1, 2)); predicted_profile = np.sum(prediction, axis=(1, 2))
            true_profile /= max(float(np.sum(true_profile)), 1e-8)
            predicted_profile /= max(float(np.sum(predicted_profile)), 1e-8)
            axis.plot(true_profile, depth_m, "k-o", label="True")
            axis.plot(predicted_profile, depth_m, "r-o", label="Predicted")
            axis.set(title=label, xlabel="Normalized density fraction")
            axis.grid(alpha=0.25); axis.invert_yaxis(); axis.legend()
        axes[0].set_ylabel("Depth-bin center (m)")
        figure.suptitle(f"{sample_id}: normalized density-depth profile")
        figure.savefig(profile_directory / f"{sample_id}_depth_profile.png", dpi=180)
        plt.close(figure)

    (output / "README.md").write_text(
        "# E09 vs E09A Depth-Supervision Comparison\n\n"
        "This analysis compares the same 100 held-out samples. E09A uses the identical E09 "
        "architecture and differs only by normalized depth-profile and Z-center supervision.\n",
        encoding="utf-8",
    )
    for row in overall:
        print(row["experiment"], {key: row[key] for key in (
            "mean_mse", "mean_iou", "mean_depth_profile_mse", "mean_absolute_z_center_error_m",
            "mean_absolute_top_depth_error_m", "mean_absolute_bottom_depth_error_m",
            "mean_absolute_thickness_error_m",
        )})


if __name__ == "__main__":
    main()
