"""Plot physical-geometry examples from a canonical single-plane dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from cnn_inversion_3d.dataset import find_repository_root
from cnn_inversion_3d.single_plane_review import SinglePlaneReviewConfig


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", default="test_manifest.csv")
    parser.add_argument("--examples", type=int, default=1)
    parser.add_argument(
        "--random-samples",
        type=int,
        default=None,
        help="Select this many manifest samples uniformly without replacement.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260727,
        help="Random-selection seed used with --random-samples.",
    )
    parser.add_argument("--sample-id", default=None)
    parser.add_argument(
        "--selection",
        action="append",
        choices=("central", "nearest-edge"),
        help="Repeat to generate both deterministic geometry examples.",
    )
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def select_geometry_rows(
    rows: list[dict[str, str]], selections: list[str]
) -> list[dict[str, str]]:
    """Select deterministic central and nearest-edge examples."""

    def center_distance(row: dict[str, str]) -> float:
        return float(
            np.hypot(
                float(row["center_x_m"]) - 320.0,
                float(row["center_y_m"]) - 320.0,
            )
        )

    def minimum_clearance(row: dict[str, str]) -> int:
        return min(
            int(row["x_start"]),
            64 - int(row["x_end"]),
            int(row["y_start"]),
            64 - int(row["y_end"]),
        )

    selected: list[dict[str, str]] = []
    for selection in selections:
        if selection == "central":
            selected.append(min(rows, key=lambda row: (center_distance(row), row["sample_id"])))
        else:
            selected.append(min(rows, key=lambda row: (minimum_clearance(row), row["sample_id"])))
    return selected


def plot_geometry_sample(
    *,
    dataset: Path,
    row: dict[str, str],
    output: Path,
    config: SinglePlaneReviewConfig | None = None,
) -> None:
    """Save one compact physical-geometry figure."""

    geometry = config or SinglePlaneReviewConfig()
    with np.load(dataset / row["relative_path"]) as sample:
        density = np.asarray(sample["density"])
        gravity = np.asarray(sample["gravity"])

    center_z = (int(row["z_start"]) + int(row["z_end"]) - 1) // 2
    center_y = (int(row["y_start"]) + int(row["y_end"]) - 1) // 2
    center_x = (int(row["x_start"]) + int(row["x_end"]) - 1) // 2
    model_extent = (0.0, 640.0, 0.0, 640.0)
    vertical_extent = (0.0, 640.0, 240.0, 0.0)
    gravity_extent = (-85.0, 715.0, -85.0, 715.0)
    figure = plt.figure(figsize=(16.0, 4.5), constrained_layout=True)
    grid = figure.add_gridspec(1, 4, width_ratios=(1.0, 1.4, 1.4, 1.0))
    axes = [figure.add_subplot(grid[0, index]) for index in range(4)]

    density_panels = (
        (density[center_z], model_extent, "lower", "Density plan", "X (m)", "Y (m)"),
        (density[:, center_y, :], vertical_extent, "upper", "X–Z section", "X (m)", "Depth (m)"),
        (density[:, :, center_x], vertical_extent, "upper", "Y–Z section", "Y (m)", "Depth (m)"),
    )
    density_image = None
    for axis, (values, extent, origin, title, xlabel, ylabel) in zip(
        axes[:3], density_panels
    ):
        density_image = axis.imshow(
            values,
            extent=extent,
            origin=origin,
            aspect="equal",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_xticks((0, 100, 200, 300, 400, 500, 640))
    axes[0].set_yticks((0, 100, 200, 300, 400, 500, 640))
    for axis in axes[1:3]:
        axis.set_yticks((0, 50, 100, 150, 200, 240))

    body_x0 = int(row["x_start"]) * geometry.dx_m
    body_x1 = int(row["x_end"]) * geometry.dx_m
    body_y0 = int(row["y_start"]) * geometry.dy_m
    body_y1 = int(row["y_end"]) * geometry.dy_m
    top = float(row["top_depth_m"])
    bottom = float(row["bottom_depth_m"])
    outlines = (
        Rectangle((body_x0, body_y0), body_x1 - body_x0, body_y1 - body_y0),
        Rectangle((body_x0, top), body_x1 - body_x0, bottom - top),
        Rectangle((body_y0, top), body_y1 - body_y0, bottom - top),
    )
    for axis, outline in zip(axes[:3], outlines):
        outline.set(fill=False, edgecolor="white", linewidth=1.0, alpha=0.85)
        axis.add_patch(outline)

    if density_image is None:
        raise RuntimeError("Density image was not created.")
    figure.colorbar(
        density_image,
        ax=axes[:3],
        location="bottom",
        shrink=0.72,
        pad=0.08,
        label="Density contrast (g/cm³)",
    )

    gravity_image = axes[3].imshow(
        gravity,
        extent=gravity_extent,
        origin="lower",
        aspect="equal",
        cmap="viridis",
        vmin=float(np.min(gravity)),
        vmax=float(np.max(gravity)),
    )
    axes[3].set_title("Surface Gz")
    axes[3].set_xlabel("X (m)")
    axes[3].set_ylabel("Y (m)")
    axes[3].set_xticks((-85, 100, 300, 500, 715))
    axes[3].set_yticks((-85, 100, 300, 500, 715))
    axes[3].add_patch(
        Rectangle(
            (0.0, 0.0), 640.0, 640.0, fill=False,
            edgecolor="white", linewidth=1.0, alpha=0.7,
        )
    )
    figure.colorbar(gravity_image, ax=axes[3], label="Gz (mGal)", pad=0.03)

    subtitle = (
        "Density: 24×64×64 | Model: 640×640×240 m\n"
        "Surface Gz: 81×81 | Observation area: 800×800 m | +z downward"
    )
    sample_text = (
        f"{row['sample_id']}  |  Top: {top:g} m  |  Bottom: {bottom:g} m  |  "
        f"Width: {float(row['width_x_m']):g} × {float(row['width_y_m']):g} m  |  "
        f"Thickness: {float(row['thickness_z_m']):g} m  |  "
        f"Density: {float(row['density_contrast']):.3f} g/cm³"
    )
    figure.suptitle("Canonical single-plane dataset geometry check", fontsize=15)
    figure.text(0.5, 0.92, subtitle, ha="center", va="top", fontsize=10)
    figure.text(
        0.5, 0.82, sample_text, ha="center", va="top", fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_gallery(
    *,
    dataset: Path,
    rows: list[dict[str, str]],
    output: Path,
    config: SinglePlaneReviewConfig | None = None,
) -> None:
    """Backward-compatible entry point for a single compact figure."""

    if len(rows) != 1:
        raise ValueError("Use --output-directory for multiple individual figures.")
    plot_geometry_sample(
        dataset=dataset, row=rows[0], output=output, config=config
    )


def main() -> None:
    """Load manifest examples and save individual figures."""

    arguments = build_argument_parser().parse_args()
    if arguments.examples < 1:
        raise ValueError("--examples must be at least one.")
    if arguments.random_samples is not None and arguments.random_samples < 1:
        raise ValueError("--random-samples must be at least one.")
    selection_modes = sum(
        (
            bool(arguments.selection),
            arguments.sample_id is not None,
            arguments.random_samples is not None,
        )
    )
    if selection_modes > 1:
        raise ValueError(
            "Use only one of --selection, --sample-id, or --random-samples."
        )
    root = find_repository_root()
    dataset = _resolve(root, arguments.dataset)
    with (dataset / arguments.manifest).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    if arguments.random_samples is not None:
        if arguments.random_samples > len(rows):
            raise ValueError(
                "--random-samples exceeds the selected manifest size."
            )
        indices = np.random.default_rng(arguments.seed).choice(
            len(rows), size=arguments.random_samples, replace=False
        )
        selected = [rows[int(index)] for index in indices]
    elif arguments.selection:
        selected = select_geometry_rows(rows, arguments.selection)
    elif arguments.sample_id is not None:
        selected = [row for row in rows if row["sample_id"] == arguments.sample_id]
        if not selected:
            raise KeyError(
                f"Sample {arguments.sample_id!r} is not in {arguments.manifest}."
            )
    else:
        selected = rows[: arguments.examples]
        if len(selected) < arguments.examples:
            raise ValueError("Manifest contains fewer rows than requested examples.")

    if arguments.output_directory is not None:
        if arguments.output is not None:
            raise ValueError("Use either --output-directory or --output, not both.")
        output_directory = _resolve(root, arguments.output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        for row in selected:
            output = output_directory / f"{row['sample_id']}_geometry.png"
            plot_geometry_sample(dataset=dataset, row=row, output=output)
            clearance = min(
                int(row["x_start"]), 64 - int(row["x_end"]),
                int(row["y_start"]), 64 - int(row["y_end"]),
            )
            print(
                f"Plotted {row['sample_id']}: {output} "
                f"(minimum horizontal clearance: {clearance} cells / "
                f"{clearance * 10} m)"
            )
        return

    if len(selected) != 1:
        raise ValueError("Multiple samples require --output-directory.")
    output = (
        _resolve(root, arguments.output)
        if arguments.output is not None
        else dataset / f"{selected[0]['sample_id']}_geometry.png"
    )
    plot_geometry_sample(dataset=dataset, row=selected[0], output=output)
    print(f"Plotted {selected[0]['sample_id']}: {output}")


if __name__ == "__main__":
    main()
