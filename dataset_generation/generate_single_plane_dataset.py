"""Generate the additive 81 x 81 single-surface Gz training dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np

from cnn_inversion_3d.single_plane_review import SinglePlaneReviewConfig
from forward_modeling.matlab_fwd3d.forward_model import FWD3DGravityForwardModel
from forward_modeling.matlab_fwd3d.grid_adapter import model_grid_from_grid_spec
from forward_modeling.matlab_fwd3d.receivers import ReceiverGrid


SEED = 20260727


def build_argument_parser() -> argparse.ArgumentParser:
    """Build single-plane dataset-generation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--split-seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _sample_body(
    rng: np.random.Generator,
    config: SinglePlaneReviewConfig,
) -> dict[str, int | float]:
    """Sample one grid-aligned body fully inside the density domain."""

    width_x = int(rng.integers(
        int(config.minimum_width_x_m / config.dx_m),
        int(config.maximum_width_x_m / config.dx_m) + 1,
    ))
    width_y = int(rng.integers(
        int(config.minimum_width_y_m / config.dy_m),
        int(config.maximum_width_y_m / config.dy_m) + 1,
    ))
    thickness = int(rng.integers(
        int(config.minimum_thickness_m / config.dz_m),
        int(config.maximum_thickness_m / config.dz_m) + 1,
    ))
    top = int(rng.integers(
        int(config.minimum_top_depth_m / config.dz_m),
        int(config.maximum_top_depth_m / config.dz_m) + 1,
    ))
    minimum_start = config.horizontal_margin_cells
    maximum_x_start = config.nx - config.horizontal_margin_cells - width_x
    maximum_y_start = config.ny - config.horizontal_margin_cells - width_y
    if maximum_x_start < minimum_start or maximum_y_start < minimum_start:
        raise ValueError("Body dimensions are incompatible with the margin.")
    x_start = int(rng.integers(minimum_start, maximum_x_start + 1))
    y_start = int(rng.integers(minimum_start, maximum_y_start + 1))
    density = float(rng.uniform(
        config.minimum_density_contrast_g_cm3,
        config.maximum_density_contrast_g_cm3,
    ))
    return {
        "x_start": x_start,
        "x_end": x_start + width_x,
        "y_start": y_start,
        "y_end": y_start + width_y,
        "z_start": top,
        "z_end": top + thickness,
        "width_x": width_x,
        "width_y": width_y,
        "thickness_z": thickness,
        "top_depth_m": top * config.dz_m,
        "bottom_depth_m": (top + thickness) * config.dz_m,
        "width_x_m": width_x * config.dx_m,
        "width_y_m": width_y * config.dy_m,
        "thickness_z_m": thickness * config.dz_m,
        "center_x_m": (
            x_start + (width_x - 1) / 2.0
        ) * config.dx_m,
        "center_y_m": (
            y_start + (width_y - 1) / 2.0
        ) * config.dy_m,
        "center_depth_m": (
            top + thickness / 2.0
        ) * config.dz_m,
        "density_contrast": density,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows using their stable first-row field order."""

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_canonical_dataset(
    output: Path,
    rows: list[dict[str, object]],
    config: SinglePlaneReviewConfig,
) -> dict[str, object]:
    """Fail unless every generated sample satisfies canonical geometry."""

    left = []
    right = []
    lower_y = []
    upper_y = []
    for row in rows:
        sample_id = str(row["sample_id"])
        with np.load(output / str(row["relative_path"])) as sample:
            density_shape = sample["density"].shape
            gravity_shape = sample["gravity"].shape
        checks = {
            "density shape": density_shape == config.density_shape,
            "gravity shape": gravity_shape == config.gravity_shape,
            "top depth": config.minimum_top_depth_m <= float(row["top_depth_m"]) <= config.maximum_top_depth_m,
            "thickness": config.minimum_thickness_m <= float(row["thickness_z_m"]) <= config.maximum_thickness_m,
            "width X": config.minimum_width_x_m <= float(row["width_x_m"]) <= config.maximum_width_x_m,
            "width Y": config.minimum_width_y_m <= float(row["width_y_m"]) <= config.maximum_width_y_m,
            "density contrast": config.minimum_density_contrast_g_cm3 <= float(row["density_contrast"]) <= config.maximum_density_contrast_g_cm3,
            "bottom depth": float(row["bottom_depth_m"]) <= config.maximum_bottom_depth_m,
            "left margin": int(row["x_start"]) >= config.horizontal_margin_cells,
            "right margin": int(row["x_end"]) <= config.nx - config.horizontal_margin_cells,
            "lower-Y margin": int(row["y_start"]) >= config.horizontal_margin_cells,
            "upper-Y margin": int(row["y_end"]) <= config.ny - config.horizontal_margin_cells,
        }
        failures = [name for name, passed in checks.items() if not passed]
        if failures:
            raise RuntimeError(
                f"Canonical validation failed for {sample_id}: {failures}"
            )
        left.append(int(row["x_start"]))
        right.append(config.nx - int(row["x_end"]))
        lower_y.append(int(row["y_start"]))
        upper_y.append(config.ny - int(row["y_end"]))

    def value_range(field: str) -> list[float]:
        values = [float(row[field]) for row in rows]
        return [min(values), max(values)]

    clearances = {
        "minimum_left_clearance_cells": min(left),
        "minimum_right_clearance_cells": min(right),
        "minimum_lower_y_clearance_cells": min(lower_y),
        "minimum_upper_y_clearance_cells": min(upper_y),
    }
    summary: dict[str, object] = {
        "passed": True,
        "samples_validated": len(rows),
        **clearances,
        **{
            key.replace("_cells", "_m"): value * config.dx_m
            for key, value in clearances.items()
        },
        "top_depth_m_range": value_range("top_depth_m"),
        "bottom_depth_m_range": value_range("bottom_depth_m"),
        "width_x_m_range": value_range("width_x_m"),
        "width_y_m_range": value_range("width_y_m"),
        "thickness_m_range": value_range("thickness_z_m"),
        "density_contrast_g_cm3_range": value_range("density_contrast"),
        "density_shape": list(config.density_shape),
        "gravity_shape": list(config.gravity_shape),
    }
    print("Canonical geometry validation passed")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """Generate, deterministically split, and summarize the dataset."""

    args = build_argument_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset already exists:\n{output}")
        shutil.rmtree(output)
    samples = output / "samples"
    samples.mkdir(parents=True)
    counts = (args.train_count, args.validation_count, args.test_count)
    if any(value < 1 for value in counts):
        raise ValueError("All split counts must be positive.")
    total = sum(counts)
    config = SinglePlaneReviewConfig()
    grid = config.grid_spec()
    receivers = ReceiverGrid(
        x=config.observation_x_m,
        y=config.observation_y_m,
        z=np.asarray([config.observation_z_m], dtype=np.float64),
    )
    forward = FWD3DGravityForwardModel(
        model_grid=model_grid_from_grid_spec(grid),
        receiver_grid=receivers,
        channel=config.gravity_channel,
        receiver_chunk_size=config.receiver_chunk_size,
    )
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    start_time = perf_counter()
    for index in range(total):
        body = _sample_body(rng, config)
        density = np.zeros(config.density_shape, dtype=np.float32)
        density[
            int(body["z_start"]):int(body["z_end"]),
            int(body["y_start"]):int(body["y_end"]),
            int(body["x_start"]):int(body["x_end"]),
        ] = float(body["density_contrast"])
        gravity = np.asarray(forward.calculate(density)[0], dtype=np.float32)
        filename = f"sample_{index:06d}.npz"
        np.savez_compressed(samples / filename, gravity=gravity, density=density)
        rows.append({
            "sample_index": index,
            "sample_id": f"sample_{index:06d}",
            "relative_path": f"samples/{filename}",
            **body,
            "gravity_minimum_mgal": float(np.min(gravity)),
            "gravity_maximum_mgal": float(np.max(gravity)),
            "gravity_mean_mgal": float(np.mean(gravity)),
            "gravity_std_mgal": float(np.std(gravity)),
        })
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == total:
            print(f"[{index + 1:,}/{total:,}] generated")
    _write_csv(output / "manifest.csv", rows)
    order = np.random.default_rng(args.split_seed).permutation(total)
    train_end = args.train_count
    validation_end = train_end + args.validation_count
    split_indices = {
        "train": order[:train_end],
        "validation": order[train_end:validation_end],
        "test": order[validation_end:],
    }
    for name, indices in split_indices.items():
        _write_csv(output / f"{name}_manifest.csv", [rows[int(i)] for i in indices])
    validation_summary = validate_canonical_dataset(output, rows, config)
    training_absolute = np.concatenate([
        np.abs(np.load(output / rows[int(i)]["relative_path"])["gravity"]).ravel()
        for i in split_indices["train"]
    ])
    scale = float(np.percentile(training_absolute, 99.0))
    distribution = {
        "source_split": "training only",
        "number_of_samples": args.train_count,
        "absolute_maximum": float(np.max(training_absolute)),
        "absolute_percentile_99": scale,
        "standard_deviation": float(np.std(training_absolute)),
        "recommended_absolute_max_scale": float(np.max(training_absolute)),
        "recommended_percentile_99_scale": scale,
        "recommended_standard_deviation_scale": float(
            np.std(training_absolute)
        ),
        "normalization_method": "percentile_99",
        "gravity_scale": scale,
        "per_sample_normalization": False,
    }
    (output / "training_distribution.json").write_text(
        json.dumps(distribution, indent=2), encoding="utf-8"
    )
    metadata = {
        "dataset_type": "single_plane_positive_rectangular_body",
        "scientific_change": "multi-height gravity observations to one horizontal surface gravity plane",
        "density_array_order": "density[z, y, x]",
        "gravity_array_order": "gravity[y, x]",
        "dataset_geometry_version": config.dataset_geometry_version,
        "density_shape": list(config.density_shape),
        "gravity_shape": list(config.gravity_shape),
        "cnn_gravity_shape": list(config.cnn_gravity_shape),
        "gravity_component": "Gz",
        "gravity_channel": config.gravity_channel,
        "gravity_unit": "mGal",
        "density_unit": "g/cm3",
        "number_of_samples": total,
        "split_counts": dict(zip(("train", "validation", "test"), counts)),
        "generation_seed": args.seed,
        "split_seed": args.split_seed,
        "normalization": distribution,
        "geometry_validation": validation_summary,
        "observation_x_coordinates_m": config.observation_x_m.tolist(),
        "observation_y_coordinates_m": config.observation_y_m.tolist(),
        "observation_z_m": config.observation_z_m,
        "body_ranges": {
            "top_depth_m": [config.minimum_top_depth_m, config.maximum_top_depth_m],
            "width_x_m": [config.minimum_width_x_m, config.maximum_width_x_m],
            "width_y_m": [config.minimum_width_y_m, config.maximum_width_y_m],
            "thickness_m": [config.minimum_thickness_m, config.maximum_thickness_m],
            "density_contrast_g_cm3": [config.minimum_density_contrast_g_cm3, config.maximum_density_contrast_g_cm3],
            "maximum_bottom_depth_m": config.maximum_bottom_depth_m,
            "horizontal_margin_cells": config.horizontal_margin_cells,
            "horizontal_margin_m": config.horizontal_margin_m,
            "bodies_per_sample": 1,
        },
        "canonical_geometry": config.to_metadata(),
        "configuration": {
            "receiver_chunk_size": config.receiver_chunk_size,
            "receivers": {
                "number_of_levels": 1,
                "first_level_z": config.observation_z_m,
                "level_spacing": 1.0,
            },
        },
        "grid": {
            "nx": grid.nx, "ny": grid.ny, "nz": grid.nz,
            "x_min": grid.x_min, "x_max": grid.x_max,
            "y_min": grid.y_min, "y_max": grid.y_max,
            "z_min": grid.z_min, "z_max": grid.z_max,
            "dx": grid.dx, "dy": grid.dy, "dz": grid.dz,
        },
        "generation_seconds": perf_counter() - start_time,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Dataset written to {output}")


if __name__ == "__main__":
    main()
