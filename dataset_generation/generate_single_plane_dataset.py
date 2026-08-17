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

    width_x = int(rng.integers(4, 17))
    width_y = int(rng.integers(4, 17))
    thickness = int(rng.integers(2, 9))
    top = int(rng.integers(2, 9))
    x_start = int(rng.integers(0, config.nx - width_x + 1))
    y_start = int(rng.integers(0, config.ny - width_y + 1))
    density = float(rng.uniform(0.2, 1.0))
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
        "density_shape": list(config.density_shape),
        "gravity_shape": [81, 81],
        "cnn_gravity_shape": [81, 81, 1],
        "gravity_component": "Gz",
        "gravity_channel": 4,
        "gravity_unit": "mGal",
        "density_unit": "g/cm3",
        "number_of_samples": total,
        "split_counts": dict(zip(("train", "validation", "test"), counts)),
        "generation_seed": args.seed,
        "split_seed": args.split_seed,
        "normalization": distribution,
        "observation_x_coordinates_m": config.observation_x_m.tolist(),
        "observation_y_coordinates_m": config.observation_y_m.tolist(),
        "observation_z_m": config.observation_z_m,
        "body_ranges": {
            "top_depth_m": [20, 80], "width_x_m": [40, 160],
            "width_y_m": [40, 160], "thickness_m": [20, 80],
            "density_contrast_g_cm3": [0.2, 1.0],
            "maximum_bottom_depth_m": 160, "bodies_per_sample": 1,
        },
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
