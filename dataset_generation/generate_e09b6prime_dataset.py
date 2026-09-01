"""Generate the balanced canonical E09B-6-prime train/validation dataset."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from cnn_inversion_3d.single_plane_review import SinglePlaneReviewConfig
from dataset_generation.generate_single_plane_dataset import (
    _write_csv, validate_canonical_dataset,
)
from forward_modeling.matlab_fwd3d.forward_model import FWD3DGravityForwardModel
from forward_modeling.matlab_fwd3d.grid_adapter import model_grid_from_grid_spec
from forward_modeling.matlab_fwd3d.receivers import ReceiverGrid

SIZE_LIMITS = (288, 560)
SIZE_NAMES = ("small", "medium", "large")
DENSITY_NAMES = ("low", "medium", "high")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=Path("datasets/canonical_single_plane_train10000_balanced_size_density"))
    result.add_argument("--reference-dataset", type=Path, default=Path("datasets/canonical_single_plane_train2000"))
    result.add_argument("--train-count", type=int, default=10000)
    result.add_argument("--validation-count", type=int, default=1000)
    result.add_argument("--seed", type=int, default=20260901)
    result.add_argument("--overwrite", action="store_true")
    return result


def valid_dimension_combinations(config: SinglePlaneReviewConfig) -> dict[str, list[tuple[int, int, int]]]:
    combinations = [(x, y, z)
                    for x in range(int(config.minimum_width_x_m/config.dx_m), int(config.maximum_width_x_m/config.dx_m)+1)
                    for y in range(int(config.minimum_width_y_m/config.dy_m), int(config.maximum_width_y_m/config.dy_m)+1)
                    for z in range(int(config.minimum_thickness_m/config.dz_m), int(config.maximum_thickness_m/config.dz_m)+1)]
    low, high = SIZE_LIMITS
    return {
        "small": [item for item in combinations if np.prod(item) <= low],
        "medium": [item for item in combinations if low < np.prod(item) <= high],
        "large": [item for item in combinations if np.prod(item) > high],
    }


def balanced_schedule(count: int, rng: np.random.Generator) -> list[tuple[str, str]]:
    cells = [(size, density) for size in SIZE_NAMES for density in DENSITY_NAMES]
    schedule = [cells[index % len(cells)] for index in range(count)]
    rng.shuffle(schedule)
    return schedule


def sample_body(rng: np.random.Generator, config: SinglePlaneReviewConfig,
                dimensions: dict[str, list[tuple[int, int, int]]],
                size_group: str, density_group: str) -> dict[str, Any]:
    choices = dimensions[size_group]
    width_x, width_y, thickness = choices[int(rng.integers(0, len(choices)))]
    top = int(rng.integers(int(config.minimum_top_depth_m/config.dz_m),
                           int(config.maximum_top_depth_m/config.dz_m)+1))
    x_start = int(rng.integers(config.horizontal_margin_cells,
                               config.nx-config.horizontal_margin_cells-width_x+1))
    y_start = int(rng.integers(config.horizontal_margin_cells,
                               config.ny-config.horizontal_margin_cells-width_y+1))
    density_edges = np.linspace(config.minimum_density_contrast_g_cm3,
                                config.maximum_density_contrast_g_cm3, 4)
    density_index = DENSITY_NAMES.index(density_group)
    density = float(rng.uniform(density_edges[density_index], density_edges[density_index+1]))
    return {
        "x_start": x_start, "x_end": x_start+width_x,
        "y_start": y_start, "y_end": y_start+width_y,
        "z_start": top, "z_end": top+thickness,
        "width_x": width_x, "width_y": width_y, "thickness_z": thickness,
        "top_depth_m": top*config.dz_m, "bottom_depth_m": (top+thickness)*config.dz_m,
        "width_x_m": width_x*config.dx_m, "width_y_m": width_y*config.dy_m,
        "thickness_z_m": thickness*config.dz_m,
        "center_x_m": (x_start+(width_x-1)/2)*config.dx_m,
        "center_y_m": (y_start+(width_y-1)/2)*config.dy_m,
        "center_depth_m": (top+thickness/2)*config.dz_m,
        "density_contrast": density, "body_volume_cells": width_x*width_y*thickness,
        "body_size_group": size_group, "density_group": density_group,
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    volume = np.asarray([float(row["body_volume_cells"]) for row in rows])
    density = np.asarray([float(row["density_contrast"]) for row in rows])
    counts = {name: sum(row["body_size_group"] == name for row in rows) for name in SIZE_NAMES}
    return {"split": split, "number_of_samples": len(rows),
            "minimum_body_volume_cells": float(volume.min()), "maximum_body_volume_cells": float(volume.max()),
            "mean_body_volume_cells": float(volume.mean()), "median_body_volume_cells": float(np.median(volume)),
            **{f"{name}_count": counts[name] for name in SIZE_NAMES},
            "minimum_density_contrast": float(density.min()), "maximum_density_contrast": float(density.max()),
            "mean_density_contrast": float(density.mean()), "median_density_contrast": float(np.median(density))}


def review(output: Path, train: list[dict[str, Any]], validation: list[dict[str, Any]], review_dir: Path) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    summaries = [summarize(train, "train"), summarize(validation, "validation")]
    _write_csv(review_dir/"dataset_summary.csv", summaries)
    size_rows = [{"split": split, "body_size_group": group,
                  "count": sum(row["body_size_group"] == group for row in rows)}
                 for split, rows in (("train", train), ("validation", validation)) for group in SIZE_NAMES]
    density_rows = [{"split": split, "density_group": group,
                     "count": sum(row["density_group"] == group for row in rows)}
                    for split, rows in (("train", train), ("validation", validation)) for group in DENSITY_NAMES]
    joint_rows = [{"split": split, "body_size_group": size, "density_group": density,
                   "count": sum(row["body_size_group"] == size and row["density_group"] == density for row in rows)}
                  for split, rows in (("train", train), ("validation", validation))
                  for size in SIZE_NAMES for density in DENSITY_NAMES]
    _write_csv(review_dir/"body_size_distribution.csv", size_rows)
    _write_csv(review_dir/"density_distribution.csv", density_rows)
    _write_csv(review_dir/"size_density_distribution.csv", joint_rows)
    for summary in summaries:
        counts = [summary[f"{name}_count"] for name in SIZE_NAMES]
        if min(counts) < 0.30*summary["number_of_samples"]:
            raise RuntimeError(f"{summary['split']} size category is severely underrepresented: {counts}")
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0,0].hist([row["body_volume_cells"] for row in train], bins=40); axes[0,0].set_title("Training body volume")
    axes[0,1].hist([row["density_contrast"] for row in train], bins=40); axes[0,1].set_title("Training density contrast")
    axes[1,0].scatter([row["body_volume_cells"] for row in train], [row["density_contrast"] for row in train], s=3, alpha=.2)
    axes[1,0].set(xlabel="Body volume (cells)", ylabel="Density (g/cm³)", title="Independent size-density coverage")
    axes[1,1].bar(SIZE_NAMES, [sum(row["body_size_group"] == name for row in train) for name in SIZE_NAMES]); axes[1,1].set_title("Training size counts")
    figure.savefig(review_dir/"dataset_distributions.png", dpi=180); plt.close(figure)
    (review_dir/"README.md").write_text(
        "# E09B-6-prime Dataset Review\n\nSize strata are derived from all 1,183 valid grid-aligned dimension combinations: "
        "small <=288 cells, medium 289–560 cells, and large >560 cells. Density 0.2–1.0 g/cm³ is split into three equal-width bins and crossed with size, giving balanced 3x3 coverage. Test labels are not used.\n",
        encoding="utf-8")


def main() -> None:
    args = parser().parse_args(); root = Path(__file__).resolve().parents[1]
    output = (args.output if args.output.is_absolute() else root/args.output).resolve()
    reference = (args.reference_dataset if args.reference_dataset.is_absolute() else root/args.reference_dataset).resolve()
    review_dir = root/"analysis_outputs"/"E09B_6prime_dataset_review"
    if output.exists():
        if not args.overwrite: raise FileExistsError(f"Dataset already exists:\n{output}")
        shutil.rmtree(output)
    output.joinpath("samples").mkdir(parents=True)
    config = SinglePlaneReviewConfig(); grid = config.grid_spec()
    forward = FWD3DGravityForwardModel(model_grid=model_grid_from_grid_spec(grid),
        receiver_grid=ReceiverGrid(x=config.observation_x_m, y=config.observation_y_m,
                                   z=np.asarray([config.observation_z_m], np.float64)),
        channel=config.gravity_channel, receiver_chunk_size=config.receiver_chunk_size)
    rng = np.random.default_rng(args.seed); dimensions = valid_dimension_combinations(config)
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    start = perf_counter(); global_index = 0
    for split, count in (("train", args.train_count), ("validation", args.validation_count)):
        for size_group, density_group in balanced_schedule(count, rng):
            body = sample_body(rng, config, dimensions, size_group, density_group)
            density = np.zeros(config.density_shape, np.float32)
            density[body["z_start"]:body["z_end"], body["y_start"]:body["y_end"], body["x_start"]:body["x_end"]] = body["density_contrast"]
            gravity = np.asarray(forward.calculate(density)[0], np.float32)
            sample_id = f"prime_{global_index:06d}"; relative = f"samples/{sample_id}.npz"
            np.savez_compressed(output/relative, gravity=gravity, density=density)
            rows_by_split[split].append({"sample_index": global_index, "sample_id": sample_id,
                "relative_path": relative, **body, "gravity_minimum_mgal": float(gravity.min()),
                "gravity_maximum_mgal": float(gravity.max()), "gravity_mean_mgal": float(gravity.mean()),
                "gravity_std_mgal": float(gravity.std())})
            global_index += 1
            if global_index == 1 or global_index % 100 == 0: print(f"[{global_index:,}/{args.train_count+args.validation_count:,}] generated")
    test_rows = read_rows(reference/"test_manifest.csv")
    for row in test_rows:
        source = reference/row["relative_path"]; destination = output/row["relative_path"]
        if destination.exists(): raise RuntimeError(f"Test sample collision: {destination}")
        shutil.copy2(source, destination)
    _write_csv(output/"train_manifest.csv", rows_by_split["train"])
    _write_csv(output/"validation_manifest.csv", rows_by_split["validation"])
    _write_csv(output/"test_manifest.csv", test_rows)
    all_rows = [*rows_by_split["train"], *rows_by_split["validation"], *test_rows]
    _write_csv(output/"manifest.csv", all_rows)
    validation = validate_canonical_dataset(output, all_rows, config)
    training_values = np.concatenate([np.abs(np.load(output/row["relative_path"])["gravity"]).ravel()
                                      for row in rows_by_split["train"]])
    distribution = {"source_split": "training only", "number_of_samples": args.train_count,
        "absolute_maximum": float(training_values.max()), "absolute_percentile_99": float(np.percentile(training_values,99)),
        "standard_deviation": float(training_values.std()), "recommended_absolute_max_scale": float(training_values.max()),
        "recommended_percentile_99_scale": float(np.percentile(training_values,99)),
        "recommended_standard_deviation_scale": float(training_values.std()), "normalization_method": "percentile_99",
        "gravity_scale": float(np.percentile(training_values,99)), "per_sample_normalization": False}
    (output/"training_distribution.json").write_text(json.dumps(distribution,indent=2),encoding="utf-8")
    summaries = {split: summarize(rows,split) for split,rows in rows_by_split.items()}
    reference_metadata = json.loads((reference/"metadata.json").read_text(encoding="utf-8"))
    metadata = {**reference_metadata,
        "dataset_name": output.name, "dataset_type": "canonical_single_plane_balanced_size_density",
        "generation_seed": args.seed, "split_counts": {"train":args.train_count,"validation":args.validation_count,"test":len(test_rows)},
        "number_of_samples": args.train_count + args.validation_count + len(test_rows),
        "test_set_source": str(reference/"test_manifest.csv"), "test_manifest_reused_byte_identical_samples": True,
        "scientific_variable": "training dataset size and balanced size-density coverage",
        "parent_reference_experiment": "E09B-6", "density_array_order":"density[z, y, x]", "gravity_array_order":"gravity[y, x]",
        "size_definitions_cells":{"small":"32–288","medium":"289–560","large":"561–2048"},
        "size_threshold_derivation":"tertiles over all valid width_x*width_y*thickness dimension combinations",
        "density_range_g_cm3":[0.2,1.0], "density_strata":"three equal-width intervals crossed independently with size",
        "source_geometry_limits":config.to_metadata(), "split_summaries":summaries,
        "normalization":distribution,"geometry_validation":validation,"generation_seconds":perf_counter()-start,
        "generation_module":"dataset_generation.generate_e09b6prime_dataset"}
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    review(output, rows_by_split["train"], rows_by_split["validation"], review_dir)
    print(f"Dataset written to {output}")


if __name__ == "__main__": main()
