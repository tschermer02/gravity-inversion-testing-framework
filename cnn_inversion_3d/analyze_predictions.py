from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Combine prediction metrics with dataset geometry and "
            "summarize held-out reconstruction performance."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset directory.",
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Prediction-output directory.",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default="prediction_metrics.csv",
        help="Prediction metrics filename.",
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default="test_manifest.csv",
        help="Dataset manifest filename.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional combined CSV output path.",
    )

    parser.add_argument(
        "--evaluate-gravity-consistency",
        action="store_true",
        help=(
            "Forward model predicted densities and calculate optional "
            "CNN gravity-consistency outputs."
        ),
    )

    parser.add_argument(
        "--save-gravity-volumes",
        action="store_true",
        help=(
            "Save true, recovered, and residual gravity NPY volumes "
            "during gravity-consistency evaluation."
        ),
    )

    parser.add_argument(
        "--gravity-comparison-receivers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Receiver indices included in gravity comparison figures. "
            "Default: shallowest, middle, and deepest."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Recompute cached gravity-consistency outputs. Existing "
            "density-only analysis behavior is unchanged."
        ),
    )

    return parser


def find_repository_root() -> Path:
    """
    Return the repository root.
    """

    return Path(__file__).resolve().parents[1]

def load_table_rows(
    path: Path,
) -> list[dict[str, Any]]:
    """
    Load rows from a CSV or JSON file.

    Parameters
    ----------
    path
        Input table path. Supported suffixes are ``.csv`` and ``.json``.

    Returns
    -------
    list of dict
        Parsed table rows.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the file format or structure is unsupported.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Input file does not exist:\n{path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".csv":
        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as input_file:
            rows = list(
                csv.DictReader(
                    input_file
                )
            )

        if not rows:
            raise ValueError(
                f"CSV file contains no rows: {path}"
            )

        return [
            dict(row)
            for row in rows
        ]

    if suffix == ".json":
        with path.open(
            "r",
            encoding="utf-8",
        ) as input_file:
            loaded = json.load(
                input_file
            )

        if isinstance(
            loaded,
            list,
        ):
            rows = loaded

        elif isinstance(
            loaded,
            dict,
        ):
            if isinstance(
                loaded.get("samples"),
                list,
            ):
                rows = loaded[
                    "samples"
                ]

            elif isinstance(
                loaded.get("metrics"),
                list,
            ):
                rows = loaded[
                    "metrics"
                ]

            elif isinstance(
                loaded.get("results"),
                list,
            ):
                rows = loaded[
                    "results"
                ]

            else:
                raise ValueError(
                    "JSON metrics file must contain a list directly or "
                    "a list under 'samples', 'metrics', or 'results'. "
                    f"Available keys: {list(loaded.keys())}"
                )

        else:
            raise ValueError(
                "JSON metrics file must contain a list or dictionary."
            )

        normalized_rows: list[
            dict[str, Any]
        ] = []

        for row_index, row in enumerate(
            rows
        ):
            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    "Every JSON metrics entry must be an object. "
                    f"Entry {row_index} has type "
                    f"{type(row).__name__}."
                )

            normalized_rows.append(
                dict(row)
            )

        if not normalized_rows:
            raise ValueError(
                f"JSON file contains no metric rows: {path}"
            )

        return normalized_rows

    raise ValueError(
        "Unsupported input format. Expected .csv or .json, "
        f"received: {path.suffix}"
    )

def resolve_path(
    *,
    repository_root: Path,
    path: Path,
) -> Path:
    """
    Resolve a path relative to the repository root.
    """

    if not path.is_absolute():
        path = repository_root / path

    return path.resolve()


def load_csv_rows(
    path: Path,
) -> list[dict[str, str]]:
    """
    Load rows from a CSV file.

    Parameters
    ----------
    path
        CSV path.

    Returns
    -------
    list of dict
        CSV rows.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file does not exist:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        rows = list(
            csv.DictReader(
                csv_file
            )
        )

    if not rows:
        raise ValueError(
            f"CSV file contains no rows: {path}"
        )

    return rows


def normalize_sample_identifier(
    value: Any,
) -> int:
    """
    Convert a numeric or filename-style sample identifier to an integer.

    Parameters
    ----------
    value
        Identifier such as ``88``, ``000088``, or ``sample_000088``.

    Returns
    -------
    int
        Integer sample identifier.
    """

    if isinstance(value, bool):
        raise ValueError(
            "A Boolean value is not a valid sample identifier."
        )

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"Sample identifier must be an integer, received {value}."
            )
        return int(value)

    normalized = str(value).strip()

    if normalized.startswith(
        "sample_"
    ):
        normalized = normalized[
            len("sample_"):
        ]

    if normalized.endswith(
        ".npz"
    ):
        normalized = normalized[
            :-4
        ]

    return int(
        normalized
    )


def find_sample_id(
    row: dict[str, Any],
) -> int:
    """
    Extract the sample ID from a metrics or manifest row.
    """

    # Prediction JSON records may use ``sample_index`` for their
    # sequential position in the evaluated subset.  In that format,
    # ``sample_path`` contains the actual dataset sample identifier and
    # must take precedence.
    possible_path_fields = (
        "sample_path",
        "relative_path",
    )

    for field in possible_path_fields:
        value = row.get(
            field
        )

        if value is not None and str(value).strip():
            return normalize_sample_identifier(
                Path(
                    str(value)
                ).stem
            )

    possible_fields = (
        "sample_index",
        "sample",
        "sample_id",
        "sample_name",
        "filename",
    )

    for field in possible_fields:
        value = row.get(
            field
        )

        if value is not None and str(value).strip():
            return normalize_sample_identifier(
                value
            )

    raise KeyError(
        "Could not determine sample identifier from row."
    )


def get_float(
    row: dict[str, Any],
    *field_names: str,
) -> float:
    """
    Return the first available numeric field from a row.

    Parameters
    ----------
    row
        CSV or JSON row.
    *field_names
        Candidate field names checked in order.

    Returns
    -------
    float
        Parsed numeric value.

    Raises
    ------
    KeyError
        If no requested field contains a value.
    ValueError
        If a populated field cannot be converted to a float.
    """

    for field_name in field_names:
        value = row.get(
            field_name
        )

        if value is None:
            continue

        if isinstance(
            value,
            bool,
        ):
            raise ValueError(
                f"Field '{field_name}' contains a Boolean value."
            )

        if isinstance(
            value,
            int | float,
        ):
            return float(
                value
            )

        if isinstance(
            value,
            str,
        ):
            stripped_value = value.strip()

            if not stripped_value:
                continue

            try:
                return float(
                    stripped_value
                )
            except ValueError as error:
                raise ValueError(
                    f"Field '{field_name}' contains a nonnumeric value: "
                    f"{value!r}."
                ) from error

        raise ValueError(
            f"Field '{field_name}' has unsupported type "
            f"{type(value).__name__}."
        )

    raise KeyError(
        "None of the requested fields contains a numeric value: "
        f"{field_names}"
    )

def combine_rows(
    *,
    manifest_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """
    Combine prediction metrics with dataset geometry.

    Parameters
    ----------
    manifest_rows
        Test-manifest rows.
    metric_rows
        Prediction-metric rows.

    Returns
    -------
    list of dict
        Combined analysis rows.
    """

    manifest_by_id = {
        find_sample_id(
            row
        ): row
        for row in manifest_rows
    }

    combined_rows: list[
        dict[str, Any]
    ] = []

    for metric_row in metric_rows:
        sample_id = find_sample_id(
            metric_row
        )

        if sample_id not in manifest_by_id:
            raise KeyError(
                f"Sample {sample_id} does not occur in the manifest."
            )

        manifest_row = manifest_by_id[
            sample_id
        ]

        width_x = int(
            manifest_row[
                "width_x"
            ]
        )

        width_y = int(
            manifest_row[
                "width_y"
            ]
        )

        thickness_z = int(
            manifest_row[
                "thickness_z"
            ]
        )

        body_volume_cells = (
            width_x
            * width_y
            * thickness_z
        )

        if (
            manifest_row.get("center_x_m", "").strip()
            and manifest_row.get("center_y_m", "").strip()
        ):
            center_x_m = get_float(manifest_row, "center_x_m")
            center_y_m = get_float(manifest_row, "center_y_m")
        else:
            # E05 single-plane manifests store exact occupied-cell slices.
            # Cell centers are 0, 10, ..., 630 m, so the physical center of
            # [start:end) is the midpoint of start and end - 1.
            center_x_m = 5.0 * (
                get_float(manifest_row, "x_start")
                + get_float(manifest_row, "x_end")
                - 1.0
            )
            center_y_m = 5.0 * (
                get_float(manifest_row, "y_start")
                + get_float(manifest_row, "y_end")
                - 1.0
            )

        survey_center_x_m = 315.0
        survey_center_y_m = 315.0

        center_distance_m = float(
            np.hypot(
                center_x_m
                - survey_center_x_m,
                center_y_m
                - survey_center_y_m,
            )
        )

        combined_rows.append(
            {
                "sample_index": sample_id,
                "mse": get_float(
                    metric_row,
                    "mse",
                    "MSE",
                ),
                "iou": get_float(
                    metric_row,
                    "iou",
                    "IoU",
                ),
                "dice": get_float(
                    metric_row,
                    "dice",
                    "Dice",
                ),
                "predicted_max": get_float(
                    metric_row,
                    "predicted_max",
                    "predicted_maximum",
                    "prediction_maximum",
                ),
                "top_depth_m": get_float(
                    manifest_row,
                    "top_depth_m",
                ),
                "bottom_depth_m": get_float(
                    manifest_row,
                    "bottom_depth_m",
                ),
                "center_depth_m": (
                    get_float(manifest_row, "center_depth_m")
                    if manifest_row.get("center_depth_m", "").strip()
                    else 0.5
                    * (
                        get_float(manifest_row, "top_depth_m")
                        + get_float(manifest_row, "bottom_depth_m")
                    )
                ),
                "thickness_z_m": get_float(
                    manifest_row,
                    "thickness_z_m",
                ),
                "width_x_m": get_float(
                    manifest_row,
                    "width_x_m",
                ),
                "width_y_m": get_float(
                    manifest_row,
                    "width_y_m",
                ),
                "body_volume_cells": (
                    body_volume_cells
                ),
                "body_volume_m3": (
                    body_volume_cells
                    * 1_000.0
                ),
                "density_contrast": get_float(
                    manifest_row,
                    "density_contrast",
                ),
                "center_distance_m": (
                    center_distance_m
                ),
                "gravity_maximum_mgal": get_float(
                    manifest_row,
                    "gravity_maximum_mgal",
                ),
                "gravity_std_mgal": get_float(
                    manifest_row,
                    "gravity_std_mgal",
                ),
            }
        )

    return combined_rows


def calculate_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate overall reconstruction statistics.
    """

    metric_names = (
        "mse",
        "iou",
        "dice",
        "predicted_max",
    )

    summary: dict[str, Any] = {
        "number_of_samples": len(
            rows
        )
    }

    for metric_name in metric_names:
        values = np.asarray(
            [
                float(
                    row[
                        metric_name
                    ]
                )
                for row in rows
            ],
            dtype=np.float64,
        )

        summary[
            metric_name
        ] = {
            "mean": float(
                np.mean(
                    values
                )
            ),
            "median": float(
                np.median(
                    values
                )
            ),
            "standard_deviation": float(
                np.std(
                    values
                )
            ),
            "minimum": float(
                np.min(
                    values
                )
            ),
            "maximum": float(
                np.max(
                    values
                )
            ),
            "percentile_25": float(
                np.percentile(
                    values,
                    25.0,
                )
            ),
            "percentile_75": float(
                np.percentile(
                    values,
                    75.0,
                )
            ),
        }

    return summary


def assign_depth_group(
    *,
    top_depth_m: float,
) -> str:
    """
    Assign one controlled top-depth group.
    """

    if top_depth_m <= 60.0:
        return "shallow"

    if top_depth_m <= 110.0:
        return "medium"

    return "deep"


def summarize_depth_groups(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """
    Summarize IoU and Dice by top-depth group.
    """

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = {
        "shallow": [],
        "medium": [],
        "deep": [],
    }

    for row in rows:
        group = assign_depth_group(
            top_depth_m=float(
                row[
                    "top_depth_m"
                ]
            )
        )

        grouped[
            group
        ].append(
            row
        )

    summaries: dict[
        str,
        dict[str, float | int],
    ] = {}

    for group_name, group_rows in grouped.items():
        if not group_rows:
            continue

        iou = np.asarray(
            [
                float(
                    row[
                        "iou"
                    ]
                )
                for row in group_rows
            ],
            dtype=np.float64,
        )

        dice = np.asarray(
            [
                float(
                    row[
                        "dice"
                    ]
                )
                for row in group_rows
            ],
            dtype=np.float64,
        )

        summaries[
            group_name
        ] = {
            "number_of_samples": len(
                group_rows
            ),
            "mean_iou": float(
                np.mean(
                    iou
                )
            ),
            "median_iou": float(
                np.median(
                    iou
                )
            ),
            "mean_dice": float(
                np.mean(
                    dice
                )
            ),
            "median_dice": float(
                np.median(
                    dice
                )
            ),
        }

    return summaries


def calculate_correlations(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    Calculate Pearson correlations with IoU and Dice.
    """

    explanatory_fields = (
        "top_depth_m",
        "center_depth_m",
        "thickness_z_m",
        "body_volume_cells",
        "density_contrast",
        "center_distance_m",
        "gravity_maximum_mgal",
        "gravity_std_mgal",
    )

    results: dict[
        str,
        dict[str, float],
    ] = {}

    for field in explanatory_fields:
        values = np.asarray(
            [
                float(
                    row[
                        field
                    ]
                )
                for row in rows
            ],
            dtype=np.float64,
        )

        iou = np.asarray(
            [
                float(
                    row[
                        "iou"
                    ]
                )
                for row in rows
            ],
            dtype=np.float64,
        )

        dice = np.asarray(
            [
                float(
                    row[
                        "dice"
                    ]
                )
                for row in rows
            ],
            dtype=np.float64,
        )

        if np.isclose(
            np.std(
                values
            ),
            0.0,
        ):
            iou_correlation = float(
                "nan"
            )
            dice_correlation = float(
                "nan"
            )
        else:
            iou_correlation = float(
                np.corrcoef(
                    values,
                    iou,
                )[0, 1]
            )

            dice_correlation = float(
                np.corrcoef(
                    values,
                    dice,
                )[0, 1]
            )

        results[
            field
        ] = {
            "iou_correlation": (
                iou_correlation
            ),
            "dice_correlation": (
                dice_correlation
            ),
        }

    return results


def write_combined_csv(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """
    Save combined per-sample results.
    """

    if not rows:
        raise ValueError(
            "Cannot save an empty result table."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def add_gravity_metrics_to_combined_rows(
    *,
    combined_rows: list[dict[str, Any]],
    gravity_rows: list[dict[str, Any]],
) -> None:
    """
    Add optional gravity metrics to matching combined analysis rows.

    Parameters
    ----------
    combined_rows
        Existing density and geometry analysis rows.
    gravity_rows
        CNN-only gravity-consistency rows.
    """

    gravity_by_id = {
        find_sample_id(
            row
        ): row
        for row in gravity_rows
    }
    gravity_metric_names = tuple(
        name
        for row in gravity_rows
        for name in row
        if name.startswith(
            "gravity_"
        )
    )
    gravity_metric_names = tuple(
        dict.fromkeys(
            gravity_metric_names
        )
    )

    for combined_row in combined_rows:
        sample_id = find_sample_id(
            combined_row
        )
        gravity_row = gravity_by_id.get(
            sample_id
        )

        if gravity_row is None:
            for name in gravity_metric_names:
                combined_row[
                    name
                ] = float("nan")
            continue

        for name in gravity_metric_names:
            combined_row[
                name
            ] = gravity_row.get(
                name,
                float("nan"),
            )


def main() -> None:
    """
    Analyze prediction results.
    """

    arguments = (
        build_argument_parser().parse_args()
    )

    repository_root = (
        find_repository_root()
    )

    dataset_directory = resolve_path(
        repository_root=repository_root,
        path=arguments.dataset,
    )

    prediction_directory = resolve_path(
        repository_root=repository_root,
        path=arguments.predictions,
    )

    manifest_path = (
        dataset_directory
        / arguments.manifest
    )

    metrics_path = (
        prediction_directory
        / arguments.metrics
    )

    manifest_rows = load_table_rows(
        manifest_path
    )

    metric_rows = load_table_rows(
        metrics_path
    )

    gravity_batch_result = None

    if arguments.evaluate_gravity_consistency:
        from cnn_inversion_3d.gravity_consistency import (
            build_cnn_forward_model_context,
            evaluate_prediction_directory,
        )

        context = build_cnn_forward_model_context(
            dataset_directory
            / "metadata.json"
        )
        gravity_batch_result = (
            evaluate_prediction_directory(
                prediction_directory=(
                    prediction_directory
                ),
                metric_rows=metric_rows,
                context=context,
                selected_receiver_indices=(
                    arguments.gravity_comparison_receivers
                ),
                save_gravity_volumes=(
                    arguments.save_gravity_volumes
                ),
                overwrite=arguments.overwrite,
            )
        )

    combined_rows = combine_rows(
        manifest_rows=manifest_rows,
        metric_rows=metric_rows,
    )

    if gravity_batch_result is not None:
        add_gravity_metrics_to_combined_rows(
            combined_rows=combined_rows,
            gravity_rows=(
                gravity_batch_result.rows
            ),
        )

    if arguments.output is None:
        combined_csv_path = (
            prediction_directory
            / "combined_test_metrics.csv"
        )
    else:
        combined_csv_path = resolve_path(
            repository_root=repository_root,
            path=arguments.output,
        )

    write_combined_csv(
        output_path=combined_csv_path,
        rows=combined_rows,
    )

    report = {
        "overall": calculate_summary(
            combined_rows
        ),
        "depth_groups": (
            summarize_depth_groups(
                combined_rows
            )
        ),
        "correlations": (
            calculate_correlations(
                combined_rows
            )
        ),
    }

    if gravity_batch_result is not None:
        from cnn_inversion_3d.gravity_consistency import (
            summarize_gravity_metric_rows,
        )

        report[
            "gravity_consistency"
        ] = {
            **summarize_gravity_metric_rows(
                gravity_batch_result.rows
            ),
            "total_evaluated_samples": len(
                gravity_batch_result.rows
            ),
            "computed_this_run": (
                gravity_batch_result.completed
            ),
            "reused_existing_results": (
                gravity_batch_result.skipped
            ),
            "failed": (
                gravity_batch_result.failed
            ),
            "residual_convention": (
                "recovered_gravity - true_gravity"
            ),
        }

    report_path = (
        prediction_directory
        / "test_analysis.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            indent=2,
        )

    print()
    print("Held-out prediction analysis")
    print("=" * 28)

    overall = report[
        "overall"
    ]

    print(
        f"Samples: "
        f"{overall['number_of_samples']}"
    )

    print(
        "Mean IoU: "
        f"{overall['iou']['mean']:.4f}"
    )

    print(
        "Median IoU: "
        f"{overall['iou']['median']:.4f}"
    )

    print(
        "Mean Dice: "
        f"{overall['dice']['mean']:.4f}"
    )

    print(
        "Median Dice: "
        f"{overall['dice']['median']:.4f}"
    )

    print(
        "Mean MSE: "
        f"{overall['mse']['mean']:.6e}"
    )

    print()
    print("Depth groups")
    print("-" * 12)

    for group_name, group_summary in report[
        "depth_groups"
    ].items():
        print(
            f"{group_name}: "
            f"n={group_summary['number_of_samples']}, "
            f"mean Dice="
            f"{group_summary['mean_dice']:.4f}, "
            f"mean IoU="
            f"{group_summary['mean_iou']:.4f}"
        )

    print()
    print(
        f"Combined CSV: {combined_csv_path}"
    )

    if gravity_batch_result is not None:
        print()
        print("Gravity consistency")
        print("-" * 19)
        print(
            "Total evaluated samples: "
            f"{len(gravity_batch_result.rows)}"
        )
        print(
            f"Computed this run: {gravity_batch_result.completed}"
        )
        print(
            "Reused existing results: "
            f"{gravity_batch_result.skipped}"
        )
        print(
            f"Failed: {gravity_batch_result.failed}"
        )
        print(
            "Metrics CSV: "
            f"{prediction_directory / 'gravity_consistency_metrics.csv'}"
        )

    print(
        f"Analysis JSON: {report_path}"
    )


if __name__ == "__main__":
    main()
