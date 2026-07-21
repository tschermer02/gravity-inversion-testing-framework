from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from synthetic_models.common.bodies import CaseSpec
from synthetic_models.common.grid import GridSpec


MetricValue = str | float | int
CaseMetrics = dict[str, MetricValue]


def _save_array_exact(
    array: np.ndarray,
    output_path: Path,
    *,
    description: str,
) -> Path:
    """Save a NumPy array and verify that it reloads exactly.

    Parameters
    ----------
    array
        Array to save.
    output_path
        Destination ``.npy`` path.
    description
        Human-readable label used in error messages.

    Returns
    -------
    Path
        Path to the saved array.

    Raises
    ------
    ValueError
        If the saved array does not reload exactly.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        array,
    )

    loaded_array = np.load(
        output_path,
        allow_pickle=False,
    )

    if not np.array_equal(array, loaded_array):
        raise ValueError(
            f"{description}: saved array did not reload exactly."
        )

    return output_path


def _to_json_compatible(
    value: Any,
) -> Any:
    """Convert nested metadata values into JSON-compatible objects.

    Parameters
    ----------
    value
        Value to convert.

    Returns
    -------
    Any
        JSON-compatible representation of the supplied value.

    Notes
    -----
    Dataclass fields currently use standard Python numeric types, but this
    helper also supports NumPy scalar values and arrays. That makes metadata
    saving robust to future body specifications and grid implementations.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): _to_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return [
            _to_json_compatible(item)
            for item in value
        ]

    if isinstance(value, list):
        return [
            _to_json_compatible(item)
            for item in value
        ]

    return value


def save_true_model(
    model: np.ndarray,
    grid: GridSpec,
    body: CaseSpec,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Save a true density model and its defining metadata.

    The body metadata is serialized using ``dataclasses.asdict``. This
    automatically supports rectangular, multi-body, and dipping-body
    specifications as long as they are dataclasses.

    Parameters
    ----------
    model
        True density model in ``(z, y, x)`` order.
    grid
        Grid specification used to generate the model.
    body
        Synthetic case specification used to generate the model.
    output_directory
        Directory in which the model and metadata should be saved.

    Returns
    -------
    tuple[Path, Path]
        Saved model path and metadata path.
    """
    model_path = output_directory / f"{body.name}.npy"
    metadata_path = output_directory / f"{body.name}_metadata.json"

    _save_array_exact(
        array=model,
        output_path=model_path,
        description=f"{body.name}: true model",
    )

    metadata = {
        "model_name": body.name,
        "case_type": type(body).__name__,
        "model_shape": list(model.shape),
        "model_dtype": str(model.dtype),
        "axis_order": ["z", "y", "x"],
        "flattening_order": "C",
        "grid": asdict(grid),
        "body": asdict(body),
    }

    json_compatible_metadata = _to_json_compatible(metadata)

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_compatible_metadata,
            file,
            indent=4,
        )

    return model_path, metadata_path


def save_gravity_response(
    gravity_anomaly: np.ndarray,
    case_name: str,
    output_directory: Path,
) -> Path:
    """Save the original synthetic gravity response.

    Parameters
    ----------
    gravity_anomaly
        Forward-modeled gravity response.
    case_name
        Name of the synthetic case.
    output_directory
        Directory in which the gravity array should be saved.

    Returns
    -------
    Path
        Path to the saved gravity array.
    """
    output_path = (
        output_directory
        / f"{case_name}_gravity.npy"
    )

    return _save_array_exact(
        array=gravity_anomaly,
        output_path=output_path,
        description=f"{case_name}: gravity response",
    )


def save_recovered_model(
    recovered_model: np.ndarray,
    case_name: str,
    output_directory: Path,
) -> Path:
    """Save the CNN-recovered density model.

    Parameters
    ----------
    recovered_model
        CNN-recovered density model.
    case_name
        Name of the synthetic case.
    output_directory
        Directory in which the model should be saved.

    Returns
    -------
    Path
        Path to the saved recovered model.
    """
    output_path = (
        output_directory
        / f"{case_name}_recovered.npy"
    )

    return _save_array_exact(
        array=recovered_model,
        output_path=output_path,
        description=f"{case_name}: recovered model",
    )


def save_recovered_gravity(
    recovered_gravity: np.ndarray,
    case_name: str,
    output_directory: Path,
) -> Path:
    """Save gravity calculated from the CNN-recovered model.

    Parameters
    ----------
    recovered_gravity
        Gravity response calculated from the recovered density model.
    case_name
        Name of the synthetic case.
    output_directory
        Directory in which the gravity array should be saved.

    Returns
    -------
    Path
        Path to the saved recovered-gravity array.
    """
    output_path = (
        output_directory
        / f"{case_name}_recovered_gravity.npy"
    )

    return _save_array_exact(
        array=recovered_gravity,
        output_path=output_path,
        description=f"{case_name}: recovered gravity",
    )


def save_metrics_csv(
    case_metrics: list[CaseMetrics],
    output_path: Path,
) -> Path:
    """Save one metrics row per experiment case.

    Every row must contain exactly the same fields as the first row.

    Parameters
    ----------
    case_metrics
        Metrics dictionaries to save.
    output_path
        Destination CSV path.

    Returns
    -------
    Path
        Path to the saved metrics file.

    Raises
    ------
    ValueError
        If no rows are supplied or the metric fields differ between rows.
    """
    if not case_metrics:
        raise ValueError(
            "No case metrics were provided."
        )

    fieldnames = list(case_metrics[0].keys())
    expected_fields = set(fieldnames)

    for row_index, metrics in enumerate(
        case_metrics,
        start=1,
    ):
        actual_fields = set(metrics.keys())

        if actual_fields != expected_fields:
            missing_fields = expected_fields - actual_fields
            extra_fields = actual_fields - expected_fields

            raise ValueError(
                f"Metrics row {row_index} does not match the first row. "
                f"Missing fields: {sorted(missing_fields)}. "
                f"Extra fields: {sorted(extra_fields)}."
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(case_metrics)

    return output_path
