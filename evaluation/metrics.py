from __future__ import annotations

import numpy as np
from scipy.ndimage import label as connected_component_label
from scipy.optimize import linear_sum_assignment

from synthetic_models.common.bodies import CaseSpec, MultiBodyCaseSpec, RectangularBodySpec, DippingBodySpec
from synthetic_models.common.grid import GridSpec

MetricValue = str | float | int
CaseMetrics = dict[str, MetricValue]

def gravity_response_metrics(
    anomaly: np.ndarray,
) -> dict[str, float | int]:
    """
    Calculate summary metrics for one 2-D gravity anomaly.
    """
    anomaly_array = np.asarray(
        anomaly,
        dtype=np.float64,
    )

    if anomaly_array.ndim != 2:
        raise ValueError(
            "Expected a 2-D gravity anomaly, received shape "
            f"{anomaly_array.shape}."
        )

    if anomaly_array.size == 0:
        raise ValueError(
            "Gravity anomaly must not be empty."
        )

    if not np.all(np.isfinite(anomaly_array)):
        raise ValueError(
            "Gravity anomaly contains NaN or infinite values."
        )

    peak_y_index, peak_x_index = np.unravel_index(
        np.argmax(np.abs(anomaly_array)),
        anomaly_array.shape,
    )

    signed_peak = float(
        anomaly_array[
            peak_y_index,
            peak_x_index,
        ]
    )

    peak_magnitude = abs(signed_peak)
    half_peak_magnitude = 0.5 * peak_magnitude

    above_half_magnitude = (
        np.abs(anomaly_array)
        >= half_peak_magnitude
    )

    half_max_y_indices, half_max_x_indices = np.where(
        above_half_magnitude
    )

    if half_max_x_indices.size == 0:
        half_max_width_x = 0
        half_max_width_y = 0
    else:
        half_max_width_x = int(
            half_max_x_indices.max()
            - half_max_x_indices.min()
            + 1
        )

        half_max_width_y = int(
            half_max_y_indices.max()
            - half_max_y_indices.min()
            + 1
        )

    return {
        "minimum": float(np.min(anomaly_array)),
        "maximum": float(np.max(anomaly_array)),
        "mean": float(np.mean(anomaly_array)),
        "sum": float(np.sum(anomaly_array)),
        "peak_signed_value": signed_peak,
        "peak_magnitude": peak_magnitude,
        "peak_x_index": int(peak_x_index),
        "peak_y_index": int(peak_y_index),
        "half_max_width_x_cells": half_max_width_x,
        "half_max_width_y_cells": half_max_width_y,
    }

def density_center_of_mass(
    model: np.ndarray,
) -> tuple[float, float, float]:
    """
    Return the density-magnitude-weighted center in ``(z, y, x)`` order.
    """
    model_array = np.asarray(
        model,
        dtype=np.float64,
    )

    if model_array.ndim != 3:
        raise ValueError(
            "Density model must be three-dimensional."
        )

    if model_array.size == 0:
        raise ValueError(
            "Density model must not be empty."
        )

    if not np.all(np.isfinite(model_array)):
        raise ValueError(
            "Density model contains NaN or infinite values."
        )

    weights = np.abs(model_array)
    total_weight = float(np.sum(weights))

    if total_weight <= 0.0:
        raise ValueError(
            "Cannot calculate density center of mass because the model "
            "contains no nonzero density."
        )

    z_indices, y_indices, x_indices = np.indices(
        weights.shape,
        dtype=np.float64,
    )

    center_z = float(
        np.sum(z_indices * weights)
        / total_weight
    )

    center_y = float(
        np.sum(y_indices * weights)
        / total_weight
    )

    center_x = float(
        np.sum(x_indices * weights)
        / total_weight
    )

    return center_z, center_y, center_x

def compare_density_models(
    true_model: np.ndarray,
    recovered_model: np.ndarray,
    body: CaseSpec,
    grid: GridSpec,
) -> CaseMetrics:
    """Calculate continuous, overlap, volume, and location metrics."""

    expected_shape = (
        grid.nz,
        grid.ny,
        grid.nx,
    )

    if true_model.shape != expected_shape:
        raise ValueError(
            f"{body.name}: expected true model shape "
            f"{expected_shape}, but received {true_model.shape}."
        )

    if recovered_model.shape != expected_shape:
        raise ValueError(
            f"{body.name}: expected recovered model shape "
            f"{expected_shape}, but received "
            f"{recovered_model.shape}."
        )

    if not np.all(np.isfinite(true_model)):
        raise ValueError(
            f"{body.name}: true model contains NaN or infinite values."
        )

    if not np.all(np.isfinite(recovered_model)):
        raise ValueError(
            f"{body.name}: recovered model contains "
            "NaN or infinite values."
        )

    return {
        **calculate_density_error_metrics(
            true_model=true_model,
            recovered_model=recovered_model,
            body=body,
        ),
        **calculate_overlap_metrics(
            true_model=true_model,
            recovered_model=recovered_model,
            body=body,
        ),
        **calculate_center_metrics(
            true_model=true_model,
            recovered_model=recovered_model,
            grid=grid,
        ),
    }

def calculate_gravity_fit_metrics(
    original_gravity: np.ndarray,
    recovered_gravity: np.ndarray,
) -> CaseMetrics:
    """Calculate how well recovered-model gravity matches input gravity."""

    if original_gravity.shape != recovered_gravity.shape:
        raise ValueError(
            "Original and recovered gravity arrays must have matching "
            f"shapes. Received {original_gravity.shape} and "
            f"{recovered_gravity.shape}."
        )

    if not np.all(np.isfinite(original_gravity)):
        raise ValueError(
            "Original gravity contains NaN or infinite values."
        )

    if not np.all(np.isfinite(recovered_gravity)):
        raise ValueError(
            "Recovered gravity contains NaN or infinite values."
        )

    return {
        **calculate_gravity_error_metrics(
            original_gravity=original_gravity,
            recovered_gravity=recovered_gravity,
        ),
        **calculate_gravity_peak_metrics(
            original_gravity=original_gravity,
            recovered_gravity=recovered_gravity,
        ),
    }

def _safe_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return correlation, or NaN when either flattened array is constant."""

    first_flat = np.asarray(first).reshape(-1)
    second_flat = np.asarray(second).reshape(-1)

    if (
        np.isclose(np.std(first_flat), 0.0)
        or np.isclose(np.std(second_flat), 0.0)
    ):
        return float("nan")

    return float(
        np.corrcoef(
            first_flat,
            second_flat,
        )[0, 1]
    )

def calculate_density_error_metrics(
    true_model: np.ndarray,
    recovered_model: np.ndarray,
    body: CaseSpec,
) -> CaseMetrics:
    """Calculate continuous cell-by-cell density error metrics."""

    residual = recovered_model - true_model
    true_norm = float(np.linalg.norm(true_model))

    if true_norm == 0.0:
        raise ValueError(
            f"{body.name}: true model norm is zero, so relative "
            "L2 error cannot be calculated."
        )

    return {
        "model_mae": float(np.mean(np.abs(residual))),
        "model_rmse": float(np.sqrt(np.mean(residual**2))),
        "model_relative_l2": float(
            np.linalg.norm(residual) / true_norm
        ),
        "model_correlation": _safe_correlation(
            true_model,
            recovered_model,
        ),
    }

def calculate_overlap_metrics(
    true_model: np.ndarray,
    recovered_model: np.ndarray,
    body: CaseSpec,
) -> CaseMetrics:
    """
    Calculate threshold-based body-overlap and volume metrics.
    """
    density_contrast = float(
        body.density_contrast
    )

    if density_contrast == 0.0:
        raise ValueError(
            f"{body.name}: density contrast must be nonzero."
        )

    threshold_magnitude = (
        0.5
        * abs(density_contrast)
    )

    true_mask = (
        np.abs(true_model)
        >= threshold_magnitude
    )

    recovered_mask = (
        np.abs(recovered_model)
        >= threshold_magnitude
    )

    intersection_mask = (
        true_mask
        & recovered_mask
    )

    union_mask = (
        true_mask
        | recovered_mask
    )

    intersection_cells = int(
        np.count_nonzero(intersection_mask)
    )

    union_cells = int(
        np.count_nonzero(union_mask)
    )

    true_body_cells = int(
        np.count_nonzero(true_mask)
    )

    recovered_body_cells = int(
        np.count_nonzero(recovered_mask)
    )

    total_mask_cells = (
        true_body_cells
        + recovered_body_cells
    )

    model_iou = (
        float(
            intersection_cells
            / union_cells
        )
        if union_cells > 0
        else 1.0
    )

    model_dice = (
        float(
            2.0
            * intersection_cells
            / total_mask_cells
        )
        if total_mask_cells > 0
        else 1.0
    )

    volume_ratio = (
        float(
            recovered_body_cells
            / true_body_cells
        )
        if true_body_cells > 0
        else float("nan")
    )

    if intersection_cells > 0:
        true_signs = np.sign(
            true_model[intersection_mask]
        )

        recovered_signs = np.sign(
            recovered_model[intersection_mask]
        )

        matching_sign_cells = int(
            np.count_nonzero(
                true_signs == recovered_signs
            )
        )

        polarity_agreement = float(
            matching_sign_cells
            / intersection_cells
        )
    else:
        matching_sign_cells = 0
        polarity_agreement = float("nan")

    expected_sign = int(
        np.sign(density_contrast)
    )

    recovered_correct_polarity_cells = int(
        np.count_nonzero(
            recovered_mask
            & (
                np.sign(recovered_model)
                == expected_sign
            )
        )
    )

    recovered_correct_polarity_fraction = (
        float(
            recovered_correct_polarity_cells
            / recovered_body_cells
        )
        if recovered_body_cells > 0
        else float("nan")
    )

    return {
        "model_threshold": float(
            threshold_magnitude
        ),
        "model_threshold_magnitude": float(
            threshold_magnitude
        ),
        "expected_density_sign": expected_sign,
        "model_intersection_cells": intersection_cells,
        "model_union_cells": union_cells,
        "true_body_cells_at_threshold": true_body_cells,
        "recovered_body_cells_at_threshold": (
            recovered_body_cells
        ),
        "model_iou": model_iou,
        "model_dice": model_dice,
        "recovered_to_true_volume_ratio": volume_ratio,
        "matching_polarity_intersection_cells": (
            matching_sign_cells
        ),
        "intersection_polarity_agreement": (
            polarity_agreement
        ),
        "recovered_correct_polarity_cells": (
            recovered_correct_polarity_cells
        ),
        "recovered_correct_polarity_fraction": (
            recovered_correct_polarity_fraction
        ),
    }

def calculate_center_metrics(
    true_model: np.ndarray,
    recovered_model: np.ndarray,
    grid: GridSpec,
) -> CaseMetrics:
    """Calculate recovered-source center and center-location errors."""
    (
        true_center_z,
        true_center_y,
        true_center_x,
    ) = density_center_of_mass(
        true_model
    )

    (
        recovered_center_z,
        recovered_center_y,
        recovered_center_x,
    ) = density_center_of_mass(recovered_model)

    error_x_cells = recovered_center_x - true_center_x
    error_y_cells = recovered_center_y - true_center_y
    error_z_cells = recovered_center_z - true_center_z

    distance_cells = float(
        np.sqrt(
            error_x_cells**2
            + error_y_cells**2
            + error_z_cells**2
        )
    )

    error_x_physical = error_x_cells * grid.dx
    error_y_physical = error_y_cells * grid.dy
    error_z_physical = error_z_cells * grid.dz

    distance_physical = float(
        np.sqrt(
            error_x_physical**2
            + error_y_physical**2
            + error_z_physical**2
        )
    )

    return {
        "true_center_x_index": float(true_center_x),
        "true_center_y_index": float(true_center_y),
        "true_center_z_index": float(true_center_z),
        "recovered_center_x_index": recovered_center_x,
        "recovered_center_y_index": recovered_center_y,
        "recovered_center_z_index": recovered_center_z,
        "center_error_x_cells": float(error_x_cells),
        "center_error_y_cells": float(error_y_cells),
        "center_error_z_cells": float(error_z_cells),
        "center_distance_cells": distance_cells,
        "center_error_x_physical": float(error_x_physical),
        "center_error_y_physical": float(error_y_physical),
        "center_error_z_physical": float(error_z_physical),
        "center_distance_physical": distance_physical,
    }

def calculate_gravity_error_metrics(
    original_gravity: np.ndarray,
    recovered_gravity: np.ndarray,
) -> CaseMetrics:
    """Calculate gravity-field residual and correlation metrics."""

    residual = recovered_gravity - original_gravity
    original_norm = float(np.linalg.norm(original_gravity))

    if original_norm == 0.0:
        raise ValueError(
            "Original gravity norm is zero, so relative error "
            "cannot be calculated."
        )

    return {
        "cnn_gravity_rmse": float(
            np.sqrt(np.mean(residual**2))
        ),
        "cnn_gravity_relative_l2": float(
            np.linalg.norm(residual) / original_norm
        ),
        "cnn_gravity_correlation": _safe_correlation(
            original_gravity,
            recovered_gravity,
        ),
        "cnn_gravity_max_absolute_error": float(
            np.max(np.abs(residual))
        ),
    }

def calculate_gravity_peak_metrics(
    original_gravity: np.ndarray,
    recovered_gravity: np.ndarray,
) -> CaseMetrics:
    """
    Calculate dominant gravity amplitude and recovered-location metrics.
    """
    original_peak_y, original_peak_x = np.unravel_index(
        np.argmax(np.abs(original_gravity)),
        original_gravity.shape,
    )

    recovered_peak_y, recovered_peak_x = np.unravel_index(
        np.argmax(np.abs(recovered_gravity)),
        recovered_gravity.shape,
    )

    original_signed_peak = float(
        original_gravity[
            original_peak_y,
            original_peak_x,
        ]
    )

    recovered_signed_peak = float(
        recovered_gravity[
            recovered_peak_y,
            recovered_peak_x,
        ]
    )

    original_peak_magnitude = abs(
        original_signed_peak
    )

    recovered_peak_magnitude = abs(
        recovered_signed_peak
    )

    peak_magnitude_ratio = (
        recovered_peak_magnitude
        / original_peak_magnitude
        if original_peak_magnitude != 0.0
        else float("nan")
    )

    signed_peak_ratio = (
        recovered_signed_peak
        / original_signed_peak
        if original_signed_peak != 0.0
        else float("nan")
    )

    peak_polarity_matches = int(
        np.sign(original_signed_peak)
        == np.sign(recovered_signed_peak)
    )

    return {
        # Keep these older keys for compatibility with existing output code.
        "original_gravity_maximum": original_signed_peak,
        "recovered_gravity_maximum": recovered_signed_peak,
        "gravity_peak_ratio": peak_magnitude_ratio,

        # Explicit polarity-safe metrics.
        "original_gravity_peak_signed_value": (
            original_signed_peak
        ),
        "recovered_gravity_peak_signed_value": (
            recovered_signed_peak
        ),
        "original_gravity_peak_magnitude": (
            original_peak_magnitude
        ),
        "recovered_gravity_peak_magnitude": (
            recovered_peak_magnitude
        ),
        "gravity_signed_peak_ratio": signed_peak_ratio,
        "gravity_peak_polarity_matches": (
            peak_polarity_matches
        ),

        "original_gravity_peak_x_index": int(
            original_peak_x
        ),
        "original_gravity_peak_y_index": int(
            original_peak_y
        ),
        "recovered_gravity_peak_x_index": int(
            recovered_peak_x
        ),
        "recovered_gravity_peak_y_index": int(
            recovered_peak_y
        ),
    }

ComponentMetricValue = str | float | int | bool
ComponentMetrics = dict[str, ComponentMetricValue]

def _component_centroid(
    component_mask: np.ndarray,
) -> tuple[float, float, float]:
    """Return a binary component centroid in (z, y, x) index order."""

    occupied_indices = np.argwhere(
        component_mask
    )

    if occupied_indices.size == 0:
        raise ValueError(
            "Cannot calculate the centroid of an empty component."
        )

    centroid = np.mean(
        occupied_indices,
        axis=0,
    )

    return (
        float(centroid[0]),
        float(centroid[1]),
        float(centroid[2]),
    )

def _true_body_properties(
    body: RectangularBodySpec,
) -> dict[str, str | float | int]:
    """Return the exact geometry of one known true body."""

    center_x = 0.5 * (
        body.x_start
        + body.x_end
        - 1
    )

    center_y = 0.5 * (
        body.y_start
        + body.y_end
        - 1
    )

    center_z = 0.5 * (
        body.z_start
        + body.z_end
        - 1
    )

    volume_cells = (
        (body.x_end - body.x_start)
        * (body.y_end - body.y_start)
        * (body.z_end - body.z_start)
    )

    return {
        "body_name": body.name,
        "center_x": float(center_x),
        "center_y": float(center_y),
        "center_z": float(center_z),
        "volume_cells": int(volume_cells),
    }

def identify_recovered_components(
    recovered_model: np.ndarray,
    *,
    threshold: float,
    minimum_component_cells: int = 5,
) -> list[dict[str, float | int]]:
    """
    Identify connected bodies in a thresholded recovered model.

    Six-neighbor connectivity is used, so cells must share a face to
    belong to the same recovered component.
    """

    if threshold <= 0.0:
        raise ValueError(
            "threshold must be greater than zero."
        )

    if minimum_component_cells <= 0:
        raise ValueError(
            "minimum_component_cells must be greater than zero."
        )

    if recovered_model.ndim != 3:
        raise ValueError(
            "recovered_model must be three-dimensional."
        )

    if not np.all(np.isfinite(recovered_model)):
        raise ValueError(
            "recovered_model contains NaN or infinite values."
        )

    recovered_mask = recovered_model >= threshold

    labeled_model = np.empty(
        recovered_mask.shape,
        dtype=np.int32,
    )

    label_result = connected_component_label(
        recovered_mask,
        output=labeled_model,
    )

    if not isinstance(
        label_result,
        (int, np.integer),
    ):
        raise TypeError(
            "Expected scipy.ndimage.label() to return an integer "
            "component count when an output array is provided."
        )

    component_count = int(label_result)

    components: list[dict[str, float | int]] = []

    for component_id in range(
        1,
        component_count + 1,
    ):
        component_mask = (
            labeled_model == component_id
        )

        volume_cells = int(
            np.count_nonzero(component_mask)
        )

        if volume_cells < minimum_component_cells:
            continue

        center_z, center_y, center_x = (
            _component_centroid(component_mask)
        )

        components.append(
            {
                "component_id": len(components) + 1,
                "center_x": center_x,
                "center_y": center_y,
                "center_z": center_z,
                "volume_cells": volume_cells,
            }
        )

    return components

def calculate_component_recovery_metrics(
    recovered_model: np.ndarray,
    case: MultiBodyCaseSpec,
    grid: GridSpec,
    *,
    threshold: float | None = None,
    minimum_component_cells: int = 5,
) -> tuple[CaseMetrics, list[ComponentMetrics]]:
    """
    Match recovered connected components to the known true bodies.
    """

    if threshold is None:
        threshold = float(
            0.5 * case.density_contrast
        )

    true_bodies = [
        _true_body_properties(body)
        for body in case.bodies
    ]

    recovered_components = identify_recovered_components(
        recovered_model=recovered_model,
        threshold=threshold,
        minimum_component_cells=minimum_component_cells,
    )

    true_count = len(true_bodies)
    recovered_count = len(recovered_components)

    component_rows: list[ComponentMetrics] = []

    if recovered_count == 0:
        for true_body in true_bodies:
            component_rows.append(
                {
                    "case_name": case.name,
                    "true_body_name": str(
                        true_body["body_name"]
                    ),
                    "matched": False,
                    "recovered_component_id": -1,
                    "true_center_x_index": float(
                        true_body["center_x"]
                    ),
                    "true_center_y_index": float(
                        true_body["center_y"]
                    ),
                    "true_center_z_index": float(
                        true_body["center_z"]
                    ),
                    "recovered_center_x_index": float("nan"),
                    "recovered_center_y_index": float("nan"),
                    "recovered_center_z_index": float("nan"),
                    "centroid_error_x_cells": float("nan"),
                    "centroid_error_y_cells": float("nan"),
                    "depth_error_cells": float("nan"),
                    "centroid_distance_cells": float("nan"),
                    "centroid_distance_physical": float("nan"),
                    "true_volume_cells": int(
                        true_body["volume_cells"]
                    ),
                    "recovered_volume_cells": 0,
                    "volume_error_cells": -int(
                        true_body["volume_cells"]
                    ),
                    "volume_ratio": 0.0,
                    "relative_volume_error": -1.0,
                }
            )

        return (
            {
                "true_component_count": true_count,
                "recovered_component_count": 0,
                "matched_component_count": 0,
                "missed_true_body_count": true_count,
                "false_positive_component_count": 0,
                "mean_component_centroid_distance": float("nan"),
                "maximum_component_centroid_distance": float("nan"),
                "mean_absolute_component_depth_error": float("nan"),
                "mean_absolute_relative_volume_error": 1.0,
            },
            component_rows,
        )

    distance_matrix = np.zeros(
        (
            true_count,
            recovered_count,
        ),
        dtype=np.float64,
    )

    for true_index, true_body in enumerate(
        true_bodies
    ):
        for recovered_index, recovered_component in enumerate(
            recovered_components
        ):
            error_x = (
                float(recovered_component["center_x"])
                - float(true_body["center_x"])
            ) * grid.dx

            error_y = (
                float(recovered_component["center_y"])
                - float(true_body["center_y"])
            ) * grid.dy

            error_z = (
                float(recovered_component["center_z"])
                - float(true_body["center_z"])
            ) * grid.dz

            distance_matrix[
                true_index,
                recovered_index,
            ] = np.sqrt(
                error_x**2
                + error_y**2
                + error_z**2
            )

    true_matches, recovered_matches = (
        linear_sum_assignment(distance_matrix)
    )

    matched_pairs = dict(
        zip(
            true_matches.tolist(),
            recovered_matches.tolist(),
        )
    )

    centroid_distances: list[float] = []
    absolute_depth_errors: list[float] = []
    absolute_relative_volume_errors: list[float] = []

    for true_index, true_body in enumerate(
        true_bodies
    ):
        recovered_index = matched_pairs.get(
            true_index
        )

        if recovered_index is None:
            component_rows.append(
                {
                    "case_name": case.name,
                    "true_body_name": str(
                        true_body["body_name"]
                    ),
                    "matched": False,
                    "recovered_component_id": -1,
                    "true_center_x_index": float(
                        true_body["center_x"]
                    ),
                    "true_center_y_index": float(
                        true_body["center_y"]
                    ),
                    "true_center_z_index": float(
                        true_body["center_z"]
                    ),
                    "recovered_center_x_index": float("nan"),
                    "recovered_center_y_index": float("nan"),
                    "recovered_center_z_index": float("nan"),
                    "centroid_error_x_cells": float("nan"),
                    "centroid_error_y_cells": float("nan"),
                    "depth_error_cells": float("nan"),
                    "centroid_distance_cells": float("nan"),
                    "centroid_distance_physical": float("nan"),
                    "true_volume_cells": int(
                        true_body["volume_cells"]
                    ),
                    "recovered_volume_cells": 0,
                    "volume_error_cells": -int(
                        true_body["volume_cells"]
                    ),
                    "volume_ratio": 0.0,
                    "relative_volume_error": -1.0,
                }
            )

            continue

        recovered_component = (
            recovered_components[recovered_index]
        )

        error_x_cells = (
            float(recovered_component["center_x"])
            - float(true_body["center_x"])
        )

        error_y_cells = (
            float(recovered_component["center_y"])
            - float(true_body["center_y"])
        )

        depth_error_cells = (
            float(recovered_component["center_z"])
            - float(true_body["center_z"])
        )

        centroid_distance_cells = float(
            np.sqrt(
                error_x_cells**2
                + error_y_cells**2
                + depth_error_cells**2
            )
        )

        centroid_distance_physical = float(
            np.sqrt(
                (error_x_cells * grid.dx) ** 2
                + (error_y_cells * grid.dy) ** 2
                + (depth_error_cells * grid.dz) ** 2
            )
        )

        true_volume = int(
            true_body["volume_cells"]
        )

        recovered_volume = int(
            recovered_component["volume_cells"]
        )

        volume_error = (
            recovered_volume
            - true_volume
        )

        volume_ratio = (
            recovered_volume
            / true_volume
        )

        relative_volume_error = (
            volume_error
            / true_volume
        )

        centroid_distances.append(
            centroid_distance_physical
        )

        absolute_depth_errors.append(
            abs(depth_error_cells * grid.dz)
        )

        absolute_relative_volume_errors.append(
            abs(relative_volume_error)
        )

        component_rows.append(
            {
                "case_name": case.name,
                "true_body_name": str(
                    true_body["body_name"]
                ),
                "matched": True,
                "recovered_component_id": int(
                    recovered_component["component_id"]
                ),
                "true_center_x_index": float(
                    true_body["center_x"]
                ),
                "true_center_y_index": float(
                    true_body["center_y"]
                ),
                "true_center_z_index": float(
                    true_body["center_z"]
                ),
                "recovered_center_x_index": float(
                    recovered_component["center_x"]
                ),
                "recovered_center_y_index": float(
                    recovered_component["center_y"]
                ),
                "recovered_center_z_index": float(
                    recovered_component["center_z"]
                ),
                "centroid_error_x_cells": error_x_cells,
                "centroid_error_y_cells": error_y_cells,
                "depth_error_cells": depth_error_cells,
                "centroid_distance_cells": centroid_distance_cells,
                "centroid_distance_physical": (
                    centroid_distance_physical
                ),
                "true_volume_cells": true_volume,
                "recovered_volume_cells": recovered_volume,
                "volume_error_cells": volume_error,
                "volume_ratio": float(volume_ratio),
                "relative_volume_error": float(
                    relative_volume_error
                ),
            }
        )

    matched_count = len(matched_pairs)

    summary_metrics: CaseMetrics = {
        "true_component_count": true_count,
        "recovered_component_count": recovered_count,
        "matched_component_count": matched_count,
        "missed_true_body_count": (
            true_count - matched_count
        ),
        "false_positive_component_count": max(
            recovered_count - matched_count,
            0,
        ),
        "mean_component_centroid_distance": (
            float(np.mean(centroid_distances))
            if centroid_distances
            else float("nan")
        ),
        "maximum_component_centroid_distance": (
            float(np.max(centroid_distances))
            if centroid_distances
            else float("nan")
        ),
        "mean_absolute_component_depth_error": (
            float(np.mean(absolute_depth_errors))
            if absolute_depth_errors
            else float("nan")
        ),
        "mean_absolute_relative_volume_error": (
            float(
                np.mean(
                    absolute_relative_volume_errors
                )
            )
            if absolute_relative_volume_errors
            else float("nan")
        ),
    }

    return summary_metrics, component_rows

def _density_body_mask(
    model: np.ndarray,
    density_contrast: float,
    *,
    threshold_fraction: float = 0.5,
) -> np.ndarray:
    """Create a threshold mask respecting density-contrast polarity."""
    if not 0.0 < threshold_fraction <= 1.0:
        raise ValueError(
            "threshold_fraction must be greater than zero and no more than one."
        )

    if density_contrast == 0.0:
        raise ValueError(
            "density_contrast must not be zero."
        )

    threshold = threshold_fraction * density_contrast

    if density_contrast > 0.0:
        return model >= threshold

    return model <= threshold