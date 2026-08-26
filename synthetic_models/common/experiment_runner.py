from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cnn_inversion.cnn_predictor import CNNGravityInverter
from forward_modeling.forward_model import GravityForwardModel
from synthetic_models.common.bodies import (
    CaseSpec,
    DippingBodySpec,
    MultiBodyCaseSpec,
    RectangularBodySpec,
    SaltDomeSpec,
    BasementReliefSpec
)
from synthetic_models.common.cnn_pipeline import run_cnn_inversion
from synthetic_models.common.forward_pipeline import (
    create_comparison_figures,
    run_forward_experiment,
)
from synthetic_models.common.grid import GridSpec
from synthetic_models.common.model_io import save_metrics_csv
from synthetic_models.common.paths import ExperimentPaths
from synthetic_models.common.validation import validate_case_names


MetricValue = str | float | int
CaseMetrics = dict[str, MetricValue]


def run_experiment(
    *,
    cases: Sequence[CaseSpec],
    grid: GridSpec,
    paths: ExperimentPaths,
    forward_model: GravityForwardModel,
    cnn_inverter: CNNGravityInverter,
    metrics_filename: str,
    create_cross_case_figures: bool = True,
    gravity_plot_cmap: str = "RdBu_r",
    gravity_plot_zero_centered: bool = True,
) -> list[CaseMetrics]:
    """Run the complete forward-model and CNN-inversion workflow.

    For each case, this function:

    1. Builds and validates the true density model.
    2. Calculates and saves the synthetic gravity response.
    3. Applies the pretrained CNN inversion.
    4. Compares the true and recovered density models.
    5. Forward models the recovered density model.
    6. Compares the original and reproduced gravity.
    7. Saves one combined metrics row.
    """
    if not cases:
        raise ValueError(
            "At least one experiment case must be provided."
        )

    if not metrics_filename.strip():
        raise ValueError(
            "metrics_filename must not be empty."
        )

    if Path(metrics_filename).suffix.lower() != ".csv":
        raise ValueError(
            "metrics_filename must end with '.csv'."
        )

    case_list = list(cases)

    _validate_supported_case_types(case_list)
    validate_case_names(case_list)

    all_anomalies: dict[str, np.ndarray] = {}
    all_metrics: list[CaseMetrics] = []

    for case_number, body in enumerate(
        case_list,
        start=1,
    ):
        print(
            f"\nCase {case_number} of {len(case_list)}: "
            f"{body.name}"
        )

        (
            true_model,
            gravity_anomaly,
            forward_metrics,
        ) = run_forward_experiment(
            body=body,
            grid=grid,
            forward_model=forward_model,
            paths=paths,
            gravity_plot_cmap=gravity_plot_cmap,
            gravity_plot_zero_centered=gravity_plot_zero_centered,
        )

        cnn_metrics = run_cnn_inversion(
            body=body,
            true_model=true_model,
            gravity_anomaly=gravity_anomaly,
            grid=grid,
            forward_model=forward_model,
            cnn_inverter=cnn_inverter,
            paths=paths,
            gravity_plot_cmap=gravity_plot_cmap,
            gravity_plot_zero_centered=gravity_plot_zero_centered,
        )

        combined_metrics: CaseMetrics = {
            **forward_metrics,
            **cnn_metrics,
        }

        all_metrics.append(combined_metrics)
        all_anomalies[body.name] = gravity_anomaly

    metrics_path = (
        paths.metrics
        / metrics_filename
    )

    save_metrics_csv(
        case_metrics=all_metrics,
        output_path=metrics_path,
    )

    if (
        create_cross_case_figures
        and len(all_anomalies) > 1
    ):
        create_comparison_figures(
            anomalies=all_anomalies,
            grid=grid,
            figures_directory=paths.figures,
        )

    print("\nExperiment complete.")
    print(f"  Cases completed: {len(all_metrics)}")
    print(f"  Metrics saved: {metrics_path}")

    return all_metrics


def _validate_supported_case_types(
    cases: Sequence[CaseSpec],
) -> None:
    """Validate that every experiment case uses a supported specification.

    Parameters
    ----------
    cases
        Synthetic case specifications to validate.

    Raises
    ------
    TypeError
        If a case uses an unsupported specification type.
    """
    supported_types = (
        RectangularBodySpec,
        MultiBodyCaseSpec,
        DippingBodySpec,
        SaltDomeSpec,
        BasementReliefSpec
    )

    for case in cases:
        if not isinstance(case, supported_types):
            raise TypeError(
                f"Unsupported case specification: "
                f"{type(case).__name__}."
            )
