from __future__ import annotations

import numpy as np

from cnn_inversion.cnn_predictor import CNNGravityInverter
from evaluation.metrics import (
    calculate_component_recovery_metrics,
    calculate_gravity_fit_metrics,
    compare_density_models,
)
from evaluation.plotting import (
    plot_gravity_fit_comparison,
    plot_true_recovered_density_comparison,
    plot_noise_inversion_comparison,
)
from forward_modeling.forward_model import GravityForwardModel
from synthetic_models.common.bodies import (
    CaseSpec,
    MultiBodyCaseSpec,
)
from synthetic_models.common.grid import GridSpec
from synthetic_models.common.model_io import (
    save_metrics_csv,
    save_recovered_gravity,
    save_recovered_model,
)
from synthetic_models.common.output import (
    print_gravity_fit_summary,
    print_recovered_model_summary,
    print_saved_paths,
)
from synthetic_models.common.paths import ExperimentPaths
from synthetic_models.common.validation import validate_density_model

MetricValue = str | float | int
CaseMetrics = dict[str, MetricValue]

def run_cnn_inversion(
    body: CaseSpec,
    true_model: np.ndarray,
    gravity_anomaly: np.ndarray,
    grid: GridSpec,
    forward_model: GravityForwardModel,
    cnn_inverter: CNNGravityInverter,
    paths: ExperimentPaths,
    clean_reference_gravity: np.ndarray | None = None,
    noise_percent: float | None = None,
    gravity_plot_cmap: str = "RdBu_r",
    gravity_plot_zero_centered: bool = True,
) -> CaseMetrics:
    """
    Run CNN inversion and evaluate the recovered density model.

    Parameters
    ----------
    body
        Synthetic case specification. Its name is used for saved outputs.

    true_model
        Original synthetic density model in ``(z, y, x)`` order.

    gravity_anomaly
        Gravity anomaly supplied directly to the pretrained CNN. For a
        noise experiment, this is the noisy gravity anomaly.

    grid
        Model-grid specification.

    forward_model
        Gravity forward-modeling interface.

    cnn_inverter
        Pretrained CNN inversion interface.

    paths
        Experiment output paths.

    clean_reference_gravity
        Optional clean gravity anomaly associated with the true model.

        When supplied, the recovered-model gravity is compared against
        both:

        1. ``gravity_anomaly`` — the data supplied to the CNN.
        2. ``clean_reference_gravity`` — the underlying clean response.

        This is particularly useful for noise experiments.

    Returns
    -------
    dict
        Recovered-model metrics, gravity fit to the CNN input, and,
        when requested, gravity fit to the clean reference.
    """

    print("\nRunning CNN inversion...")

    recovered_model = _predict_recovered_model(
        gravity_anomaly=gravity_anomaly,
        cnn_inverter=cnn_inverter,
    )

    validate_density_model(
        model=recovered_model,
        body=body,
        grid=grid,
        model_label="recovered",
    )

    model_metrics = compare_density_models(
        true_model=true_model,
        recovered_model=recovered_model,
        body=body,
        grid=grid,
    )

    component_summary = _calculate_optional_component_metrics(
        body=body,
        recovered_model=recovered_model,
        grid=grid,
        paths=paths,
    )

    model_peak_z, model_peak_y, model_peak_x = np.unravel_index(
        np.argmax(np.abs(recovered_model)),
        recovered_model.shape,
    )

    print_recovered_model_summary(
        recovered_model=recovered_model,
    )

    recovered_model_path = save_recovered_model(
        recovered_model=recovered_model,
        case_name=body.name,
        output_directory=paths.cnn_outputs,
    )

    comparison_figure_path = (
        paths.figures
        / f"{body.name}_true_recovered_comparison.png"
    )

    plot_true_recovered_density_comparison(
        true_model=true_model,
        recovered_model=recovered_model,
        body=body,
        grid=grid,
        case_name=body.name,
        output_path=comparison_figure_path,
    )

    recovered_gravity = forward_model.calculate(
        model=recovered_model,
    )

    recovered_gravity = np.ascontiguousarray(
        recovered_gravity,
        dtype=np.float32,
    )

    noise_workflow_figure_path = None

    if clean_reference_gravity is not None:
        if noise_percent is None:
            raise ValueError(
                "noise_percent must be provided when "
                "clean_reference_gravity is supplied."
            )

        noise_workflow_figure_path = (
            paths.figures
            / f"{body.name}_noise_inversion_workflow.png"
        )

        plot_noise_inversion_comparison(
            clean_gravity=clean_reference_gravity,
            noisy_gravity=gravity_anomaly,
            recovered_gravity=recovered_gravity,
            recovered_model=recovered_model,
            grid=grid,
            case_name=body.name,
            noise_percent=noise_percent,
            output_path=noise_workflow_figure_path,
        )

    # Fit against the gravity data actually supplied to the CNN.
    input_gravity_fit = calculate_gravity_fit_metrics(
        original_gravity=gravity_anomaly,
        recovered_gravity=recovered_gravity,
    )

    print("\nRecovered-model fit to CNN input gravity")
    print_gravity_fit_summary(
        gravity_metrics=input_gravity_fit,
    )

    input_gravity_metrics = {
        f"input_{key}": value
        for key, value in input_gravity_fit.items()
    }

    recovered_gravity_path = save_recovered_gravity(
        recovered_gravity=recovered_gravity,
        case_name=body.name,
        output_directory=paths.forward_responses,
    )

    input_fit_figure_path = (
        paths.figures
        / f"{body.name}_input_gravity_fit_comparison.png"
    )

    plot_gravity_fit_comparison(
        original_gravity=gravity_anomaly,
        recovered_gravity=recovered_gravity,
        grid=grid,
        case_name=f"{body.name}: fit to CNN input",
        output_path=input_fit_figure_path,
        gravity_cmap=gravity_plot_cmap,
        gravity_zero_centered=gravity_plot_zero_centered,
    )

    clean_gravity_metrics: CaseMetrics = {}
    clean_fit_figure_path = None

    if clean_reference_gravity is not None:
        expected_shape = (
            grid.ny,
            grid.nx,
        )

        if clean_reference_gravity.shape != expected_shape:
            raise ValueError(
                f"{body.name}: expected clean reference gravity shape "
                f"{expected_shape}, but received "
                f"{clean_reference_gravity.shape}."
            )

        if not np.all(np.isfinite(clean_reference_gravity)):
            raise ValueError(
                f"{body.name}: clean reference gravity contains "
                "NaN or infinite values."
            )

        clean_gravity_fit = calculate_gravity_fit_metrics(
            original_gravity=clean_reference_gravity,
            recovered_gravity=recovered_gravity,
        )

        print("\nRecovered-model fit to clean gravity")
        print_gravity_fit_summary(
            gravity_metrics=clean_gravity_fit,
        )

        clean_gravity_metrics = {
            f"clean_{key}": value
            for key, value in clean_gravity_fit.items()
        }

        clean_fit_figure_path = (
            paths.figures
            / f"{body.name}_clean_gravity_fit_comparison.png"
        )

        plot_gravity_fit_comparison(
            original_gravity=clean_reference_gravity,
            recovered_gravity=recovered_gravity,
            grid=grid,
            case_name=f"{body.name}: fit to clean gravity",
            output_path=clean_fit_figure_path,
            gravity_cmap=gravity_plot_cmap,
            gravity_zero_centered=gravity_plot_zero_centered,
        )

    saved_paths = {
        "Recovered model": recovered_model_path,
        "True/recovered comparison": comparison_figure_path,
        "Recovered gravity": recovered_gravity_path,
        "Input-gravity fit comparison": input_fit_figure_path,
    }

    if clean_fit_figure_path is not None:
        saved_paths[
            "Clean-gravity fit comparison"
        ] = clean_fit_figure_path

    if noise_workflow_figure_path is not None:
        saved_paths[
            "Noise inversion workflow"
        ] = noise_workflow_figure_path

    print_saved_paths(
        paths=saved_paths,
    )

    return {
        "recovered_model_peak_x_index": int(model_peak_x),
        "recovered_model_peak_y_index": int(model_peak_y),
        "recovered_model_peak_z_index": int(model_peak_z),
        "recovered_model_minimum": float(
            recovered_model.min()
        ),
        "recovered_model_maximum": float(
            recovered_model.max()
        ),
        "recovered_model_mean": float(
            recovered_model.mean()
        ),
        **model_metrics,
        **component_summary,
        **input_gravity_metrics,
        **clean_gravity_metrics,
    }

def _predict_recovered_model(
    *,
    gravity_anomaly: np.ndarray,
    cnn_inverter: CNNGravityInverter,
) -> np.ndarray:
    """Apply the CNN and normalize its output array representation.
    """
    prediction = cnn_inverter.predict(
        gravity_anomaly=gravity_anomaly,
    )

    return np.ascontiguousarray(
        np.asarray(
            prediction,
            dtype=np.float32,
        )
    )

def _calculate_optional_component_metrics(
    *,
    body: CaseSpec,
    recovered_model: np.ndarray,
    grid: GridSpec,
    paths: ExperimentPaths,
) -> CaseMetrics:
    """Calculate and save component metrics for multi-body cases.
    """
    if not isinstance(body, MultiBodyCaseSpec):
        return {}

    (
        component_summary,
        component_rows,
    ) = calculate_component_recovery_metrics(
        recovered_model=recovered_model,
        case=body,
        grid=grid,
        threshold=0.5 * body.density_contrast,
        minimum_component_cells=5,
    )

    if not component_rows:
        return component_summary

    component_metrics_path = (
        paths.metrics
        / f"{body.name}_component_metrics.csv"
    )

    save_metrics_csv(
        case_metrics=component_rows,
        output_path=component_metrics_path,
    )

    print_saved_paths(
        paths={
            "Component metrics": component_metrics_path,
        },
    )

    return component_summary
