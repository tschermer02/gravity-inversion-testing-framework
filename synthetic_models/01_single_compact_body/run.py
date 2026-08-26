from __future__ import annotations
from pathlib import Path

from synthetic_models.common.grid import GridSpec
from synthetic_models.common.generators import generate_single_body
from synthetic_models.common.experiment_runner import run_experiment
from synthetic_models.common.paths import build_experiment_paths
from forward_modeling.forward_model import GravityForwardModel
from cnn_inversion.cnn_predictor import CNNGravityInverter

def define_cases(
    grid: GridSpec,
) -> list:
    """Define the cases included in this experiment run."""

    return [
        generate_single_body(
            name="sample_000044",
            x_start=24,
            y_start=19,
            z_start=7,
            x_width=7,
            y_width=14,
            z_thickness=7,
            density_contrast=0.8129409408729518,
            grid=grid,
        ),
    ]

def main() -> None:
    """Configure and run the single compact-body experiment."""

    grid = GridSpec()

    paths = build_experiment_paths(
        experiment_directory=(
            Path(__file__).resolve().parent
        ),
    )

    cases = define_cases(
        grid=grid,
    )

    forward_model = GravityForwardModel(
        grid=grid,
        cppforward_path=paths.cppforward,
    )

    cnn_inverter = CNNGravityInverter(
        grid=grid,
        inv_model_path=paths.inv_model,
        weights_path=paths.cnn_weights,
    )

    run_experiment(
        cases=cases,
        grid=grid,
        paths=paths,
        forward_model=forward_model,
        cnn_inverter=cnn_inverter,
        metrics_filename=(
            "single_compact_body_metrics.csv"
        ),
        create_cross_case_figures=True,
        gravity_plot_cmap="viridis",
        gravity_plot_zero_centered=False,
    )


if __name__ == "__main__":
    main()
