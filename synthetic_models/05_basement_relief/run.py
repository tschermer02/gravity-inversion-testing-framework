from __future__ import annotations

from pathlib import Path

from cnn_inversion.cnn_predictor import CNNGravityInverter
from forward_modeling.forward_model import GravityForwardModel
from synthetic_models.common.bodies import BasementReliefSpec
from synthetic_models.common.experiment_runner import run_experiment
from synthetic_models.common.generators import (
    generate_basement_uplift,
    generate_flat_basement,
    generate_sinusoidal_basement,
    generate_tilted_basement,
)
from synthetic_models.common.grid import GridSpec
from synthetic_models.common.paths import build_experiment_paths


def define_cases(
    grid: GridSpec,
) -> list[BasementReliefSpec]:
    """Define deterministic basement-relief experiments."""
    center_x = grid.nx / 2.0
    center_y = grid.ny / 2.0

    return [
        generate_flat_basement(
            name="flat_basement_control",
            depth=14.0,
            density_contrast=0.5,
            grid=grid,
        ),

        generate_tilted_basement(
            name="tilted_basement_x",
            base_depth=14.0,
            slope_x=0.12,
            slope_y=0.0,
            density_contrast=0.5,
            grid=grid,
        ),

        generate_basement_uplift(
            name="circular_basement_uplift",
            base_depth=17.0,
            uplift_height=8.0,
            center_x=center_x,
            center_y=center_y,
            scale_x=10.0,
            scale_y=10.0,
            density_contrast=0.5,
            grid=grid,
        ),

        generate_basement_uplift(
            name="elongated_basement_uplift",
            base_depth=17.0,
            uplift_height=8.0,
            center_x=center_x,
            center_y=center_y,
            scale_x=15.0,
            scale_y=7.0,
            density_contrast=0.5,
            grid=grid,
        ),

        generate_sinusoidal_basement(
            name="sinusoidal_basement_relief",
            base_depth=14.0,
            amplitude=4.0,
            wavelength=40.0,
            azimuth_degrees=90.0,
            density_contrast=0.5,
            grid=grid,
        ),
    ]


def main() -> None:
    """Configure and run Experiment 5: basement relief."""
    grid = GridSpec()

    paths = build_experiment_paths(
        experiment_directory=Path(__file__).resolve().parent,
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
        metrics_filename="basement_relief_metrics.csv",
        create_cross_case_figures=True,
    )


if __name__ == "__main__":
    main()