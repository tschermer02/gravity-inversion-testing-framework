from __future__ import annotations
from pathlib import Path

from synthetic_models.common.grid import GridSpec
from synthetic_models.common.generators import generate_dipping_body, generate_dipping_body_from_shallow_center, generate_random_dipping_cases
from synthetic_models.common.experiment_runner import run_experiment
from synthetic_models.common.paths import build_experiment_paths
from forward_modeling.forward_model import GravityForwardModel
from cnn_inversion.cnn_predictor import CNNGravityInverter

def define_cases(
    grid: GridSpec,
) -> list:
    """Define deterministic inclined-body cases for experiment three."""

    return [
        generate_dipping_body_from_shallow_center(
            name="dipping_slab_30_deg",
            shallow_center_x=20.0,
            shallow_center_y=grid.ny / 2.0,
            shallow_center_z=4.0,
            strike_length=20.0,
            dip_length=16.0,
            thickness=3.0,
            strike_degrees=0.0,
            dip_degrees=30.0,
            density_contrast=0.5,
            grid=grid,
        ),
        generate_dipping_body_from_shallow_center(
            name="steep_dike_70_deg",
            shallow_center_x=18.0,
            shallow_center_y=grid.ny / 2.0,
            shallow_center_z=3.0,
            strike_length=18.0,
            dip_length=18.0,
            thickness=2.5,
            strike_degrees=0.0,
            dip_degrees=70.0,
            density_contrast=0.5,
            grid=grid,
        ),
        generate_dipping_body(
            name="elongated_prism_45_deg",
            center_x=grid.nx / 2.0,
            center_y=grid.ny / 2.0,
            center_z=grid.nz / 2.0,
            strike_length=24.0,
            dip_length=10.0,
            thickness=4.0,
            strike_degrees=45.0,
            dip_degrees=45.0,
            density_contrast=0.5,
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

    cases = define_cases(grid)
    # cases =  generate_random_dipping_cases(
    #     number_of_cases=2,
    #     grid=grid,
    #     seed=42,
    #     minimum_strike_length=12.0,
    #     maximum_strike_length=25.0,
    #     minimum_dip_length=8.0,
    #     maximum_dip_length=20.0,
    #     minimum_thickness=2.0,
    #     maximum_thickness=5.0,
    #     minimum_dip_degrees=15.0,
    #     maximum_dip_degrees=80.0,
    #     edge_margin_x=2.0,
    #     edge_margin_y=2.0,
    #     edge_margin_z=2.0,
    # )


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
            "dipping_body_metrics.csv"
        ),
        create_cross_case_figures=True,
    )


if __name__ == "__main__":
    main()