from __future__ import annotations
from pathlib import Path

from synthetic_models.common.grid import GridSpec
from synthetic_models.common.generators import generate_single_body, generate_multi_body_case, generate_random_multi_body_case
from synthetic_models.common.experiment_runner import run_experiment
from synthetic_models.common.paths import build_experiment_paths
from forward_modeling.forward_model import GravityForwardModel
from cnn_inversion.cnn_predictor import CNNGravityInverter
from synthetic_models.common.bodies import MultiBodyCaseSpec

def define_cases(
    grid: GridSpec,
) -> list[MultiBodyCaseSpec]:
        shallow_left = generate_single_body(
            name="shallow_left_body",
            x_start=8,
            y_start=27,
            z_start=2,
            x_width=10,
            y_width=10,
            z_thickness=5,
            density_contrast=0.5,
            grid=grid,
        )

        medium_center = generate_single_body(
            name="medium_center_body",
            x_start=27,
            y_start=27,
            z_start=9,
            x_width=10,
            y_width=10,
            z_thickness=5,
            density_contrast=0.5,
            grid=grid,
        )

        deep_right = generate_single_body(
            name="deep_right_body",
            x_start=46,
            y_start=27,
            z_start=16,
            x_width=10,
            y_width=10,
            z_thickness=5,
            density_contrast=0.5,
            grid=grid,
        )

        multi_body_case = generate_multi_body_case(
            name="three_bodies_different_depths",
            bodies=[
                shallow_left,
                medium_center,
                deep_right,
            ],
            grid=grid,
            allow_overlap=False,
        )
        return [multi_body_case]

def main() -> None:
    """Configure and run the single compact-body experiment."""

    grid = GridSpec()

    paths = build_experiment_paths(
        experiment_directory=(
            Path(__file__).resolve().parent
        ),
    )
    cases = define_cases(grid)

    # cases = [
    #     generate_random_multi_body_case(
    #         name="random_five_body_case",
    #         number_of_bodies=5,
    #         grid=grid,
    #         seed=42,
    #         minimum_x_width=4,
    #         maximum_x_width=12,
    #         minimum_y_width=4,
    #         maximum_y_width=12,
    #         minimum_z_thickness=3,
    #         maximum_z_thickness=7,
    #         edge_margin_x=2,
    #         edge_margin_y=2,
    #         edge_margin_z=1,
    #     ),
    # ]

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
            "multi_body_depth_metrics.csv"
        ),
        create_cross_case_figures=False,
    )


if __name__ == "__main__":
    main()