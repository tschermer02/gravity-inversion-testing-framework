from __future__ import annotations
from pathlib import Path

from synthetic_models.common.grid import GridSpec
from synthetic_models.common.generators import generate_cylindrical_salt_plug, generate_tapered_salt_dome, generate_bulbous_salt_dome, generate_mushroom_salt_dome, generate_salt_dome
from synthetic_models.common.experiment_runner import run_experiment
from synthetic_models.common.paths import build_experiment_paths
from forward_modeling.forward_model import GravityForwardModel
from cnn_inversion.cnn_predictor import CNNGravityInverter

def define_cases(
    grid: GridSpec,
) -> list:
    """Define deterministic salt-dome cases."""

    center_x = grid.nx / 2.0
    center_y = grid.ny / 2.0

    return [
        generate_cylindrical_salt_plug(
            name="cylindrical_salt_plug_negative",
            center_x=center_x,
            center_y=center_y,
            top_depth=3.0,
            bottom_depth=21.0,
            radius_x=6.0,
            radius_y=6.0,
            density_contrast=-0.5,
            grid=grid,
        ),

        generate_cylindrical_salt_plug(
            name="cylindrical_salt_plug_positive_control",
            center_x=center_x,
            center_y=center_y,
            top_depth=3.0,
            bottom_depth=21.0,
            radius_x=6.0,
            radius_y=6.0,
            density_contrast=0.5,
            grid=grid,
        ),

        generate_tapered_salt_dome(
            name="tapered_salt_dome_negative",
            center_x=center_x,
            center_y=center_y,
            top_depth=2.0,
            bottom_depth=21.0,
            top_radius_x=8.0,
            top_radius_y=8.0,
            taper_fraction=0.55,
            density_contrast=-0.5,
            grid=grid,
        ),

        generate_bulbous_salt_dome(
            name="bulbous_salt_dome_negative",
            center_x=center_x,
            center_y=center_y,
            top_depth=2.0,
            bottom_depth=22.0,
            stem_radius_x=4.5,
            stem_radius_y=4.5,
            bulb_additional_radius_x=5.0,
            bulb_additional_radius_y=5.0,
            bulb_center_depth=9.0,
            bulb_vertical_scale=4.0,
            taper_fraction=0.15,
            density_contrast=-0.5,
            grid=grid,
        ),

        generate_mushroom_salt_dome(
            name="mushroom_salt_dome_negative",
            center_x=center_x,
            center_y=center_y,
            top_depth=2.0,
            bottom_depth=22.0,
            stem_radius_x=3.5,
            stem_radius_y=3.5,
            cap_additional_radius_x=7.0,
            cap_additional_radius_y=7.0,
            cap_center_depth=5.5,
            cap_vertical_scale=2.5,
            taper_fraction=0.10,
            density_contrast=-0.5,
            grid=grid,
        ),

        generate_mushroom_salt_dome(
            name="mushroom_salt_dome_positive_control",
            center_x=center_x,
            center_y=center_y,
            top_depth=2.0,
            bottom_depth=22.0,
            stem_radius_x=3.5,
            stem_radius_y=3.5,
            cap_additional_radius_x=7.0,
            cap_additional_radius_y=7.0,
            cap_center_depth=5.5,
            cap_vertical_scale=2.5,
            taper_fraction=0.10,
            density_contrast=0.5,
            grid=grid,
        ),

        generate_salt_dome(
            name="elliptical_salt_dome_negative",
            center_x=center_x,
            center_y=center_y,
            top_depth=3.0,
            bottom_depth=21.0,
            stem_radius_x=7.0,
            stem_radius_y=4.0,
            bulb_additional_radius_x=4.0,
            bulb_additional_radius_y=2.5,
            bulb_center_depth=8.0,
            bulb_vertical_scale=3.5,
            taper_fraction=0.20,
            density_contrast=-0.5,
            grid=grid,
        ),
    ]


def main() -> None:
    """Configure and run Experiment 4: synthetic salt domes."""
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
        metrics_filename="salt_dome_metrics.csv",
        create_cross_case_figures=True,
    )


if __name__ == "__main__":
    main()
