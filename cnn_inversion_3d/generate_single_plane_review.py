from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from cnn_inversion_3d.single_plane_review import (
    SinglePlaneBody,
    SinglePlaneReviewConfig,
    build_single_plane_density,
    compute_common_gravity_limits,
    controlled_example_change,
    controlled_single_plane_review_examples,
    forward_model_single_plane,
    observation_coordinate_metadata,
    plot_controlled_examples_comparison,
    plot_coordinate_system_geometry,
    plot_left_right_position_comparison,
    plot_model_geometry_summary,
    plot_observation_coordinates_summary,
    plot_single_plane_example,
    save_json,
    single_plane_review_examples,
    validate_single_plane_review_geometry,
    write_geometry_summary,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic single-observation-plane examples "
            "for physical-geometry review."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "synthetic_models/single_plane_review"
        ),
        help=(
            "Review output directory. Relative paths are interpreted "
            "from the repository root."
        ),
    )
    parser.add_argument(
        "--gravity-display-margin-m",
        type=float,
        default=None,
        help=(
            "Plotting-only margin around the density-domain edges. "
            "Does not alter gravity arrays or receiver coordinates."
        ),
    )
    parser.add_argument(
        "--full-gravity-display-extent",
        action="store_true",
        help="Display the full observation extent on gravity figures.",
    )
    parser.add_argument(
        "--examples",
        nargs="+",
        default=None,
        help=(
            "Optional example names. By default the five preserved legacy "
            "examples and eight controlled examples are generated."
        ),
    )
    parser.add_argument(
        "--observation-marker-stride",
        type=int,
        default=None,
        help=(
            "Display every Nth observation point in X and Y on gravity "
            "figures. Default: configuration value 8."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing review output directory.",
    )
    return parser


def find_repository_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[1]


def resolve_output_directory(
    *,
    repository_root: Path,
    output: Path,
) -> Path:
    """Resolve the requested output directory."""

    if not output.is_absolute():
        output = (
            repository_root
            / output
        )

    return output.resolve()


def select_examples(
    requested_names: list[str] | None,
    examples: tuple[SinglePlaneBody, ...],
) -> tuple[SinglePlaneBody, ...]:
    """
    Select deterministic examples by name.

    Parameters
    ----------
    requested_names
        Requested names or ``None`` for every example.
    examples
        Complete deterministic example collection.

    Returns
    -------
    tuple
        Selected examples in requested order.
    """

    if requested_names is None:
        return examples

    examples_by_name = {
        example.name: example
        for example in examples
    }
    unknown_names = [
        name
        for name in requested_names
        if name not in examples_by_name
    ]

    if unknown_names:
        raise ValueError(
            "Unknown single-plane example name: "
            f"{unknown_names[0]!r}. Available names: "
            f"{list(examples_by_name)}"
        )

    return tuple(
        examples_by_name[name]
        for name in requested_names
    )


def prepare_output_directory(
    *,
    output_directory: Path,
    overwrite: bool,
) -> None:
    """Create an empty scenario output directory."""

    if output_directory.exists():
        if not overwrite:
            raise FileExistsError(
                "Single-plane output directory already exists:\n"
                f"{output_directory}\n"
                "Use --overwrite to replace it."
            )

        shutil.rmtree(
            output_directory
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )


def main() -> None:
    """Generate deterministic single-plane review examples."""

    arguments = (
        build_argument_parser().parse_args()
    )
    repository_root = find_repository_root()
    output_directory = (
        resolve_output_directory(
            repository_root=repository_root,
            output=arguments.output,
        )
    )
    config = SinglePlaneReviewConfig()

    if arguments.observation_marker_stride is not None:
        if arguments.observation_marker_stride < 1:
            raise ValueError(
                "--observation-marker-stride must be at least one."
            )

        config = replace(
            config,
            observation_marker_stride=(
                arguments.observation_marker_stride
            ),
        )
    if arguments.full_gravity_display_extent:
        config = replace(
            config,
            gravity_display_margin_m=None,
        )
    elif arguments.gravity_display_margin_m is not None:
        if arguments.gravity_display_margin_m < 0.0:
            raise ValueError(
                "--gravity-display-margin-m cannot be negative."
            )
        config = replace(
            config,
            gravity_display_margin_m=(
                arguments.gravity_display_margin_m
            ),
        )

    legacy_examples = single_plane_review_examples()
    controlled_examples = (
        controlled_single_plane_review_examples(config)
    )
    examples = select_examples(
        arguments.examples,
        legacy_examples + controlled_examples,
    )
    validate_single_plane_review_geometry(
        config,
        examples,
    )
    prepare_output_directory(
        output_directory=output_directory,
        overwrite=arguments.overwrite,
    )

    densities: dict[str, np.ndarray] = {}
    gravity_maps: dict[str, np.ndarray] = {}

    print()
    print("Single-plane physical review")
    print("=" * 28)
    print(
        f"Density shape: {config.density_shape}"
    )
    print(
        "Density physical edges: "
        f"X {config.density_x_edges_m} m, "
        f"Y {config.density_y_edges_m} m, "
        f"Z {config.density_z_edges_m} m"
    )
    print(
        "Observation plane: "
        f"{config.observation_y_m.size} x "
        f"{config.observation_x_m.size} at "
        f"z={config.observation_z_m:g} m"
    )
    print(
        "Observation X/Y ranges: "
        f"{config.observation_x_m[0]:g} to "
        f"{config.observation_x_m[-1]:g} m"
    )
    print(
        f"Examples: {len(examples)}"
    )
    print()

    for index, body in enumerate(
        examples,
        start=1,
    ):
        density = build_single_plane_density(
            config,
            body,
        )
        gravity = forward_model_single_plane(
            density,
            config=config,
        )
        densities[
            body.name
        ] = density
        gravity_maps[
            body.name
        ] = gravity
        print(
            f"[{index}/{len(examples)}] {body.name}: "
            f"top={body.top_depth_m:g} m, "
            f"bottom={body.bottom_depth_m:g} m, "
            f"gravity max={np.max(gravity):.6e} "
            f"{config.gravity_unit}"
        )

    # Keep every density figure on the full configured physical scale, even
    # when a caller selects only examples whose bodies have lower contrast.
    density_maximum = (
        config.maximum_density_contrast_g_cm3
    )
    (
        gravity_minimum,
        gravity_maximum,
    ) = compute_common_gravity_limits(
        [
            gravity_maps[
                body.name
            ]
            for body in examples
        ]
    )

    save_json(
        output_directory
        / "scenario_config.json",
        config.to_metadata(),
    )
    save_json(
        output_directory
        / "examples.json",
        {
            "horizontal_placement": (
                "Centered on the density domain as closely as "
                "cell-edge alignment permits."
            ),
            "examples": [
                {
                    **body.to_metadata(),
                    "review_set": (
                        "controlled"
                        if body.name.startswith("controlled_")
                        else "legacy"
                    ),
                    **(
                        controlled_example_change(body)
                        if body.name.startswith("controlled_")
                        else {}
                    ),
                }
                for body in examples
            ],
        },
    )
    save_json(
        output_directory
        / "review_metadata.json",
        {
            "gravity_terminology": (
                "Vertical gravity anomaly Gz"
            ),
            "gravity_unit": config.gravity_unit,
            "peak_gravity_definition": (
                "maximum absolute value of the saved gravity array: "
                "max(abs(Gz))"
            ),
            "common_gravity_color_limits": {
                "vmin": gravity_minimum,
                "vmax": gravity_maximum,
                "unit": config.gravity_unit,
            },
            "individual_gravity_ranges": {
                body.name: {
                    "minimum": float(
                        np.min(gravity_maps[body.name])
                    ),
                    "maximum": float(
                        np.max(gravity_maps[body.name])
                    ),
                    "peak_amplitude": float(
                        np.max(
                            np.abs(gravity_maps[body.name])
                        )
                    ),
                    "unit": config.gravity_unit,
                }
                for body in examples
            },
            "common_density_color_limits": {
                "vmin": 0.0,
                "vmax": (
                    config.maximum_density_contrast_g_cm3
                ),
                "unit": config.density_unit,
            },
            "observation_marker_stride": (
                config.observation_marker_stride
            ),
            "observation_markers": (
                "Actual configured observation locations subsampled "
                "for display; the continuous gravity map contains all "
                "observation values."
            ),
            "gravity_display_crop": {
                "display_only": True,
                "x_limits_m": list(config.gravity_display_xlim),
                "y_limits_m": list(config.gravity_display_ylim),
                "full_gravity_arrays_and_coordinates_preserved": True,
            },
            "density_plan_view": (
                "Maximum density contrast at each X-Y location over "
                "the full depth dimension."
            ),
        },
    )

    for body in examples:
        example_directory = (
            output_directory
            / body.name
        )
        example_directory.mkdir(
            parents=True,
            exist_ok=False,
        )
        density = densities[
            body.name
        ]
        gravity = gravity_maps[
            body.name
        ]
        np.save(
            example_directory
            / "density.npy",
            density,
        )
        np.save(
            example_directory
            / "gravity_plane.npy",
            gravity,
        )
        save_json(
            example_directory
            / "model_parameters.json",
            {
                **body.to_metadata(),
                **(
                    controlled_example_change(body)
                    if body.name.startswith("controlled_")
                    else {}
                ),
            },
        )
        save_json(
            example_directory
            / "forward_metadata.json",
            {
                "solver": (
                    "FWD3DGravityForwardModel"
                ),
                "gravity_component": (
                    config.gravity_component
                ),
                "gravity_channel": (
                    config.gravity_channel
                ),
                "gravity_unit": (
                    config.gravity_unit
                ),
                "density_unit": (
                    config.density_unit
                ),
                "density_array_order": (
                    "density[z, y, x]"
                ),
                "gravity_array_order": (
                    "gravity[y, x]"
                ),
                "density_shape": list(
                    density.shape
                ),
                "gravity_shape": list(
                    gravity.shape
                ),
                "observation_x_coordinates_m": (
                    config.observation_x_m.tolist()
                ),
                "observation_y_coordinates_m": (
                    config.observation_y_m.tolist()
                ),
                "observation_z_m": (
                    config.observation_z_m
                ),
                "gravity_minimum_mgal": float(
                    np.min(gravity)
                ),
                "gravity_maximum_mgal": float(
                    np.max(gravity)
                ),
                "gravity_peak_amplitude_mgal": float(
                    np.max(np.abs(gravity))
                ),
                "individual_gravity_color_limits_mgal": [
                    float(np.min(gravity)),
                    float(np.max(gravity)),
                ],
                "common_gravity_color_limits_mgal": [
                    gravity_minimum,
                    gravity_maximum,
                ],
                "gravity_display_limits_m": {
                    "x": list(config.gravity_display_xlim),
                    "y": list(config.gravity_display_ylim),
                    "display_only": True,
                },
            },
        )
        plot_single_plane_example(
            density,
            gravity,
            body,
            config,
            example_directory,
            density_maximum=(
                density_maximum
            ),
            gravity_limits=(
                gravity_minimum,
                gravity_maximum,
            ),
            observation_marker_stride=(
                config.observation_marker_stride
            ),
        )

    plot_coordinate_system_geometry(
        config,
        (
            controlled_examples[0]
            if controlled_examples[0] in examples
            else examples[0]
        ),
        output_directory
        / "coordinate_system_geometry.png",
    )
    plot_model_geometry_summary(
        config,
        examples,
        output_directory
        / "model_geometry_summary.png",
    )
    plot_observation_coordinates_summary(
        config,
        output_directory
        / "observation_coordinates_summary.png",
    )
    coordinate_metadata = observation_coordinate_metadata(config)
    save_json(
        output_directory
        / "observation_coordinates.json",
        coordinate_metadata,
    )
    (
        output_directory
        / "observation_coordinates.md"
    ).write_text(
        "# Observation Coordinates\n\n"
        f"- Observation plane z: "
        f"{coordinate_metadata['observation_plane_z_m']:g} m\n"
        f"- X: {coordinate_metadata['x_min_m']:g}, "
        f"{coordinate_metadata['x_min_m']:g} + "
        f"{coordinate_metadata['x_spacing_m']:g}, ..., "
        f"{coordinate_metadata['x_max_m']:g} m\n"
        f"- Y: {coordinate_metadata['y_min_m']:g}, "
        f"{coordinate_metadata['y_min_m']:g} + "
        f"{coordinate_metadata['y_spacing_m']:g}, ..., "
        f"{coordinate_metadata['y_max_m']:g} m\n"
        f"- Grid: {coordinate_metadata['number_of_x_points']} x "
        f"{coordinate_metadata['number_of_y_points']}\n"
        f"- Total points: "
        f"{coordinate_metadata['total_observation_points']}\n"
        f"- Extent: {coordinate_metadata['total_x_extent_m']:g} x "
        f"{coordinate_metadata['total_y_extent_m']:g} m\n",
        encoding="utf-8",
    )
    selected_controlled = tuple(
        body
        for body in controlled_examples
        if body in examples
    )
    controlled_a_to_f = selected_controlled[:6]
    if len(controlled_a_to_f) == 6:
        plot_controlled_examples_comparison(
            config,
            controlled_a_to_f,
            densities,
            gravity_maps,
            (gravity_minimum, gravity_maximum),
            output_directory
            / "controlled_examples_comparison.png",
        )
    position_examples = tuple(
        body
        for body in selected_controlled
        if body.name.endswith(
            ("left_shifted_position", "right_shifted_position")
        )
    )
    if len(position_examples) == 2:
        plot_left_right_position_comparison(
            config,
            position_examples,
            densities,
            gravity_maps,
            (gravity_minimum, gravity_maximum),
            output_directory
            / "left_right_position_comparison.png",
        )
    write_geometry_summary(
        output_directory
        / "geometry_summary.md",
        config,
        examples,
        common_gravity_limits=(
            gravity_minimum,
            gravity_maximum,
        ),
    )

    print()
    print(
        f"Review outputs: {output_directory}"
    )
    print(
        "Geometry summary: "
        f"{output_directory / 'geometry_summary.md'}"
    )


if __name__ == "__main__":
    main()
