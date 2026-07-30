from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cnn_inversion_3d.single_plane_review import (
    SinglePlaneBody,
    SinglePlaneReviewConfig,
    build_single_plane_density,
    compute_common_gravity_limits,
    compute_individual_gravity_limits,
    controlled_example_change,
    controlled_single_plane_review_examples,
    format_body_parameter_text,
    format_gravity_range_text,
    gravity_range_statistics,
    forward_model_single_plane,
    gravity_panel_title,
    observation_marker_coordinates,
    observation_coordinate_metadata,
    plot_controlled_examples_comparison,
    plot_coordinate_system_geometry,
    plot_model_geometry_summary,
    plot_left_right_position_comparison,
    plot_observation_coordinates_summary,
    plot_single_plane_example,
    save_json,
    single_plane_review_examples,
    validate_single_plane_review_geometry,
)


def test_single_plane_forward_shape() -> None:
    """Verify that the adapter removes the singleton receiver dimension."""

    config = SinglePlaneReviewConfig()
    body = single_plane_review_examples()[0]
    density = build_single_plane_density(
        config,
        body,
    )
    gravity = forward_model_single_plane(
        density,
        config=config,
    )

    assert gravity.shape == (
        config.observation_y_m.size,
        config.observation_x_m.size,
    )
    assert np.all(
        np.isfinite(gravity)
    )


def test_observation_geometry_is_fixed_for_examples() -> None:
    """Verify a single horizontal plane is shared by every example."""

    config = SinglePlaneReviewConfig()
    examples = single_plane_review_examples()
    coordinates = [
        (
            tuple(
                config.observation_x_m
            ),
            tuple(
                config.observation_y_m
            ),
            config.observation_z_m,
        )
        for _ in examples
    ]

    assert all(
        coordinates[0] == coordinates_i
        for coordinates_i in coordinates[1:]
    )
    assert config.observation_z_m == pytest.approx(
        0.0
    )


def test_review_bodies_are_valid_and_inside_domain() -> None:
    """Verify all deterministic physical geometries."""

    config = SinglePlaneReviewConfig()
    examples = single_plane_review_examples()

    validate_single_plane_review_geometry(
        config,
        examples,
    )

    assert max(
        body.bottom_depth_m
        for body in examples
    ) <= 160.0


@pytest.mark.parametrize(
    "body",
    single_plane_review_examples(),
    ids=lambda body: body.name,
)
def test_density_dimensions_match_requested_meters(
    body: SinglePlaneBody,
) -> None:
    """Verify occupied voxel dimensions against requested meter values."""

    config = SinglePlaneReviewConfig()
    density = build_single_plane_density(
        config,
        body,
    )
    occupied = np.argwhere(
        density > 0.0
    )
    occupied_shape = (
        np.ptp(
            occupied,
            axis=0,
        )
        + 1
    )

    assert occupied_shape[0] * config.dz_m == pytest.approx(
        body.thickness_m
    )
    assert occupied_shape[1] * config.dy_m == pytest.approx(
        body.width_y_m
    )
    assert occupied_shape[2] * config.dx_m == pytest.approx(
        body.width_x_m
    )


def test_five_times_criterion() -> None:
    """Verify the observation span is exactly the required 800 m."""

    config = SinglePlaneReviewConfig()
    examples = single_plane_review_examples()
    maximum_dimension = max(
        max(
            body.width_x_m
            for body in examples
        ),
        max(
            body.width_y_m
            for body in examples
        ),
        max(
            body.bottom_depth_m
            for body in examples
        ),
    )

    assert (
        config.observation_x_m[-1]
        - config.observation_x_m[0]
    ) == pytest.approx(
        5.0 * maximum_dimension
    )


def test_invalid_geometry_has_physical_error() -> None:
    """Verify a clear error for excessive bottom depth."""

    config = SinglePlaneReviewConfig(
        maximum_bottom_depth_m=150.0
    )
    invalid = single_plane_review_examples()[
        -1
    ]

    with pytest.raises(
        ValueError,
        match="bottom depth 160 m exceeds 150 m",
    ):
        validate_single_plane_review_geometry(
            config,
            [invalid],
        )


def test_metadata_contains_meter_coordinates(
    tmp_path: Path,
) -> None:
    """Verify saved configuration and body coordinates use meters."""

    config = SinglePlaneReviewConfig()
    body = single_plane_review_examples()[0]
    config_path = (
        tmp_path
        / "config.json"
    )
    body_path = (
        tmp_path
        / "body.json"
    )
    save_json(
        config_path,
        config.to_metadata(),
    )
    save_json(
        body_path,
        body.to_metadata(),
    )
    saved_config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )
    saved_body = json.loads(
        body_path.read_text(
            encoding="utf-8"
        )
    )

    assert saved_config[
        "observation_x_coordinates_m"
    ][0] == pytest.approx(-85.0)
    assert saved_config[
        "observation_x_coordinates_m"
    ][-1] == pytest.approx(715.0)
    assert saved_body[
        "top_depth_m"
    ] == pytest.approx(20.0)
    assert saved_body[
        "x_bounds_m"
    ] == pytest.approx(
        [295.0, 335.0]
    )


def test_single_plane_plot_files_created(
    tmp_path: Path,
) -> None:
    """Verify every required review figure is written."""

    config = SinglePlaneReviewConfig(
        observation_x_min_m=215.0,
        observation_x_max_m=415.0,
        observation_y_min_m=215.0,
        observation_y_max_m=415.0,
    )
    body = single_plane_review_examples()[0]
    density = build_single_plane_density(
        config,
        body,
    )
    gravity = forward_model_single_plane(
        density,
        config=config,
    )
    plot_single_plane_example(
        density,
        gravity,
        body,
        config,
        tmp_path,
        density_maximum=1.0,
        gravity_limits=(
            float(
                np.min(gravity)
            ),
            float(
                np.max(gravity)
            ),
        ),
        observation_marker_stride=4,
    )

    for filename in (
        "density_plan_view.png",
        "density_xz_section.png",
        "density_yz_section.png",
        "gravity_map.png",
        "gravity_map_individual_scale.png",
        "model_gravity_summary.png",
    ):
        output_path = (
            tmp_path
            / filename
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_common_gravity_limits_include_every_map() -> None:
    """Verify one global range contains every example value."""

    maps = (
        np.asarray(
            [[1.0, 2.0]]
        ),
        np.asarray(
            [[-3.0, 0.5]]
        ),
        np.asarray(
            [[0.0, 8.0]]
        ),
    )

    assert compute_common_gravity_limits(
        maps
    ) == pytest.approx(
        (
            -3.0,
            8.0,
        )
    )


def test_individual_gravity_limits_use_each_array_range() -> None:
    """Verify per-example limits are not inherited from the global range."""

    weak = np.asarray([[0.01, 0.03]])
    strong = np.asarray([[0.2, 0.8]])

    assert compute_individual_gravity_limits(
        weak
    ) == pytest.approx((0.01, 0.03))
    assert compute_individual_gravity_limits(
        strong
    ) == pytest.approx((0.2, 0.8))
    assert compute_common_gravity_limits(
        (weak, strong)
    ) == pytest.approx((0.01, 0.8))


def test_gravity_range_statistics_match_array() -> None:
    """Verify displayed and saved numerical values derive from Gz."""

    gravity = np.asarray(
        [[-0.04, 0.01], [0.03, 0.02]]
    )

    assert gravity_range_statistics(gravity) == pytest.approx(
        {
            "minimum": -0.04,
            "maximum": 0.03,
            "peak_amplitude": 0.04,
        }
    )


@pytest.mark.parametrize("constant", [0.0, 0.125])
def test_constant_gravity_map_plots_without_error(
    tmp_path: Path,
    constant: float,
) -> None:
    """Verify zero and nonzero constant maps get stable linear limits."""

    config = SinglePlaneReviewConfig(
        observation_x_min_m=215.0,
        observation_x_max_m=415.0,
        observation_y_min_m=215.0,
        observation_y_max_m=415.0,
    )
    body = single_plane_review_examples()[0]
    density = build_single_plane_density(config, body)
    gravity = np.full(
        (
            config.observation_y_m.size,
            config.observation_x_m.size,
        ),
        constant,
    )
    output = tmp_path / f"constant_{constant:g}"
    output.mkdir()

    plot_single_plane_example(
        density,
        gravity,
        body,
        config,
        output,
        density_maximum=1.0,
        gravity_limits=(constant, constant),
        observation_marker_stride=4,
    )

    assert (
        output / "gravity_map_individual_scale.png"
    ).stat().st_size > 0
    assert (
        output / "model_gravity_summary.png"
    ).stat().st_size > 0


def test_observation_markers_use_configured_coordinates() -> None:
    """Verify marker subsampling uses actual observation points."""

    config = SinglePlaneReviewConfig()
    marker_x, marker_y = (
        observation_marker_coordinates(
            config,
            stride=8,
        )
    )

    assert set(
        np.unique(marker_x)
    ) == set(
        config.observation_x_m[
            ::8
        ]
    )
    assert set(
        np.unique(marker_y)
    ) == set(
        config.observation_y_m[
            ::8
        ]
    )


def test_parameter_text_matches_body_metadata() -> None:
    """Verify parameter boxes are built from saved metadata fields."""

    body = single_plane_review_examples()[3]
    text = format_body_parameter_text(
        body.to_metadata(),
        density_unit="g/cm3",
    )

    assert "Center X: 320 m" in text
    assert "Center Y: 315 m" in text
    assert "Top depth: 70 m" in text
    assert "Bottom depth: 120 m" in text
    assert "Width X: 50 m" in text
    assert "Width Y: 60 m" in text
    assert "Thickness: 50 m" in text
    assert "Density contrast: 0.8 g/cm3" in text


def test_observation_plane_title_uses_active_coordinate() -> None:
    """Verify precise Gz terminology and configured plane coordinate."""

    title = gravity_panel_title(
        SinglePlaneReviewConfig(
            observation_z_m=0.0
        )
    )

    assert "Common-scale vertical gravity anomaly Gz" in title
    assert "z = 0 m" in title
    assert "positive downward" not in title


def test_peak_annotation_matches_gravity_array() -> None:
    """Verify the compact annotation uses max absolute saved Gz."""

    gravity = np.asarray([[-0.0123, 0.008]])
    assert format_gravity_range_text(
        gravity,
        unit="mGal",
    ) == "Peak Gz: 0.0123 mGal"


def test_display_crop_does_not_change_gravity_array() -> None:
    """Verify plotting limits are spatial display metadata only."""

    full = SinglePlaneReviewConfig(
        gravity_display_margin_m=None
    )
    cropped = SinglePlaneReviewConfig(
        gravity_display_margin_m=0.0
    )
    body = controlled_single_plane_review_examples(full)[0]
    density = build_single_plane_density(full, body)

    full_gravity = forward_model_single_plane(density, config=full)
    cropped_gravity = forward_model_single_plane(density, config=cropped)

    np.testing.assert_array_equal(full_gravity, cropped_gravity)
    assert cropped.gravity_display_xlim == pytest.approx(
        cropped.density_x_edges_m
    )
    assert cropped.gravity_display_ylim == pytest.approx(
        cropped.density_y_edges_m
    )


def test_controlled_examples_change_only_intended_parameters() -> None:
    """Verify B–E are one-variable controls and F is combined."""

    examples = controlled_single_plane_review_examples(
        SinglePlaneReviewConfig()
    )
    baseline = examples[0]
    fields = (
        "top_depth_m",
        "width_x_m",
        "width_y_m",
        "thickness_m",
        "density_contrast_g_cm3",
        "center_x_m",
        "center_y_m",
    )
    expected = {
        1: {"top_depth_m"},
        2: {"width_x_m", "width_y_m"},
        3: {"density_contrast_g_cm3"},
        4: {"thickness_m"},
    }
    for index, changed_fields in expected.items():
        actual = {
            field
            for field in fields
            if getattr(examples[index], field) != getattr(baseline, field)
        }
        assert actual == changed_fields
    assert (
        controlled_example_change(examples[5])[
            "changed_parameter_relative_to_baseline"
        ]
        == "multiple_parameters"
    )


def test_shifted_examples_are_symmetric_inside_domain() -> None:
    """Verify G/H change only position and remain physically valid."""

    config = SinglePlaneReviewConfig()
    examples = controlled_single_plane_review_examples(config)
    baseline, left, right = examples[0], examples[6], examples[7]
    for shifted in (left, right):
        assert shifted.center_y_m == baseline.center_y_m
        assert shifted.top_depth_m == baseline.top_depth_m
        assert shifted.width_x_m == baseline.width_x_m
        assert shifted.width_y_m == baseline.width_y_m
        assert shifted.thickness_m == baseline.thickness_m
        assert (
            shifted.density_contrast_g_cm3
            == baseline.density_contrast_g_cm3
        )
        assert shifted.x_bounds_m[0] >= config.density_x_edges_m[0]
        assert shifted.x_bounds_m[1] <= config.density_x_edges_m[1]
    assert baseline.center_x_m - left.center_x_m == pytest.approx(
        right.center_x_m - baseline.center_x_m
    )


def test_shifted_gravity_peaks_move_laterally() -> None:
    """Verify translated bodies translate the sampled Gz maximum."""

    config = SinglePlaneReviewConfig()
    examples = controlled_single_plane_review_examples(config)
    peak_x = []
    for body in (examples[6], examples[7]):
        gravity = forward_model_single_plane(
            build_single_plane_density(config, body),
            config=config,
        )
        _, x_index = np.unravel_index(
            np.argmax(gravity),
            gravity.shape,
        )
        peak_x.append(config.observation_x_m[x_index])
    assert peak_x[0] < peak_x[1]


def test_observation_coordinate_metadata_matches_config() -> None:
    """Verify coordinate summary data exactly matches configured arrays."""

    config = SinglePlaneReviewConfig()
    metadata = observation_coordinate_metadata(config)
    assert metadata["x_coordinates_m"] == config.observation_x_m.tolist()
    assert metadata["y_coordinates_m"] == config.observation_y_m.tolist()
    assert metadata["number_of_x_points"] == config.observation_x_m.size
    assert metadata["number_of_y_points"] == config.observation_y_m.size
    assert metadata["total_observation_points"] == (
        config.observation_x_m.size * config.observation_y_m.size
    )


def test_presentation_geometry_figures_created(
    tmp_path: Path,
) -> None:
    """Verify both new presentation-ready geometry figures."""

    config = SinglePlaneReviewConfig()
    examples = single_plane_review_examples()
    coordinate_path = (
        tmp_path
        / "coordinate_system_geometry.png"
    )
    summary_path = (
        tmp_path
        / "model_geometry_summary.png"
    )

    plot_coordinate_system_geometry(
        config,
        examples[0],
        coordinate_path,
    )
    plot_model_geometry_summary(
        config,
        examples,
        summary_path,
    )

    assert coordinate_path.exists()
    assert coordinate_path.stat().st_size > 0
    assert summary_path.exists()
    assert summary_path.stat().st_size > 0


def test_controlled_presentation_figures_created(
    tmp_path: Path,
) -> None:
    """Verify controlled, position, and coordinate summaries are saved."""

    config = SinglePlaneReviewConfig()
    bodies = controlled_single_plane_review_examples(config)
    densities = {
        body.name: build_single_plane_density(config, body)
        for body in bodies
    }
    gravity_maps = {
        body.name: np.zeros(
            (
                config.observation_y_m.size,
                config.observation_x_m.size,
            )
        )
        for body in bodies
    }
    controlled_path = tmp_path / "controlled_examples_comparison.png"
    position_path = tmp_path / "left_right_position_comparison.png"
    coordinates_path = tmp_path / "observation_coordinates_summary.png"

    plot_controlled_examples_comparison(
        config,
        bodies[:6],
        densities,
        gravity_maps,
        (0.0, 0.0),
        controlled_path,
    )
    plot_left_right_position_comparison(
        config,
        bodies[6:],
        densities,
        gravity_maps,
        (0.0, 0.0),
        position_path,
    )
    plot_observation_coordinates_summary(config, coordinates_path)

    for path in (controlled_path, position_path, coordinates_path):
        assert path.exists()
        assert path.stat().st_size > 0
