from __future__ import annotations

from cnn_inversion_3d.analyze_predictions import (
    build_argument_parser,
    combine_rows,
)


def test_gravity_consistency_is_opt_in() -> None:
    """Verify unchanged density-only analysis defaults."""

    arguments = build_argument_parser().parse_args(
        [
            "--dataset",
            "dataset",
            "--predictions",
            "predictions",
        ]
    )

    assert (
        arguments.evaluate_gravity_consistency
        is False
    )
    assert arguments.save_gravity_volumes is False
    assert arguments.gravity_comparison_receivers is None
    assert arguments.overwrite is False


def test_gravity_consistency_options_parse() -> None:
    """Verify additive CNN gravity-consistency CLI options."""

    arguments = build_argument_parser().parse_args(
        [
            "--dataset",
            "dataset",
            "--predictions",
            "predictions",
            "--evaluate-gravity-consistency",
            "--save-gravity-volumes",
            "--gravity-comparison-receivers",
            "0",
            "3",
            "7",
            "--overwrite",
        ]
    )

    assert (
        arguments.evaluate_gravity_consistency
        is True
    )
    assert arguments.save_gravity_volumes is True
    assert arguments.gravity_comparison_receivers == [
        0,
        3,
        7,
    ]
    assert arguments.overwrite is True


def test_e05_manifest_centers_are_derived_from_cell_slices() -> None:
    """Verify existing E05 manifests need not be regenerated."""

    manifest = {
        "sample_index": "796",
        "sample_id": "sample_000796",
        "relative_path": "samples/sample_000796.npz",
        "x_start": "3",
        "x_end": "18",
        "y_start": "2",
        "y_end": "18",
        "width_x": "15",
        "width_y": "16",
        "thickness_z": "3",
        "top_depth_m": "40",
        "bottom_depth_m": "70",
        "thickness_z_m": "30",
        "width_x_m": "150",
        "width_y_m": "160",
        "density_contrast": "0.5",
        "gravity_maximum_mgal": "0.19",
        "gravity_std_mgal": "0.03",
    }
    metrics = {
        "sample_id": "sample_000796",
        "mse": 0.01,
        "iou": 0.2,
        "dice": 0.3,
        "prediction_maximum": 0.8,
    }

    row = combine_rows(
        manifest_rows=[manifest],
        metric_rows=[metrics],
    )[0]

    assert row["center_depth_m"] == 55.0
    assert row["center_distance_m"] > 0.0
