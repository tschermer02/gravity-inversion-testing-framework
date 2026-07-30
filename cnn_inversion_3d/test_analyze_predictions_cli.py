from __future__ import annotations

from cnn_inversion_3d.analyze_predictions import (
    build_argument_parser,
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
