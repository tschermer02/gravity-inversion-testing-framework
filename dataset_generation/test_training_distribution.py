from __future__ import annotations

import numpy as np

from dataset_generation.analyze_training_distribution import (
    calculate_distribution_summary,
)


def test_distribution_summary_uses_all_values() -> None:
    """
    Verify basic statistics across multiple gravity volumes.
    """

    gravity_a = np.array(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ],
        dtype=np.float64,
    )

    gravity_b = np.array(
        [
            [
                [-1.0, -2.0],
                [-3.0, -4.0],
            ]
        ],
        dtype=np.float64,
    )

    summary = calculate_distribution_summary(
        [
            gravity_a,
            gravity_b,
        ]
    )

    assert summary.number_of_samples == 2
    assert summary.number_of_values == 8

    assert summary.global_minimum == -4.0
    assert summary.global_maximum == 4.0
    assert summary.global_absolute_maximum == 4.0

    assert np.isclose(
        summary.global_mean,
        0.0,
    )

    assert summary.sample_maximum_minimum == 4.0
    assert summary.sample_maximum_median == 4.0
    assert summary.sample_maximum_maximum == 4.0


def test_distribution_summary_recommended_scales_are_positive() -> None:
    """
    Verify all proposed scale values are finite and positive.
    """

    gravity = np.arange(
        1.0,
        65.0,
        dtype=np.float64,
    ).reshape(
        1,
        8,
        8,
    )

    summary = calculate_distribution_summary(
        [
            gravity,
        ]
    )

    assert np.isfinite(
        summary.recommended_absolute_max_scale
    )
    assert np.isfinite(
        summary.recommended_percentile_99_scale
    )
    assert np.isfinite(
        summary.recommended_standard_deviation_scale
    )

    assert (
        summary.recommended_absolute_max_scale
        > 0.0
    )
    assert (
        summary.recommended_percentile_99_scale
        > 0.0
    )
    assert (
        summary.recommended_standard_deviation_scale
        > 0.0
    )


def test_distribution_summary_rejects_zero_gravity() -> None:
    """
    Verify that an identically zero training set is rejected.
    """

    gravity = np.zeros(
        (
            8,
            64,
            64,
        ),
        dtype=np.float64,
    )

    try:
        calculate_distribution_summary(
            [
                gravity,
            ]
        )
    except ValueError as error:
        assert "identically zero" in str(
            error
        )
    else:
        raise AssertionError(
            "Expected zero gravity to raise ValueError."
        )