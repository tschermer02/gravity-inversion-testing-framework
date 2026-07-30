from __future__ import annotations

import pytest
import tensorflow as tf

from cnn_inversion_3d.diagnostics import (
    CollapseDetectionCallback,
    PredictionAboveThreshold,
    PredictionExtremum,
)


def test_prediction_maximum_tracks_epoch_extremum() -> None:
    """Verify that prediction maximum spans multiple batches."""

    metric = PredictionExtremum(
        mode="maximum",
        name="prediction_maximum",
    )
    target = tf.zeros(
        (1, 1),
        dtype=tf.float32,
    )

    metric.update_state(
        target,
        tf.constant(
            [[0.1, 0.3]],
            dtype=tf.float32,
        ),
    )
    metric.update_state(
        target,
        tf.constant(
            [[0.2, 0.25]],
            dtype=tf.float32,
        ),
    )

    assert float(metric.result()) == pytest.approx(
        0.3
    )


def test_prediction_fraction_above_threshold() -> None:
    """Verify aggregation of the near-zero voxel diagnostic."""

    metric = PredictionAboveThreshold(
        threshold=1.0e-4,
    )
    target = tf.zeros(
        (1, 1),
        dtype=tf.float32,
    )

    metric.update_state(
        target,
        tf.constant(
            [[0.0, 2.0e-4]],
            dtype=tf.float32,
        ),
    )
    metric.update_state(
        target,
        tf.constant(
            [[3.0e-4, 1.0e-5]],
            dtype=tf.float32,
        ),
    )

    assert float(metric.result()) == pytest.approx(
        0.5
    )


def test_collapse_requires_consecutive_epochs() -> None:
    """Verify callback patience and diagnostic recording."""

    callback = CollapseDetectionCallback(
        threshold=1.0e-5,
        patience=2,
    )

    callback.on_epoch_end(
        0,
        {
            "val_prediction_maximum": 1.0e-6,
            "val_prediction_mean": 1.0e-8,
        },
    )
    assert callback.collapse_epoch is None

    callback.on_epoch_end(
        1,
        {
            "val_prediction_maximum": 2.0e-6,
            "val_prediction_mean": 2.0e-8,
        },
    )

    result = callback.result()
    assert result["detected"] is True
    assert result["collapse_epoch"] == 2
    assert result["diagnostics"][
        "val_prediction_maximum"
    ] == pytest.approx(
        2.0e-6
    )


def test_noncollapsed_epoch_resets_patience() -> None:
    """Verify that collapse epochs must be consecutive."""

    callback = CollapseDetectionCallback(
        threshold=1.0e-5,
        patience=2,
    )

    callback.on_epoch_end(
        0,
        {
            "val_prediction_maximum": 1.0e-6,
        },
    )
    callback.on_epoch_end(
        1,
        {
            "val_prediction_maximum": 1.0e-3,
        },
    )
    callback.on_epoch_end(
        2,
        {
            "val_prediction_maximum": 1.0e-6,
        },
    )

    assert callback.result()[
        "detected"
    ] is False
