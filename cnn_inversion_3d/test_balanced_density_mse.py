from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from cnn_inversion_3d.model import (
    BalancedDensityMSE,
)


def density_batch(
    values: list[list[float]],
) -> tf.Tensor:
    """
    Build a small five-dimensional density batch.

    Parameters
    ----------
    values
        Per-sample one-dimensional voxel values.

    Returns
    -------
    tensorflow.Tensor
        Tensor with shape ``(batch, 1, 1, voxels, 1)``.
    """

    array = np.asarray(
        values,
        dtype=np.float32,
    )
    return tf.convert_to_tensor(
        array[:, None, None, :, None]
    )


def test_perfect_prediction_is_zero() -> None:
    """Verify that a perfect prediction has zero balanced loss."""

    target = density_batch(
        [
            [0.0, 0.5, 1.0],
            [0.2, 0.0, 0.0],
        ]
    )
    loss = BalancedDensityMSE()(
        target,
        target,
    )

    assert float(loss) == pytest.approx(
        0.0
    )


def test_zero_prediction_has_expected_body_penalty() -> None:
    """Verify the body penalty for an all-zero prediction."""

    target = density_batch(
        [[0.0, 2.0]]
    )
    prediction = tf.zeros_like(
        target
    )
    loss = BalancedDensityMSE(
        body_fraction=0.5
    )(
        target,
        prediction,
    )

    assert float(loss) == pytest.approx(
        2.0
    )


def test_background_error_affects_background_component() -> None:
    """Verify that background errors contribute to the loss."""

    target = density_batch(
        [[0.0, 2.0]]
    )
    prediction = density_batch(
        [[1.0, 2.0]]
    )
    loss = BalancedDensityMSE(
        body_fraction=0.5
    )(
        target,
        prediction,
    )

    assert float(loss) == pytest.approx(
        0.5
    )


def test_body_and_background_weights_are_applied() -> None:
    """Verify the configured component weights."""

    target = density_batch(
        [[0.0, 2.0]]
    )
    prediction = density_batch(
        [[1.0, 0.0]]
    )
    loss = BalancedDensityMSE(
        body_fraction=0.25
    )(
        target,
        prediction,
    )

    assert float(loss) == pytest.approx(
        1.75
    )


def test_batch_size_greater_than_one() -> None:
    """Verify loss evaluation for multiple samples."""

    target = density_batch(
        [
            [0.0, 1.0],
            [0.0, 2.0],
        ]
    )
    prediction = tf.zeros_like(
        target
    )
    loss = BalancedDensityMSE()(
        target,
        prediction,
    )

    assert float(loss) == pytest.approx(
        1.25
    )


def test_different_body_sizes_are_averaged_per_sample() -> None:
    """Verify that large bodies do not receive extra sample weight."""

    target = density_batch(
        [
            [1.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
        ]
    )
    prediction = tf.zeros_like(
        target
    )
    loss = BalancedDensityMSE()(
        target,
        prediction,
    )

    assert float(loss) == pytest.approx(
        1.25
    )


@pytest.mark.parametrize(
    ("target_values", "prediction_values", "expected"),
    [
        (
            [[0.0, 0.0]],
            [[1.0, 1.0]],
            0.5,
        ),
        (
            [[1.0, 1.0]],
            [[0.0, 0.0]],
            0.5,
        ),
    ],
)
def test_empty_masks_are_safe(
    target_values: list[list[float]],
    prediction_values: list[list[float]],
    expected: float,
) -> None:
    """Verify safe behavior when one mask contains no voxels."""

    loss = BalancedDensityMSE()(
        density_batch(
            target_values
        ),
        density_batch(
            prediction_values
        ),
    )

    assert np.isfinite(
        float(loss)
    )
    assert float(loss) == pytest.approx(
        expected
    )


def test_near_zero_prediction_gradient_is_finite_and_nonzero() -> None:
    """Verify recoverable prediction-space gradients near zero."""

    target = density_batch(
        [[0.0, 0.5, 1.0]]
    )
    prediction = tf.Variable(
        tf.fill(
            target.shape,
            1.0e-8,
        )
    )

    with tf.GradientTape() as tape:
        loss = BalancedDensityMSE()(
            target,
            prediction,
        )

    gradient = tape.gradient(
        loss,
        prediction,
    )

    assert gradient is not None
    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                gradient
            )
        )
    )
    assert float(
        tf.reduce_sum(
            tf.abs(gradient)
        )
    ) > 0.0
