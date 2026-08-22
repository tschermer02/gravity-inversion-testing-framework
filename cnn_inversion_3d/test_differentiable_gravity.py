"""Numerical and gradient tests for the E07 training-only forward model."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from cnn_inversion_3d.dataset import DENSITY_SHAPE, SINGLE_PLANE_GRAVITY_SHAPE
from cnn_inversion_3d.differentiable_gravity import (
    DifferentiableSinglePlaneGz,
    PhysicsConsistencyTrainingModel,
    global_normalized_gravity_mse,
    excess_occupied_volume_fraction_mse,
    combine_density_gravity_volume_losses,
    soft_occupied_fraction,
)
from cnn_inversion_3d.model import (
    ModelConfig,
    build_single_plane_learned_depth_seed_model,
)
from cnn_inversion_3d.single_plane_review import (
    SinglePlaneReviewConfig,
    forward_model_single_plane,
)


def test_differentiable_forward_shape_and_source_agreement() -> None:
    """Verify shallow, deep, and edge cells match validated FWD3D."""

    config = SinglePlaneReviewConfig()
    operator = DifferentiableSinglePlaneGz(
        config, calculation_dtype=tf.float64
    )
    for index in ((0, 32, 32), (23, 32, 32), (0, 1, 1), (23, 62, 60)):
        density = np.zeros(config.density_shape, dtype=np.float64)
        density[index] = 0.7
        expected = forward_model_single_plane(density, config=config)
        actual = operator(
            tf.constant(density[None, ..., None], dtype=tf.float64)
        ).numpy()[0, ..., 0]
        relative_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        assert actual.shape == (81, 81)
        assert relative_l2 <= 1.0e-5


def test_differentiable_forward_gradient_exists_and_is_finite() -> None:
    """Verify automatic differentiation reaches every density voxel."""

    operator = DifferentiableSinglePlaneGz(calculation_dtype=tf.float64)
    density = tf.Variable(np.ones((1, *DENSITY_SHAPE), dtype=np.float64))
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(tf.square(operator(density)))
    gradient = tape.gradient(loss, density)
    assert gradient is not None
    assert gradient.shape == density.shape
    assert np.all(np.isfinite(gradient.numpy()))


def test_global_normalized_gravity_mse_is_numerically_correct() -> None:
    """Verify global scaling without a per-sample energy denominator."""

    true = tf.constant([[[[1.0]], [[2.0]]]], dtype=tf.float32)
    predicted = tf.constant([[[[2.0]], [[4.0]]]], dtype=tf.float32)
    loss = global_normalized_gravity_mse(
        true, predicted, gravity_scale=2.0
    )
    assert float(loss.numpy()) == 0.625

    weak = global_normalized_gravity_mse(
        true * 0.01, predicted * 0.01, gravity_scale=2.0
    )
    assert float(weak.numpy()) < float(loss.numpy())


def test_e07_total_loss_identity_when_volume_weight_is_zero() -> None:
    """Verify a zero volume weight reproduces the exact E07 objective."""

    inversion = build_single_plane_learned_depth_seed_model(
        ModelConfig(base_filters=1)
    )
    wrapper = PhysicsConsistencyTrainingModel(
        inversion,
        DifferentiableSinglePlaneGz(),
        gravity_scale=0.22938017547130585,
        gravity_loss_weight=0.001,
        volume_loss_weight=0.0,
    )
    gravity = np.ones((1, *SINGLE_PLANE_GRAVITY_SHAPE), np.float32) * 0.1
    density = np.zeros((1, *DENSITY_SHAPE), np.float32)
    density[:, 2:4, 30:34, 30:34, :] = 0.5
    terms = wrapper.compute_loss_terms(gravity, density, training=False)
    np.testing.assert_allclose(
        terms[4].numpy(),
        terms[1].numpy() + 0.001 * terms[2].numpy(),
        rtol=1.0e-6,
    )
    assert wrapper.inversion_model is inversion


def _volume(values: list[float]) -> tf.Tensor:
    return tf.reshape(tf.constant(values, tf.float32), (1, 1, 1, 4, 1))


def test_soft_occupancy_is_zero_at_zero() -> None:
    """Verify zero-floor normalization."""

    value = soft_occupied_fraction(tf.zeros((1, 1, 1, 1, 1))).numpy()[0]
    assert value == 0.0


def test_soft_occupancy_is_monotonic() -> None:
    """Verify soft occupancy increases with density."""

    fractions = soft_occupied_fraction(
        tf.reshape(tf.constant([0.0, 0.05, 0.1, 0.15]), (4, 1, 1, 1, 1))
    ).numpy()
    assert np.all(np.diff(fractions) > 0.0)


def test_excess_support_produces_positive_loss() -> None:
    """Verify excessive support activates the corrected penalty."""

    truth = _volume([1.0, 1.0, 0.0, 0.0])
    excess = _volume([1.0, 1.0, 0.2, 0.0])
    assert float(excess_occupied_volume_fraction_mse(truth, excess)[0]) > 0.0


def test_deficient_support_produces_zero_excess_loss() -> None:
    """Verify deficient support is left to BalancedDensityMSE."""

    truth = _volume([1.0, 1.0, 0.0, 0.0])
    deficit = _volume([1.0, 0.0, 0.0, 0.0])
    assert float(excess_occupied_volume_fraction_mse(truth, deficit)[0]) == 0.0


def test_matching_support_produces_zero_excess_loss() -> None:
    """Verify matching occupied fractions have no excess penalty."""

    truth = _volume([1.0, 1.0, 0.0, 0.0])
    matching = _volume([1.0, 1.0, 0.0, 0.0])
    assert float(excess_occupied_volume_fraction_mse(truth, matching)[0]) < 1.0e-12


def test_soft_occupied_fraction_gradient_reduces_excess_volume() -> None:
    """Verify gradient descent pushes excessive soft occupancy downward."""

    truth = _volume([1.0, 1.0, 0.0, 0.0])
    prediction = tf.Variable(_volume([1.0, 1.0, 0.2, 0.0]))
    with tf.GradientTape() as tape:
        loss = excess_occupied_volume_fraction_mse(truth, prediction)[0]
    gradient = tape.gradient(loss, prediction)
    assert gradient is not None
    assert np.all(np.isfinite(gradient.numpy()))
    assert float(gradient.numpy()[0, 0, 0, 2, 0]) > 0.0


def test_soft_occupancy_default_has_nonzero_threshold_gradient() -> None:
    """Verify the default sharpness retains a useful local derivative."""

    density = tf.Variable([[[[[0.1]]]]], dtype=tf.float32)
    with tf.GradientTape() as tape:
        fraction = soft_occupied_fraction(density)
    gradient = tape.gradient(fraction, density)
    assert np.all(np.isfinite(gradient.numpy()))
    assert float(gradient.numpy().squeeze()) > 10.0


def test_zero_volume_weight_exactly_recovers_e07_loss() -> None:
    """Verify the corrected integration remains backward compatible with E07."""

    density = tf.constant(0.2)
    gravity = tf.constant(3.0)
    arbitrary_volume = tf.constant(99.0)
    total = combine_density_gravity_volume_losses(
        density,
        gravity,
        arbitrary_volume,
        gravity_loss_weight=0.001,
        volume_loss_weight=0.0,
    )
    np.testing.assert_allclose(total.numpy(), (density + 0.001 * gravity).numpy())
