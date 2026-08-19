"""Numerical and gradient tests for the E07 training-only forward model."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from cnn_inversion_3d.dataset import DENSITY_SHAPE, SINGLE_PLANE_GRAVITY_SHAPE
from cnn_inversion_3d.differentiable_gravity import (
    DifferentiableSinglePlaneGz,
    PhysicsConsistencyTrainingModel,
    global_normalized_gravity_mse,
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


def test_e07_total_loss_identity_and_tiny_step() -> None:
    """Verify E07 changes the objective, not the E06 inference network."""

    inversion = build_single_plane_learned_depth_seed_model(
        ModelConfig(base_filters=1)
    )
    wrapper = PhysicsConsistencyTrainingModel(
        inversion,
        DifferentiableSinglePlaneGz(),
        gravity_scale=0.22938017547130585,
        gravity_loss_weight=0.001,
    )
    wrapper.compile(optimizer=tf.keras.optimizers.Adam(1.0e-3))
    gravity = np.ones((1, *SINGLE_PLANE_GRAVITY_SHAPE), np.float32) * 0.1
    density = np.zeros((1, *DENSITY_SHAPE), np.float32)
    density[:, 2:4, 30:34, 30:34, :] = 0.5
    terms = wrapper.compute_loss_terms(gravity, density, training=False)
    np.testing.assert_allclose(
        terms[4].numpy(),
        terms[1].numpy() + 0.001 * terms[2].numpy(),
        rtol=1.0e-6,
    )
    logs = wrapper.train_on_batch(gravity, density, return_dict=True)
    assert np.isfinite(logs["loss"])
    assert wrapper.inversion_model is inversion
