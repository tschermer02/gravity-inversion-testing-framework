"""Focused tests for the E09A loss-only depth-supervision ablation."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from cnn_inversion_3d.dataset import DENSITY_SHAPE, SINGLE_PLANE_GRAVITY_SHAPE
from cnn_inversion_3d.e09a_training import (
    E09ADepthLossConfig,
    E09ATrainingModel,
    density_depth_profile,
    depth_profile_mse,
    normalized_depth_profile,
    z_center_mse,
)
from cnn_inversion_3d.model import ModelConfig, build_asymmetric_2d_unet_model


def _density() -> tf.Tensor:
    values = np.zeros((1, *DENSITY_SHAPE), np.float32)
    values[:, 2:5, 30:34, 31:35, :] = 0.6
    return tf.constant(values)


def test_e09a_reuses_identical_e09_architecture() -> None:
    e09 = build_asymmetric_2d_unet_model(ModelConfig(base_filters=1))
    e09a = build_asymmetric_2d_unet_model(ModelConfig(base_filters=1))
    assert e09.to_json() == e09a.to_json()
    assert e09.count_params() == e09a.count_params()
    assert e09a.output_shape == (None, *DENSITY_SHAPE)
    assert [layer.name for layer in e09.layers] == [layer.name for layer in e09a.layers]
    assert not any(isinstance(layer, tf.keras.layers.Conv3D) for layer in e09a.layers)


def test_depth_profile_shape_and_normalization() -> None:
    density = _density()
    profile = density_depth_profile(density)
    normalized = normalized_depth_profile(density)
    assert profile.shape == (1, 24)
    np.testing.assert_allclose(tf.reduce_sum(normalized, axis=1), [1.0], rtol=1e-6)


def test_depth_losses_are_finite_and_differentiable() -> None:
    truth = _density()
    prediction = tf.Variable(tf.ones_like(truth) * 0.05)
    with tf.GradientTape(persistent=True) as tape:
        profile_loss = depth_profile_mse(truth, prediction)
        center_loss = z_center_mse(truth, prediction)
    for loss in (profile_loss, center_loss):
        gradient = tape.gradient(loss, prediction)
        assert gradient is not None
        assert np.all(np.isfinite(gradient.numpy()))
    assert np.all(np.isfinite([profile_loss.numpy(), center_loss.numpy()]))


def test_e09a_gradients_reach_e09_parameters_and_smoke_train() -> None:
    inversion = build_asymmetric_2d_unet_model(ModelConfig(base_filters=1))
    wrapper = E09ATrainingModel(inversion, loss_config=E09ADepthLossConfig())
    wrapper.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    gravity = tf.zeros((1, *SINGLE_PLANE_GRAVITY_SHAPE))
    truth = _density()
    with tf.GradientTape() as tape:
        terms = wrapper.compute_loss_terms(gravity, truth, training=True)
    gradients = tape.gradient(terms[-1], inversion.trainable_variables)
    assert any(gradient is not None for gradient in gradients)
    result = wrapper.train_on_batch(gravity, truth, return_dict=True)
    assert {
        "loss", "density_loss", "depth_profile_loss", "z_center_loss", "depth_loss"
    } <= set(result)
    assert np.all(np.isfinite(list(result.values())))


def test_e09a_total_loss_adds_exact_weighted_depth_objective() -> None:
    gravity = tf.zeros((1, *SINGLE_PLANE_GRAVITY_SHAPE))
    truth = _density()
    density_only_model = build_asymmetric_2d_unet_model(ModelConfig(base_filters=1))
    depth_model = build_asymmetric_2d_unet_model(ModelConfig(base_filters=1))
    depth_model.set_weights(density_only_model.get_weights())
    density_only = E09ATrainingModel(
        density_only_model,
        loss_config=E09ADepthLossConfig(
            lambda_density=1.0, lambda_depth=0.0, alpha_center=1.0
        ),
    )
    with_depth = E09ATrainingModel(
        depth_model,
        loss_config=E09ADepthLossConfig(
            lambda_density=1.0, lambda_depth=2.0, alpha_center=0.5
        ),
    )
    density_terms = density_only.compute_loss_terms(gravity, truth, training=False)
    depth_terms = with_depth.compute_loss_terms(gravity, truth, training=False)
    np.testing.assert_allclose(density_terms[-1], density_terms[1], rtol=1e-6)
    expected_depth = depth_terms[2] + 0.5 * depth_terms[3]
    np.testing.assert_allclose(depth_terms[4], expected_depth, rtol=1e-6)
    np.testing.assert_allclose(
        depth_terms[-1], depth_terms[1] + 2.0 * expected_depth, rtol=1e-6
    )
    assert float(expected_depth.numpy()) > 0.0
    assert float(depth_terms[-1].numpy()) > float(density_terms[-1].numpy())
