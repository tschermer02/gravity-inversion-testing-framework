"""Focused tests for the E09B sensitivity-compensation ablation."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from cnn_inversion_3d.dataset import DENSITY_SHAPE, SINGLE_PLANE_GRAVITY_SHAPE
from cnn_inversion_3d.e09a_training import E09ADepthLossConfig, E09ATrainingModel
from cnn_inversion_3d.e09b_training import (
    E09BLossConfig, E09BTrainingModel, build_e09b_sensitivity_weights,
    integrated_sensitivity_compensated_mse,
)
from cnn_inversion_3d.model import ModelConfig, build_asymmetric_2d_unet_model
from cnn_inversion_3d.train import build_inversion_model_for_architecture


@pytest.fixture(scope="module")
def sensitivity_data() -> tuple[np.ndarray, np.ndarray]:
    return build_e09b_sensitivity_weights(
        gamma=0.5, weight_min=0.5, weight_max=5.0
    )


def _batch() -> tuple[tf.Tensor, tf.Tensor]:
    gravity = tf.zeros((1, *SINGLE_PLANE_GRAVITY_SHAPE))
    density = np.zeros((1, *DENSITY_SHAPE), np.float32)
    density[:, 3:7, 28:34, 30:36, :] = 0.6
    return gravity, tf.constant(density)


def test_e09b_weights_are_physical_bounded_mean_one_and_deeper_larger(
    sensitivity_data: tuple[np.ndarray, np.ndarray],
) -> None:
    sensitivity, weights = sensitivity_data
    assert sensitivity.shape == weights.shape == (24, 64, 64)
    assert np.all(np.isfinite(sensitivity)) and np.all(np.isfinite(weights))
    np.testing.assert_allclose(np.mean(weights), 1.0, atol=1e-6)
    assert float(np.min(weights)) >= 0.5 and float(np.max(weights)) <= 5.0
    assert float(np.mean(sensitivity[-4:])) < float(np.mean(sensitivity[:4]))
    assert float(np.mean(weights[-4:])) > float(np.mean(weights[:4]))


def test_e09b_architecture_is_identical_to_e09_and_e09a() -> None:
    e09 = build_asymmetric_2d_unet_model(ModelConfig(base_filters=1))
    e09b = build_asymmetric_2d_unet_model(ModelConfig(base_filters=1))
    assert e09.to_json() == e09b.to_json()
    assert e09.count_params() == e09b.count_params()
    assert not any(isinstance(layer, tf.keras.layers.Conv3D) for layer in e09b.layers)


def test_e09b_training_selector_dispatches_to_e09_model() -> None:
    model = build_inversion_model_for_architecture(
        "single_plane_asymmetric_2d_unet_sensitivity_loss",
        ModelConfig(base_filters=1),
    )
    assert model.name == "e09_asymmetric_2d_unet"
    assert model.input_shape == (None, 81, 81, 1)
    assert model.output_shape == (None, 24, 64, 64, 1)


def test_e09b_zero_lambda_reproduces_e09a_and_positive_adds_exact_term(
    sensitivity_data: tuple[np.ndarray, np.ndarray],
) -> None:
    _, weights = sensitivity_data
    gravity, truth = _batch()
    e09a_model = build_asymmetric_2d_unet_model(ModelConfig(base_filters=1))
    zero_model = build_asymmetric_2d_unet_model(ModelConfig(base_filters=1))
    positive_model = build_asymmetric_2d_unet_model(ModelConfig(base_filters=1))
    zero_model.set_weights(e09a_model.get_weights())
    positive_model.set_weights(e09a_model.get_weights())
    e09a = E09ATrainingModel(e09a_model, loss_config=E09ADepthLossConfig())
    zero = E09BTrainingModel(
        zero_model, weights, loss_config=E09BLossConfig(lambda_sensitivity=0.0)
    )
    positive = E09BTrainingModel(
        positive_model, weights, loss_config=E09BLossConfig(lambda_sensitivity=2.0)
    )
    a_terms = e09a.compute_loss_terms(gravity, truth, training=False)
    zero_terms = zero.compute_loss_terms(gravity, truth, training=False)
    positive_terms = positive.compute_loss_terms(gravity, truth, training=False)
    np.testing.assert_allclose(zero_terms[-1], a_terms[-1], rtol=1e-6)
    expected = positive_terms[1] + positive_terms[4] + 2.0 * positive_terms[5]
    np.testing.assert_allclose(positive_terms[-1], expected, rtol=1e-6)
    assert float(positive_terms[5].numpy()) > 0.0


def test_e09b_sensitivity_loss_is_differentiable_and_reaches_e09_parameters(
    sensitivity_data: tuple[np.ndarray, np.ndarray],
) -> None:
    _, weights = sensitivity_data
    gravity, truth = _batch()
    prediction = tf.Variable(tf.ones_like(truth) * 0.05)
    with tf.GradientTape() as tape:
        direct_loss = integrated_sensitivity_compensated_mse(truth, prediction, weights)
    direct_gradient = tape.gradient(direct_loss, prediction)
    assert direct_gradient is not None and np.any(direct_gradient.numpy() != 0.0)

    inversion = build_asymmetric_2d_unet_model(ModelConfig(base_filters=1))
    wrapper = E09BTrainingModel(inversion, weights, loss_config=E09BLossConfig())
    with tf.GradientTape() as tape:
        sensitivity_loss = wrapper.compute_loss_terms(gravity, truth, training=True)[5]
    gradients = tape.gradient(sensitivity_loss, inversion.trainable_variables)
    assert any(value is not None and np.any(value.numpy() != 0.0) for value in gradients)
