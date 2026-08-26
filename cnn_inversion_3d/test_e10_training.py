"""Focused architecture and loss tests for E10."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from cnn_inversion_3d.dataset import DENSITY_SHAPE, SINGLE_PLANE_GRAVITY_SHAPE
from cnn_inversion_3d.differentiable_gravity import (
    DifferentiableSinglePlaneGz,
    build_depth_kernels,
)
from cnn_inversion_3d.e10_training import (
    E10LossConfig,
    E10TrainingModel,
    build_e10_sensitivity_weights,
    data_weighted_gravity_loss,
    sensitivity_balanced_mse,
    soft_iou_loss,
)
from cnn_inversion_3d.model import ModelConfig, build_e10_sensitivity_unet_model
from cnn_inversion_3d.single_plane_review import SinglePlaneReviewConfig


def test_e10_shape_progression_padding_and_no_3d_convolutions() -> None:
    model = build_e10_sensitivity_unet_model(ModelConfig(base_filters=1))
    expected = {
        "e10_pad_complete_81_to_128": (None, 128, 128, 1),
        "e10_encoder_128_conv_2": (None, 128, 128, 1),
        "e10_encoder_64_conv_2": (None, 64, 64, 2),
        "e10_encoder_32_conv_2": (None, 32, 32, 4),
        "e10_bottleneck_16_conv_2": (None, 16, 16, 8),
        "e10_decoder_32_conv_2": (None, 32, 32, 4),
        "e10_decoder_64_conv_2": (None, 64, 64, 2),
        "e10_final_64_features_conv_2": (None, 64, 64, 1),
        "e10_density_depth_channels": (None, 64, 64, 24),
        "e10_permute_depth_channels_first": (None, 24, 64, 64),
    }
    for name, shape in expected.items():
        assert tuple(model.get_layer(name).output.shape) == shape
    assert model.output_shape == (None, *DENSITY_SHAPE)
    assert not any(isinstance(layer, (tf.keras.layers.Conv3D, tf.keras.layers.Conv3DTranspose)) for layer in model.layers)

    pad_model = tf.keras.Model(model.input, model.get_layer("e10_pad_complete_81_to_128").output)
    values = np.arange(81 * 81, dtype=np.float32).reshape(1, 81, 81, 1)
    padded = pad_model(values).numpy()
    np.testing.assert_array_equal(padded[:, 23:104, 23:104, :], values)


def test_e10_weights_are_finite_mean_one_and_match_kernel_ordering() -> None:
    sensitivity, weights, data_weights = build_e10_sensitivity_weights(gamma=0.5)
    assert sensitivity.shape == (24, 64, 64)
    assert data_weights.shape == (81, 81)
    assert np.all(np.isfinite(weights))
    np.testing.assert_allclose(np.mean(weights), 1.0, rtol=1.0e-6)
    assert float(np.mean(weights[-1])) > float(np.mean(weights[0]))

    kernels = build_depth_kernels(SinglePlaneReviewConfig())
    z, y, x = 3, 17, 29
    expected_s2 = sum(
        float(kernels[z, 63 + ry - y, 63 + rx - x]) ** 2
        for ry in range(81) for rx in range(81)
    )
    np.testing.assert_allclose(sensitivity[z, y, x] ** 2, expected_s2, rtol=2.0e-5)
    receiver_y, receiver_x = 40, 40
    expected_wd2 = sum(
        float(kernels[depth, 63 + receiver_y - cell_y, 63 + receiver_x - cell_x]) ** 2
        for depth in range(24) for cell_y in range(64) for cell_x in range(64)
    )
    np.testing.assert_allclose(
        data_weights[receiver_y, receiver_x] ** 2, expected_wd2, rtol=2.0e-5
    )


def test_e10_three_terms_are_finite_and_differentiable() -> None:
    truth = tf.zeros((1, *DENSITY_SHAPE), tf.float32)
    truth = tf.tensor_scatter_nd_update(truth, [[0, 2, 30, 30, 0]], [0.5])
    prediction = tf.Variable(tf.ones_like(truth) * 0.1)
    sensitivity_weights = tf.ones(DENSITY_SHAPE[:-1], tf.float32)
    observed = tf.zeros((1, *SINGLE_PLANE_GRAVITY_SHAPE), tf.float32)
    predicted_gravity = tf.Variable(tf.ones_like(observed) * 0.01)
    with tf.GradientTape(persistent=True) as tape:
        iou = soft_iou_loss(truth, prediction, threshold=0.1, sharpness=60, epsilon=1e-8)
        sensitivity = sensitivity_balanced_mse(truth, prediction, sensitivity_weights)
        gravity = data_weighted_gravity_loss(observed, predicted_gravity, tf.ones((81, 81)))
    assert tape.gradient(iou, prediction) is not None
    assert tape.gradient(sensitivity, prediction) is not None
    assert tape.gradient(gravity, predicted_gravity) is not None
    assert np.all(np.isfinite([iou.numpy(), sensitivity.numpy(), gravity.numpy()]))


def test_e10_wrapper_accepts_one_smoke_batch() -> None:
    inversion = build_e10_sensitivity_unet_model(ModelConfig(base_filters=1))
    _, weights, data_weights = build_e10_sensitivity_weights(gamma=0.5)
    wrapper = E10TrainingModel(
        inversion,
        DifferentiableSinglePlaneGz(),
        weights,
        data_weights,
        gravity_scale=0.22938017547130585,
        loss_config=E10LossConfig(),
    )
    wrapper.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    gravity = np.zeros((1, *SINGLE_PLANE_GRAVITY_SHAPE), np.float32)
    density = np.zeros((1, *DENSITY_SHAPE), np.float32)
    density[:, 2:4, 30:34, 30:34, :] = 0.5
    result = wrapper.train_on_batch(gravity, density, return_dict=True)
    assert set(("loss", "iou_loss", "sensitivity_loss", "gravity_loss")) <= set(result)
    assert np.all(np.isfinite(list(result.values())))
