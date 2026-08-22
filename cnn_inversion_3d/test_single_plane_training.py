"""Focused tests for the additive single-plane CNN workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from cnn_inversion_3d.dataset import (
    DENSITY_SHAPE,
    GRAVITY_SHAPE,
    SINGLE_PLANE_GRAVITY_SHAPE,
    load_npz_sample,
)
from cnn_inversion_3d.gravity_consistency import (
    SinglePlaneForwardOperator,
    evaluate_gravity_consistency,
)
from cnn_inversion_3d.model import (
    ModelConfig,
    build_asymmetric_2d_unet_model,
    build_baseline_model,
    build_single_plane_model,
    build_single_plane_learned_depth_seed_model,
    compile_baseline_model,
)
from cnn_inversion_3d.differentiable_gravity import DifferentiableSinglePlaneGz


def _write_sample(path: Path) -> None:
    """Write one minimal correctly shaped single-plane pair."""

    density = np.zeros(DENSITY_SHAPE[:-1], dtype=np.float32)
    density[2:4, 30:34, 30:34] = 0.5
    np.savez_compressed(
        path,
        gravity=np.zeros(SINGLE_PLANE_GRAVITY_SHAPE[:-1], np.float32),
        density=density,
    )


def test_single_plane_sample_and_tensor_shapes(tmp_path: Path) -> None:
    """Verify disk samples become channels-last 81 x 81 inputs."""

    path = tmp_path / "sample.npz"
    _write_sample(path)
    gravity, density = load_npz_sample(
        path, gravity_shape=SINGLE_PLANE_GRAVITY_SHAPE
    )
    assert gravity.shape == SINGLE_PLANE_GRAVITY_SHAPE
    assert density.shape == DENSITY_SHAPE


def test_single_plane_model_forward_and_output_shape() -> None:
    """Verify the dedicated model maps a surface to the density volume."""

    model = build_single_plane_model(
        ModelConfig(base_filters=1, vertical_expansion="transpose")
    )
    prediction = model(
        tf.zeros((1, *SINGLE_PLANE_GRAVITY_SHAPE)), training=False
    )
    assert model.input_shape == (None, *SINGLE_PLANE_GRAVITY_SHAPE)
    assert tuple(prediction.shape) == (1, *DENSITY_SHAPE)


def test_single_plane_tiny_training_batch() -> None:
    """Verify BalancedDensityMSE training accepts one surface batch."""

    config = ModelConfig(base_filters=1, vertical_expansion="transpose")
    model = build_single_plane_model(config)
    compile_baseline_model(model, config)
    gravity = np.zeros((1, *SINGLE_PLANE_GRAVITY_SHAPE), np.float32)
    density = np.zeros((1, *DENSITY_SHAPE), np.float32)
    density[:, 2:4, 30:34, 30:34, :] = 0.5
    loss = model.train_on_batch(gravity, density)
    assert np.all(np.isfinite(np.asarray(loss)))


def test_single_plane_saved_model_prediction_loading(tmp_path: Path) -> None:
    """Verify the additive architecture survives Keras serialization."""

    model = build_single_plane_model(ModelConfig(base_filters=1))
    path = tmp_path / "model.keras"
    model.save(path)
    loaded = tf.keras.models.load_model(path, compile=False)
    assert loaded.input_shape == (None, *SINGLE_PLANE_GRAVITY_SHAPE)
    assert loaded.output_shape == (None, *DENSITY_SHAPE)


class _OneLevelForward:
    """Tiny three-dimensional one-level operator for adapter testing."""

    input_shape = DENSITY_SHAPE[:-1]
    output_shape = (1, 81, 81)

    def calculate(self, model: np.ndarray) -> np.ndarray:
        value = float(np.mean(model))
        return np.full(self.output_shape, value, dtype=np.float64)


def test_single_plane_gravity_consistency() -> None:
    """Verify consistency evaluation accepts two-dimensional surface Gz."""

    operator = SinglePlaneForwardOperator(_OneLevelForward())
    density = np.zeros(DENSITY_SHAPE[:-1], np.float32)
    result = evaluate_gravity_consistency(
        np.zeros((81, 81)), density, forward_model=operator
    )
    assert result.recovered_gravity.shape == (1, 81, 81)
    assert result.metrics["gravity_rmse"] == 0.0


def test_legacy_model_shape_is_unchanged() -> None:
    """Verify the E00–E04 builder retains its multi-height interface."""

    model = build_baseline_model(ModelConfig(base_filters=1))
    assert model.input_shape == (None, *GRAVITY_SHAPE)
    assert model.output_shape == (None, *DENSITY_SHAPE)


def test_e05_repeated_depth_seed_is_unchanged() -> None:
    """Verify the original E05 lifting operation remains intact."""

    model = build_single_plane_model(ModelConfig(base_filters=1))
    layer = model.get_layer("seed_six_depth_planes")
    assert isinstance(layer, tf.keras.layers.UpSampling3D)
    assert tuple(layer.output.shape) == (None, 6, 64, 64, 1)


def test_e06_learned_depth_seed_shape_and_layers() -> None:
    """Verify E06 learns channels and explicitly permutes depth first."""

    model = build_single_plane_learned_depth_seed_model(
        ModelConfig(base_filters=2)
    )
    assert model.input_shape == (None, *SINGLE_PLANE_GRAVITY_SHAPE)
    assert model.output_shape == (None, *DENSITY_SHAPE)
    assert tuple(
        model.get_layer("reshape_learned_depth_seed").output.shape
    ) == (None, 6, 64, 64, 2)
    assert isinstance(
        model.get_layer("learn_depth_specific_features"),
        tf.keras.layers.Conv2D,
    )
    assert not any(
        isinstance(layer, tf.keras.layers.UpSampling3D)
        and layer.name == "seed_six_depth_planes"
        for layer in model.layers
    )


def test_e06_forward_and_tiny_training_step() -> None:
    """Verify the learned-seed model supports inference and training."""

    config = ModelConfig(base_filters=1)
    model = build_single_plane_learned_depth_seed_model(config)
    compile_baseline_model(model, config)
    gravity = np.zeros((1, *SINGLE_PLANE_GRAVITY_SHAPE), np.float32)
    density = np.zeros((1, *DENSITY_SHAPE), np.float32)
    density[:, 2:4, 30:34, 30:34, :] = 0.5
    prediction = model(gravity, training=False)
    loss = model.train_on_batch(gravity, density)
    assert tuple(prediction.shape) == (1, *DENSITY_SHAPE)
    assert np.all(np.isfinite(np.asarray(loss)))


def test_e06_serialization_loading(tmp_path: Path) -> None:
    """Verify E06 model configuration survives Keras serialization."""

    model = build_single_plane_learned_depth_seed_model(
        ModelConfig(base_filters=1)
    )
    path = tmp_path / "e06.keras"
    model.save(path)
    loaded = tf.keras.models.load_model(path, compile=False)
    assert loaded.input_shape == (None, *SINGLE_PLANE_GRAVITY_SHAPE)
    assert loaded.output_shape == (None, *DENSITY_SHAPE)
    assert loaded.get_layer("reshape_learned_depth_seed") is not None


def test_e09_shape_progression_and_direct_depth_channels() -> None:
    """Verify E09 maps the padded 2D U-Net directly to canonical density."""

    model = build_asymmetric_2d_unet_model(ModelConfig(base_filters=1))
    assert model.input_shape == (None, 81, 81, 1)
    assert tuple(model.get_layer("e09_pad_complete_81_to_96").output.shape) == (
        None, 96, 96, 1
    )
    assert tuple(model.get_layer("e09_bottleneck_conv_2").output.shape) == (
        None, 12, 12, 8
    )
    assert tuple(
        model.get_layer("e09_learned_spatial_transform_96_to_64").output.shape
    ) == (None, 64, 64, 1)
    assert tuple(model.get_layer("e09_density_depth_channels").output.shape) == (
        None, 64, 64, 24
    )
    assert tuple(
        model.get_layer("e09_permute_depth_channels_first").output.shape
    ) == (None, 24, 64, 64)
    assert model.output_shape == (None, 24, 64, 64, 1)
    assert not any(isinstance(layer, tf.keras.layers.Conv3D) for layer in model.layers)
    assert not any(isinstance(layer, tf.keras.layers.Conv3DTranspose) for layer in model.layers)


def test_e09_padding_preserves_every_raw_gravity_sample() -> None:
    """Verify no 81 x 81 observation is discarded before feature extraction."""

    model = build_asymmetric_2d_unet_model(ModelConfig(base_filters=1))
    padding_model = tf.keras.Model(
        model.input, model.get_layer("e09_pad_complete_81_to_96").output
    )
    values = np.arange(81 * 81, dtype=np.float32).reshape(1, 81, 81, 1)
    padded = padding_model(values, training=False).numpy()
    np.testing.assert_array_equal(padded[:, 7:88, 7:88, :], values)
    assert np.count_nonzero(padded[:, :7]) == 0
    assert np.count_nonzero(padded[:, 88:]) == 0


def test_e09_smoke_training_batch_and_serialization(tmp_path: Path) -> None:
    """Verify BalancedDensityMSE updates E09 and saved inference reloads."""

    config = ModelConfig(base_filters=1)
    model = build_asymmetric_2d_unet_model(config)
    compile_baseline_model(model, config)
    gravity = np.zeros((1, *SINGLE_PLANE_GRAVITY_SHAPE), np.float32)
    density = np.zeros((1, *DENSITY_SHAPE), np.float32)
    density[:, 2:4, 30:34, 30:34, :] = 0.5
    loss = model.train_on_batch(gravity, density)
    assert np.all(np.isfinite(np.asarray(loss)))
    path = tmp_path / "e09.keras"
    model.save(path)
    loaded = tf.keras.models.load_model(path, compile=False)
    prediction = loaded(gravity, training=False)
    assert tuple(prediction.shape) == (1, *DENSITY_SHAPE)


def test_e09_prediction_is_accepted_by_differentiable_gravity() -> None:
    """Verify E09 output needs no adapter for the E07 physics operator."""

    model = build_asymmetric_2d_unet_model(ModelConfig(base_filters=1))
    density = model(
        tf.zeros((1, *SINGLE_PLANE_GRAVITY_SHAPE)), training=False
    )
    gravity = DifferentiableSinglePlaneGz()(density)
    assert tuple(density.shape) == (1, *DENSITY_SHAPE)
    assert tuple(gravity.shape) == (1, *SINGLE_PLANE_GRAVITY_SHAPE)
