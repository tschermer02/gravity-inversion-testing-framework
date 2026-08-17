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
    build_baseline_model,
    build_single_plane_model,
    compile_baseline_model,
)


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
