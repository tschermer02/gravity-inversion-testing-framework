"""Differentiable single-plane Gz operator and E07 training wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from cnn_inversion_3d.diagnostics import build_prediction_diagnostics
from cnn_inversion_3d.model import BalancedDensityMSE
from cnn_inversion_3d.single_plane_review import SinglePlaneReviewConfig
from forward_modeling.matlab_fwd3d.gravity import GRAVITATIONAL_CONSTANT


FFT_SIZE = 256
KERNEL_SIZE = 144
OUTPUT_START = 63


def build_depth_kernels(
    config: SinglePlaneReviewConfig,
    *,
    dtype: np.dtype[Any] = np.dtype(np.float64),
) -> np.ndarray:
    """Build exact FWD3D Gz convolution kernels for all model depths."""

    # Full convolution uses h(q)=f(-q), q=-63,...,80. The stored kernel
    # index k=q+63 therefore has cell-receiver lateral offset 715-10*k.
    kernel_index = np.arange(KERNEL_SIZE, dtype=np.float64)
    delta_x = (
        config.observation_x_m[-1]
        - config.density_x_min_center_m
        - config.dx_m * kernel_index
    )
    delta_y = (
        config.observation_y_m[-1]
        - config.density_y_min_center_m
        - config.dy_m * kernel_index
    )
    dy, dx = np.meshgrid(delta_y, delta_x, indexing="ij")
    cell_volume = config.dx_m * config.dy_m * config.dz_m
    gamma = 1.0e8 * GRAVITATIONAL_CONSTANT * cell_volume
    kernels = np.empty(
        (config.nz, KERNEL_SIZE, KERNEL_SIZE), dtype=np.float64
    )
    for depth_index in range(config.nz):
        dz = config.density_z_min_center_m + depth_index * config.dz_m
        radius_squared = dx**2 + dy**2 + dz**2
        kernels[depth_index] = gamma * dz / np.power(radius_squared, 1.5)
    return kernels.astype(dtype, copy=False)


class DifferentiableSinglePlaneGz(tf.keras.layers.Layer):
    """Apply the fixed FWD3D Gz operator using differentiable 2D FFTs."""

    def __init__(
        self,
        config: SinglePlaneReviewConfig | None = None,
        *,
        calculation_dtype: tf.dtypes.DType = tf.float32,
        name: str = "differentiable_single_plane_gz",
    ) -> None:
        super().__init__(name=name, trainable=False, dtype=calculation_dtype)
        self.config = config or SinglePlaneReviewConfig()
        self.calculation_dtype = tf.as_dtype(calculation_dtype)
        kernels = build_depth_kernels(
            self.config,
            dtype=np.dtype(self.calculation_dtype.as_numpy_dtype),
        )
        padded = np.pad(
            kernels,
            ((0, 0), (0, FFT_SIZE - KERNEL_SIZE), (0, FFT_SIZE - KERNEL_SIZE)),
        )
        self.kernel_spectrum = tf.constant(
            np.fft.rfft2(padded, axes=(-2, -1)),
            dtype=(
                tf.complex128
                if self.calculation_dtype == tf.float64
                else tf.complex64
            ),
        )

    def call(self, density: tf.Tensor) -> tf.Tensor:
        """Return surface Gz in mGal with shape ``(batch,81,81,1)``."""

        values = tf.cast(density, self.calculation_dtype)
        if values.shape.rank != 5:
            raise ValueError("Density must have rank five: (batch,z,y,x,1).")
        values = tf.squeeze(values, axis=-1)
        values = tf.pad(values, [[0, 0], [0, 0], [0, 192], [0, 192]])
        density_spectrum = tf.signal.rfft2d(values)
        full = tf.signal.irfft2d(
            density_spectrum * self.kernel_spectrum[tf.newaxis, ...],
            fft_length=(FFT_SIZE, FFT_SIZE),
        )
        surface = tf.reduce_sum(
            full[
                :,
                :,
                OUTPUT_START : OUTPUT_START + 81,
                OUTPUT_START : OUTPUT_START + 81,
            ],
            axis=1,
        )
        return surface[..., tf.newaxis]


def relative_gravity_loss(
    true_gravity_mgal: tf.Tensor,
    predicted_gravity_mgal: tf.Tensor,
    *,
    epsilon: float = 1.0e-12,
) -> tf.Tensor:
    """Return batch-mean per-sample squared relative L2 gravity error."""

    true_values = tf.cast(true_gravity_mgal, predicted_gravity_mgal.dtype)
    axes = tuple(range(1, predicted_gravity_mgal.shape.rank))
    numerator = tf.reduce_sum(
        tf.square(predicted_gravity_mgal - true_values), axis=axes
    )
    denominator = tf.reduce_sum(tf.square(true_values), axis=axes)
    return tf.reduce_mean(numerator / (denominator + epsilon))


class PhysicsConsistencyTrainingModel(tf.keras.Model):
    """Train an unchanged inversion model with the additive E07 objective."""

    def __init__(
        self,
        inversion_model: tf.keras.Model,
        forward_operator: DifferentiableSinglePlaneGz,
        *,
        gravity_scale: float,
        gravity_loss_weight: float = 0.1,
        body_loss_fraction: float = 0.5,
    ) -> None:
        super().__init__(name="e07_physics_consistency_training_wrapper")
        self.inversion_model = inversion_model
        self.forward_operator = forward_operator
        self.gravity_scale = float(gravity_scale)
        self.gravity_loss_weight = float(gravity_loss_weight)
        self.density_loss_function = BalancedDensityMSE(
            body_fraction=body_loss_fraction
        )
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.density_loss_tracker = tf.keras.metrics.Mean(name="density_loss")
        self.gravity_loss_tracker = tf.keras.metrics.Mean(name="gravity_loss")
        self.weighted_gravity_tracker = tf.keras.metrics.Mean(
            name="weighted_gravity_loss"
        )
        self.gravity_rmse_tracker = tf.keras.metrics.Mean(name="gravity_rmse")
        self.gravity_relative_l2_tracker = tf.keras.metrics.Mean(
            name="gravity_relative_l2"
        )
        self.gravity_correlation_tracker = tf.keras.metrics.Mean(
            name="gravity_correlation"
        )
        self.density_diagnostics = build_prediction_diagnostics()

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        """Return all epoch-reset Keras metrics."""

        return [
            self.total_loss_tracker,
            self.density_loss_tracker,
            self.gravity_loss_tracker,
            self.weighted_gravity_tracker,
            self.gravity_rmse_tracker,
            self.gravity_relative_l2_tracker,
            self.gravity_correlation_tracker,
            *self.density_diagnostics,
        ]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Delegate inference to the unchanged E06 inversion model."""

        return self.inversion_model(inputs, training=training)

    def compute_loss_terms(
        self,
        gravity_normalized: tf.Tensor,
        true_density: tf.Tensor,
        *,
        training: bool,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Return prediction, density, gravity, weighted, and total losses."""

        predicted_density = self.inversion_model(
            gravity_normalized, training=training
        )
        density_loss = self.density_loss_function(
            true_density, predicted_density
        )
        true_gravity_mgal = tf.cast(
            gravity_normalized * self.gravity_scale, tf.float32
        )
        predicted_gravity_mgal = self.forward_operator(predicted_density)
        gravity_loss = relative_gravity_loss(
            true_gravity_mgal, predicted_gravity_mgal
        )
        weighted_gravity = self.gravity_loss_weight * gravity_loss
        total_loss = density_loss + weighted_gravity
        return (
            predicted_density,
            density_loss,
            gravity_loss,
            weighted_gravity,
            total_loss,
        )

    def _update_metrics(
        self,
        gravity_normalized: tf.Tensor,
        true_density: tf.Tensor,
        terms: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    ) -> None:
        """Update loss, gravity, and existing density diagnostics."""

        prediction, density_loss, gravity_loss, weighted, total = terms
        true_gravity = tf.cast(gravity_normalized * self.gravity_scale, tf.float32)
        predicted_gravity = self.forward_operator(prediction)
        residual = predicted_gravity - true_gravity
        self.total_loss_tracker.update_state(total)
        self.density_loss_tracker.update_state(density_loss)
        self.gravity_loss_tracker.update_state(gravity_loss)
        self.weighted_gravity_tracker.update_state(weighted)
        self.gravity_rmse_tracker.update_state(tf.sqrt(tf.reduce_mean(residual**2)))
        self.gravity_relative_l2_tracker.update_state(tf.sqrt(gravity_loss))
        true_flat = tf.reshape(true_gravity, (tf.shape(true_gravity)[0], -1))
        pred_flat = tf.reshape(predicted_gravity, (tf.shape(predicted_gravity)[0], -1))
        true_centered = true_flat - tf.reduce_mean(true_flat, axis=1, keepdims=True)
        pred_centered = pred_flat - tf.reduce_mean(pred_flat, axis=1, keepdims=True)
        correlation = tf.reduce_sum(true_centered * pred_centered, axis=1) / (
            tf.norm(true_centered, axis=1) * tf.norm(pred_centered, axis=1) + 1.0e-12
        )
        self.gravity_correlation_tracker.update_state(tf.reduce_mean(correlation))
        for metric in self.density_diagnostics:
            metric.update_state(true_density, prediction)

    def train_step(self, data: Any) -> dict[str, tf.Tensor]:
        """Apply one differentiable E07 optimizer step."""

        gravity, density, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            terms = self.compute_loss_terms(gravity, density, training=True)
        gradients = tape.gradient(terms[-1], self.inversion_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.inversion_model.trainable_variables)
        )
        self._update_metrics(gravity, density, terms)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: Any) -> dict[str, tf.Tensor]:
        """Evaluate one validation batch without optimizer updates."""

        gravity, density, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        terms = self.compute_loss_terms(gravity, density, training=False)
        self._update_metrics(gravity, density, terms)
        return {metric.name: metric.result() for metric in self.metrics}


class SaveBestInversionModel(tf.keras.callbacks.Callback):
    """Save the underlying inference network at the best validation loss."""

    def __init__(
        self,
        inversion_model: tf.keras.Model,
        output_path: Path,
    ) -> None:
        super().__init__()
        self.inversion_model = inversion_model
        self.output_path = output_path
        self.best = float("inf")

    def on_epoch_end(
        self, epoch: int, logs: dict[str, float] | None = None
    ) -> None:
        """Save when ``val_loss`` reaches a new finite minimum."""

        del epoch
        value = (logs or {}).get("val_loss")
        if value is not None and np.isfinite(value) and value < self.best:
            self.best = float(value)
            self.inversion_model.save(self.output_path)
