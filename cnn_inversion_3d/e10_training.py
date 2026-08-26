"""E10 sensitivity- and physics-informed losses and training wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from cnn_inversion_3d.diagnostics import build_prediction_diagnostics
from cnn_inversion_3d.differentiable_gravity import (
    DifferentiableSinglePlaneGz,
    build_depth_kernels,
)
from cnn_inversion_3d.single_plane_review import SinglePlaneReviewConfig


@dataclass(frozen=True)
class E10LossConfig:
    """Configurable E10 loss weights and numerical parameters."""

    lambda_shape: float = 1.0
    lambda_sensitivity: float = 1.0
    lambda_physics: float = 1.0e-3
    sensitivity_gamma: float = 0.5
    occupancy_threshold: float = 0.1
    occupancy_sharpness: float = 60.0
    epsilon: float = 1.0e-8
    body_fraction: float = 0.5

    def validate(self) -> None:
        """Validate E10 loss parameters."""

        for name in ("lambda_shape", "lambda_sensitivity", "lambda_physics"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must not be negative.")
        if self.sensitivity_gamma < 0.0:
            raise ValueError("sensitivity_gamma must not be negative.")
        if not 0.0 < self.occupancy_threshold < 1.0:
            raise ValueError("occupancy_threshold must be between zero and one.")
        if self.occupancy_sharpness <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("occupancy_sharpness and epsilon must be positive.")
        if not 0.0 < self.body_fraction < 1.0:
            raise ValueError("body_fraction must be between zero and one.")


def _full_convolution(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Return full 2D convolution using deterministic NumPy FFTs."""

    output_shape = (
        array.shape[0] + kernel.shape[0] - 1,
        array.shape[1] + kernel.shape[1] - 1,
    )
    fft_shape = tuple(1 << (size - 1).bit_length() for size in output_shape)
    result = np.fft.irfft2(
        np.fft.rfft2(array, fft_shape) * np.fft.rfft2(kernel, fft_shape),
        fft_shape,
    )
    return result[: output_shape[0], : output_shape[1]]


def build_e10_sensitivity_weights(
    *,
    gamma: float = 0.5,
    epsilon: float = 1.0e-8,
    config: SinglePlaneReviewConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute ``S_xyz``, mean-one inverse weights, and ``W_d``.

    The fixed kernels are the exact kernels used by
    :class:`DifferentiableSinglePlaneGz`.  Convolution extraction follows its
    canonical ``density[z,y,x]`` and surface ``gravity[y,x]`` ordering.
    """

    geometry = config or SinglePlaneReviewConfig()
    kernels = build_depth_kernels(geometry, dtype=np.dtype(np.float64))
    receiver_mask = np.ones(geometry.gravity_shape, dtype=np.float64)
    cell_mask = np.ones((geometry.ny, geometry.nx), dtype=np.float64)
    sensitivity_squared = np.empty(geometry.density_shape, dtype=np.float64)
    data_weight_squared = np.zeros(geometry.gravity_shape, dtype=np.float64)
    for depth, kernel in enumerate(kernels):
        squared = np.square(kernel)
        receiver_convolution = _full_convolution(receiver_mask, squared)
        sensitivity_squared[depth] = receiver_convolution[
            80 : 80 + geometry.ny, 80 : 80 + geometry.nx
        ]
        cell_convolution = _full_convolution(cell_mask, squared)
        data_weight_squared += cell_convolution[
            63 : 63 + geometry.gravity_shape[0],
            63 : 63 + geometry.gravity_shape[1],
        ]
    sensitivity = np.sqrt(np.maximum(sensitivity_squared, 0.0))
    inverse = np.power(1.0 / (sensitivity + epsilon), gamma)
    sensitivity_weights = inverse / np.mean(inverse)
    data_weights = np.sqrt(np.maximum(data_weight_squared, 0.0))
    if not (
        np.all(np.isfinite(sensitivity))
        and np.all(np.isfinite(sensitivity_weights))
        and np.all(np.isfinite(data_weights))
    ):
        raise ValueError("E10 sensitivity/data weights contain NaN or Inf.")
    if not np.isclose(np.mean(sensitivity_weights), 1.0, rtol=1.0e-6):
        raise ValueError("E10 sensitivity weights do not have mean one.")
    return (
        sensitivity.astype(np.float32),
        sensitivity_weights.astype(np.float32),
        data_weights.astype(np.float32),
    )


def soft_iou_loss(
    true_density: tf.Tensor,
    predicted_density: tf.Tensor,
    *,
    threshold: float,
    sharpness: float,
    epsilon: float,
) -> tf.Tensor:
    """Return mean differentiable 3D IoU loss over a batch."""

    truth = tf.cast(true_density > 0.0, predicted_density.dtype)
    prediction = tf.sigmoid(
        tf.cast(sharpness, predicted_density.dtype)
        * (predicted_density - tf.cast(threshold, predicted_density.dtype))
    )
    axes = tf.range(1, tf.rank(prediction))
    intersection = tf.reduce_sum(prediction * truth, axis=axes)
    union = (
        tf.reduce_sum(prediction, axis=axes)
        + tf.reduce_sum(truth, axis=axes)
        - intersection
    )
    return tf.reduce_mean(
        1.0 - intersection / (union + tf.cast(epsilon, union.dtype))
    )


def sensitivity_balanced_mse(
    true_density: tf.Tensor,
    predicted_density: tf.Tensor,
    sensitivity_weights: tf.Tensor,
    *,
    body_fraction: float = 0.5,
    epsilon: float = 1.0e-8,
) -> tf.Tensor:
    """Combine mean-one sensitivity weights with 50/50 body/background MSE."""

    weights = tf.cast(sensitivity_weights, predicted_density.dtype)[None, ..., None]
    squared_error = tf.square(predicted_density - true_density)
    body = tf.cast(true_density > 0.0, predicted_density.dtype)
    background = 1.0 - body
    axes = tf.range(1, tf.rank(predicted_density))

    def weighted_region(mask: tf.Tensor) -> tf.Tensor:
        numerator = tf.reduce_sum(weights * mask * squared_error, axis=axes)
        denominator = tf.reduce_sum(weights * mask, axis=axes)
        return numerator / (denominator + tf.cast(epsilon, denominator.dtype))

    body_loss = weighted_region(body)
    background_loss = weighted_region(background)
    return tf.reduce_mean(
        tf.cast(body_fraction, predicted_density.dtype) * body_loss
        + tf.cast(1.0 - body_fraction, predicted_density.dtype) * background_loss
    )


def data_weighted_gravity_loss(
    observed_gravity_mgal: tf.Tensor,
    predicted_gravity_mgal: tf.Tensor,
    data_weights: tf.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> tf.Tensor:
    """Return mean squared ``W_d^-1 (d_obs - G m_pred)`` residual."""

    residual = tf.cast(observed_gravity_mgal, predicted_gravity_mgal.dtype) - predicted_gravity_mgal
    weights = tf.cast(data_weights, predicted_gravity_mgal.dtype)[None, ..., None]
    normalized = residual / (weights + tf.cast(epsilon, weights.dtype))
    return tf.reduce_mean(tf.square(normalized))


class E10TrainingModel(tf.keras.Model):
    """Train an E10 inversion network with shape, sensitivity, and physics losses."""

    def __init__(
        self,
        inversion_model: tf.keras.Model,
        forward_operator: DifferentiableSinglePlaneGz,
        sensitivity_weights: np.ndarray,
        data_weights: np.ndarray,
        *,
        gravity_scale: float,
        loss_config: E10LossConfig,
    ) -> None:
        super().__init__(name="e10_training_wrapper")
        loss_config.validate()
        self.inversion_model = inversion_model
        self.forward_operator = forward_operator
        self.gravity_scale = float(gravity_scale)
        self.loss_config = loss_config
        self.sensitivity_weights = tf.constant(sensitivity_weights, tf.float32)
        self.data_weights = tf.constant(data_weights, tf.float32)
        median_weight = float(np.median(sensitivity_weights))
        self.low_sensitivity_mask = tf.constant(
            sensitivity_weights >= median_weight, tf.float32
        )[None, ..., None]
        self.high_sensitivity_mask = 1.0 - self.low_sensitivity_mask
        self.total_tracker = tf.keras.metrics.Mean(name="loss")
        self.iou_tracker = tf.keras.metrics.Mean(name="iou_loss")
        self.sensitivity_tracker = tf.keras.metrics.Mean(name="sensitivity_loss")
        self.gravity_tracker = tf.keras.metrics.Mean(name="gravity_loss")
        self.low_sensitivity_mse_tracker = tf.keras.metrics.Mean(
            name="low_sensitivity_mse"
        )
        self.high_sensitivity_mse_tracker = tf.keras.metrics.Mean(
            name="high_sensitivity_mse"
        )
        self.density_diagnostics = build_prediction_diagnostics()

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [
            self.total_tracker,
            self.iou_tracker,
            self.sensitivity_tracker,
            self.gravity_tracker,
            self.low_sensitivity_mse_tracker,
            self.high_sensitivity_mse_tracker,
            *self.density_diagnostics,
        ]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.inversion_model(inputs, training=training)

    def compute_loss_terms(
        self, gravity_normalized: tf.Tensor, true_density: tf.Tensor, *, training: bool
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        prediction = self.inversion_model(gravity_normalized, training=training)
        cfg = self.loss_config
        iou = soft_iou_loss(
            true_density, prediction, threshold=cfg.occupancy_threshold,
            sharpness=cfg.occupancy_sharpness, epsilon=cfg.epsilon,
        )
        sensitivity = sensitivity_balanced_mse(
            true_density, prediction, self.sensitivity_weights,
            body_fraction=cfg.body_fraction, epsilon=cfg.epsilon,
        )
        observed_mgal = tf.cast(gravity_normalized * self.gravity_scale, tf.float32)
        predicted_mgal = self.forward_operator(prediction)
        gravity = data_weighted_gravity_loss(
            observed_mgal, predicted_mgal, self.data_weights, epsilon=cfg.epsilon
        )
        total = (
            cfg.lambda_shape * iou
            + cfg.lambda_sensitivity * sensitivity
            + cfg.lambda_physics * gravity
        )
        return prediction, iou, sensitivity, gravity, total

    def _update_metrics(
        self, truth: tf.Tensor, terms: tuple[tf.Tensor, ...]
    ) -> None:
        prediction, iou, sensitivity, gravity, total = terms
        self.total_tracker.update_state(total)
        self.iou_tracker.update_state(iou)
        self.sensitivity_tracker.update_state(sensitivity)
        self.gravity_tracker.update_state(gravity)
        squared = tf.square(prediction - truth)
        self.low_sensitivity_mse_tracker.update_state(
            tf.math.divide_no_nan(
                tf.reduce_sum(squared * self.low_sensitivity_mask),
                tf.reduce_sum(self.low_sensitivity_mask) * tf.cast(tf.shape(truth)[0], tf.float32),
            )
        )
        self.high_sensitivity_mse_tracker.update_state(
            tf.math.divide_no_nan(
                tf.reduce_sum(squared * self.high_sensitivity_mask),
                tf.reduce_sum(self.high_sensitivity_mask) * tf.cast(tf.shape(truth)[0], tf.float32),
            )
        )
        for metric in self.density_diagnostics:
            metric.update_state(truth, prediction)

    def train_step(self, data: Any) -> dict[str, tf.Tensor]:
        gravity, density, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            terms = self.compute_loss_terms(gravity, density, training=True)
        gradients = tape.gradient(terms[-1], self.inversion_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.inversion_model.trainable_variables))
        self._update_metrics(density, terms)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: Any) -> dict[str, tf.Tensor]:
        gravity, density, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        terms = self.compute_loss_terms(gravity, density, training=False)
        self._update_metrics(density, terms)
        return {metric.name: metric.result() for metric in self.metrics}
