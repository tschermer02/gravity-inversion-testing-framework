"""Integrated-sensitivity compensation for the unchanged E09 architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from cnn_inversion_3d.diagnostics import build_prediction_diagnostics
from cnn_inversion_3d.e09a_training import depth_profile_mse, z_center_mse
from cnn_inversion_3d.e10_training import build_e10_sensitivity_weights
from cnn_inversion_3d.model import BalancedDensityMSE


@dataclass(frozen=True)
class E09BLossConfig:
    lambda_density: float = 1.0
    lambda_depth: float = 1.0
    alpha_center: float = 1.0
    lambda_sensitivity: float = 1.0
    sensitivity_gamma: float = 0.5
    sensitivity_weight_min: float = 0.5
    sensitivity_weight_max: float = 5.0
    epsilon: float = 1.0e-8
    body_fraction: float = 0.5

    def validate(self) -> None:
        for name in (
            "lambda_density", "lambda_depth", "alpha_center",
            "lambda_sensitivity", "sensitivity_gamma",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must not be negative.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if not 0.0 < self.sensitivity_weight_min <= 1.0:
            raise ValueError("sensitivity_weight_min must be in (0, 1].")
        if self.sensitivity_weight_max < 1.0:
            raise ValueError("sensitivity_weight_max must be at least one.")
        if self.sensitivity_weight_min > self.sensitivity_weight_max:
            raise ValueError("sensitivity weight minimum exceeds maximum.")
        if not 0.0 < self.body_fraction < 1.0:
            raise ValueError("body_fraction must be between zero and one.")


def _bounded_mean_one(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    """Scale positive values so clipped output has mean one and fixed bounds."""

    low, high = 1.0e-12, float(np.max(values)) * 1.0e12
    for _ in range(100):
        scale = np.sqrt(low * high)
        mean = float(np.mean(np.clip(values / scale, minimum, maximum)))
        if mean > 1.0:
            low = scale
        else:
            high = scale
    return np.clip(values / np.sqrt(low * high), minimum, maximum)


def build_e09b_sensitivity_weights(
    *, gamma: float = 0.5, epsilon: float = 1.0e-8,
    weight_min: float = 0.5, weight_max: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical integrated sensitivity and bounded mean-one compensation."""

    sensitivity, _, _ = build_e10_sensitivity_weights(gamma=0.0, epsilon=epsilon)
    raw = np.power(float(np.max(sensitivity)) / (sensitivity.astype(np.float64) + epsilon), gamma)
    weights = _bounded_mean_one(raw, weight_min, weight_max)
    if not np.all(np.isfinite(weights)) or not np.isclose(np.mean(weights), 1.0, atol=1.0e-6):
        raise ValueError("E09B sensitivity weights are not finite and mean-one.")
    if float(np.min(weights)) < weight_min - 1.0e-6 or float(np.max(weights)) > weight_max + 1.0e-6:
        raise ValueError("E09B sensitivity weights violate configured clipping bounds.")
    return sensitivity.astype(np.float32), weights.astype(np.float32)


def integrated_sensitivity_compensated_mse(
    true_density: tf.Tensor, predicted_density: tf.Tensor, weights: tf.Tensor,
) -> tf.Tensor:
    """Per-sample weighted voxel MSE, then averaged across the batch."""

    weight_volume = tf.cast(weights, predicted_density.dtype)[None, ..., None]
    numerator = tf.reduce_sum(
        weight_volume * tf.square(predicted_density - true_density), axis=(1, 2, 3, 4)
    )
    denominator = tf.reduce_sum(weight_volume, axis=(1, 2, 3, 4))
    return tf.reduce_mean(tf.math.divide_no_nan(numerator, denominator))


class E09BTrainingModel(tf.keras.Model):
    """Train unchanged E09 with corrected E09A plus sensitivity compensation."""

    def __init__(
        self, inversion_model: tf.keras.Model, sensitivity_weights: np.ndarray,
        *, loss_config: E09BLossConfig,
    ) -> None:
        super().__init__(name="e09b_integrated_sensitivity_wrapper")
        loss_config.validate()
        self.inversion_model = inversion_model
        self.loss_config = loss_config
        self.sensitivity_weights = tf.constant(sensitivity_weights, tf.float32)
        self.density_loss_function = BalancedDensityMSE(body_fraction=loss_config.body_fraction)
        self.trackers = {
            name: tf.keras.metrics.Mean(name=name)
            for name in (
                "loss", "density_loss", "depth_profile_loss", "z_center_loss",
                "depth_loss", "sensitivity_loss",
            )
        }
        self.density_diagnostics = build_prediction_diagnostics()

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [*self.trackers.values(), *self.density_diagnostics]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.inversion_model(inputs, training=training)

    def compute_loss_terms(self, gravity: tf.Tensor, truth: tf.Tensor, *, training: bool) -> tuple[tf.Tensor, ...]:
        prediction = self.inversion_model(gravity, training=training)
        cfg = self.loss_config
        density = self.density_loss_function(truth, prediction)
        profile = depth_profile_mse(truth, prediction, epsilon=cfg.epsilon)
        center = z_center_mse(truth, prediction, epsilon=cfg.epsilon)
        depth = profile + cfg.alpha_center * center
        sensitivity = integrated_sensitivity_compensated_mse(
            truth, prediction, self.sensitivity_weights
        )
        total = (
            cfg.lambda_density * density + cfg.lambda_depth * depth
            + cfg.lambda_sensitivity * sensitivity
        )
        return prediction, density, profile, center, depth, sensitivity, total

    def _update(self, truth: tf.Tensor, terms: tuple[tf.Tensor, ...]) -> None:
        prediction, density, profile, center, depth, sensitivity, total = terms
        for name, value in (
            ("loss", total), ("density_loss", density), ("depth_profile_loss", profile),
            ("z_center_loss", center), ("depth_loss", depth),
            ("sensitivity_loss", sensitivity),
        ):
            self.trackers[name].update_state(value)
        for metric in self.density_diagnostics:
            metric.update_state(truth, prediction)

    def train_step(self, data: Any) -> dict[str, tf.Tensor]:
        gravity, truth, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            terms = self.compute_loss_terms(gravity, truth, training=True)
        gradients = tape.gradient(terms[-1], self.inversion_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.inversion_model.trainable_variables))
        self._update(truth, terms)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: Any) -> dict[str, tf.Tensor]:
        gravity, truth, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        terms = self.compute_loss_terms(gravity, truth, training=False)
        self._update(truth, terms)
        return {metric.name: metric.result() for metric in self.metrics}
