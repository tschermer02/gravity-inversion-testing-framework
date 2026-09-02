"""Body-density and physics ablations layered onto unchanged E09B-6-prime."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from cnn_inversion_3d.differentiable_gravity import (
    DifferentiableSinglePlaneGz, global_normalized_gravity_mse,
)
from cnn_inversion_3d.e09b_training import E09BLossConfig, E09BTrainingModel


@dataclass(frozen=True)
class E09B911LossConfig(E09BLossConfig):
    lambda_body_density: float = 0.0
    lambda_gravity: float = 0.0

    def validate(self) -> None:
        super().validate()
        if self.lambda_body_density < 0.0 or self.lambda_gravity < 0.0:
            raise ValueError("E09B-9/10/11 loss weights must not be negative.")


def body_density_mse_per_sample(
    truth: tf.Tensor, prediction: tf.Tensor, *, epsilon: float = 1.0e-8
) -> tf.Tensor:
    """Voxel-density MSE restricted to each sample's true body support."""

    mask = tf.cast(truth > 0.0, prediction.dtype)
    axes = (1, 2, 3, 4)
    return tf.math.divide_no_nan(
        tf.reduce_sum(mask * tf.square(prediction - truth), axis=axes),
        tf.reduce_sum(mask, axis=axes) + tf.cast(epsilon, prediction.dtype),
    )


class E09B911TrainingModel(E09BTrainingModel):
    """Add optional body-only density and fixed forward-gravity objectives."""

    def __init__(self, inversion_model: tf.keras.Model, sensitivity_weights: np.ndarray,
                 forward_operator: DifferentiableSinglePlaneGz, *, gravity_scale: float,
                 loss_config: E09B911LossConfig) -> None:
        loss_config.validate()
        super().__init__(inversion_model, sensitivity_weights, loss_config=loss_config)
        self._name = "e09b_density_physics_ablation_wrapper"
        self.forward_operator = forward_operator
        self.gravity_scale = float(gravity_scale)
        if self.gravity_scale <= 0.0: raise ValueError("gravity_scale must be positive.")
        tracker_names = ["body_density_loss"]
        if loss_config.lambda_gravity > 0.0:
            tracker_names.extend(("gravity_loss", "weighted_gravity_loss",
                                  "gravity_rmse", "gravity_correlation"))
        for name in tracker_names:
            self.trackers[name] = tf.keras.metrics.Mean(name=name)

    def compute_loss_terms(self, gravity: tf.Tensor, truth: tf.Tensor, *, training: bool) -> tuple[tf.Tensor, ...]:
        base = super().compute_loss_terms(gravity, truth, training=training)
        prediction = base[0]; cfg = self.loss_config
        body_density = tf.reduce_mean(body_density_mse_per_sample(
            truth, prediction, epsilon=cfg.epsilon))
        if cfg.lambda_gravity > 0.0:
            true_gravity = tf.cast(gravity * self.gravity_scale, tf.float32)
            predicted_gravity = self.forward_operator(prediction)
            gravity_loss = global_normalized_gravity_mse(
                true_gravity, predicted_gravity, gravity_scale=self.gravity_scale)
        else:
            gravity_loss = tf.zeros((), dtype=prediction.dtype)
        weighted_gravity = cfg.lambda_gravity * gravity_loss
        total = base[-1] + cfg.lambda_body_density * body_density + weighted_gravity
        return (*base[:-1], body_density, gravity_loss, weighted_gravity, total)

    def _update_extended(self, gravity: tf.Tensor, truth: tf.Tensor,
                         terms: tuple[tf.Tensor, ...]) -> None:
        super()._update(truth, (*terms[:7], terms[-1]))
        body_density, gravity_loss, weighted_gravity = terms[7:10]
        self.trackers["body_density_loss"].update_state(body_density)
        if self.loss_config.lambda_gravity <= 0.0:
            return
        prediction = terms[0]
        true_gravity = tf.cast(gravity * self.gravity_scale, tf.float32)
        predicted_gravity = self.forward_operator(prediction)
        residual = predicted_gravity - true_gravity
        true_flat = tf.reshape(true_gravity, (tf.shape(true_gravity)[0], -1))
        predicted_flat = tf.reshape(predicted_gravity, (tf.shape(predicted_gravity)[0], -1))
        true_centered = true_flat - tf.reduce_mean(true_flat, axis=1, keepdims=True)
        predicted_centered = predicted_flat - tf.reduce_mean(predicted_flat, axis=1, keepdims=True)
        correlation = tf.math.divide_no_nan(
            tf.reduce_sum(true_centered * predicted_centered, axis=1),
            tf.norm(true_centered, axis=1) * tf.norm(predicted_centered, axis=1),
        )
        for name, value in (("gravity_loss", gravity_loss),
                            ("weighted_gravity_loss", weighted_gravity),
                            ("gravity_rmse", tf.sqrt(tf.reduce_mean(tf.square(residual)))),
                            ("gravity_correlation", tf.reduce_mean(correlation))):
            self.trackers[name].update_state(value)

    def train_step(self, data: Any) -> dict[str, tf.Tensor]:
        gravity, truth, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            terms = self.compute_loss_terms(gravity, truth, training=True)
        gradients = tape.gradient(terms[-1], self.inversion_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.inversion_model.trainable_variables))
        self._update_extended(gravity, truth, terms)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: Any) -> dict[str, tf.Tensor]:
        gravity, truth, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        terms = self.compute_loss_terms(gravity, truth, training=False)
        self._update_extended(gravity, truth, terms)
        return {metric.name: metric.result() for metric in self.metrics}
