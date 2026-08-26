"""Loss-only depth supervision for the unchanged E09 inversion model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from cnn_inversion_3d.diagnostics import build_prediction_diagnostics
from cnn_inversion_3d.model import BalancedDensityMSE
from cnn_inversion_3d.single_plane_review import SinglePlaneReviewConfig


@dataclass(frozen=True)
class E09ADepthLossConfig:
    lambda_density: float = 1.0
    lambda_depth: float = 1.0
    alpha_center: float = 1.0
    epsilon: float = 1.0e-8
    body_fraction: float = 0.5

    def validate(self) -> None:
        for name in ("lambda_density", "lambda_depth", "alpha_center"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must not be negative.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if not 0.0 < self.body_fraction < 1.0:
            raise ValueError("body_fraction must be between zero and one.")


def density_depth_profile(density: tf.Tensor) -> tf.Tensor:
    """Collapse canonical ``batch,z,y,x,channel`` density to ``batch,z``."""

    density = tf.convert_to_tensor(density)
    return tf.reduce_sum(density, axis=(2, 3, 4))


def normalized_depth_profile(density: tf.Tensor, *, epsilon: float = 1.0e-8) -> tf.Tensor:
    """Return a differentiable unit-sum density profile for each sample."""

    profile = density_depth_profile(density)
    denominator = tf.reduce_sum(profile, axis=1, keepdims=True)
    return profile / (denominator + tf.cast(epsilon, profile.dtype))


def depth_profile_mse(
    true_density: tf.Tensor, predicted_density: tf.Tensor, *, epsilon: float = 1.0e-8
) -> tf.Tensor:
    """Mean squared difference between normalized 24-bin depth profiles."""

    truth = normalized_depth_profile(true_density, epsilon=epsilon)
    prediction = normalized_depth_profile(predicted_density, epsilon=epsilon)
    return tf.reduce_mean(tf.square(prediction - truth))


def z_center_mse(
    true_density: tf.Tensor,
    predicted_density: tf.Tensor,
    *,
    epsilon: float = 1.0e-8,
    geometry: SinglePlaneReviewConfig | None = None,
) -> tf.Tensor:
    """Return squared normalized density-weighted center-depth error."""

    config = geometry or SinglePlaneReviewConfig()
    coordinates = tf.cast(
        config.density_z_min_center_m + tf.range(config.nz, dtype=tf.float32) * config.dz_m,
        predicted_density.dtype,
    )[None, :]
    true_profile = density_depth_profile(true_density)
    predicted_profile = density_depth_profile(predicted_density)

    def center(profile: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(profile * coordinates, axis=1) / (
            tf.reduce_sum(profile, axis=1) + tf.cast(epsilon, profile.dtype)
        )

    depth_span = tf.cast((config.nz - 1) * config.dz_m, predicted_density.dtype)
    return tf.reduce_mean(tf.square((center(predicted_profile) - center(true_profile)) / depth_span))


class E09ATrainingModel(tf.keras.Model):
    """Train an unchanged E09 model with additive depth supervision."""

    def __init__(self, inversion_model: tf.keras.Model, *, loss_config: E09ADepthLossConfig) -> None:
        super().__init__(name="e09a_depth_supervision_wrapper")
        loss_config.validate()
        self.inversion_model = inversion_model
        self.loss_config = loss_config
        self.density_loss_function = BalancedDensityMSE(body_fraction=loss_config.body_fraction)
        self.trackers = {
            name: tf.keras.metrics.Mean(name=name)
            for name in ("loss", "density_loss", "depth_profile_loss", "z_center_loss", "depth_loss")
        }
        self.density_diagnostics = build_prediction_diagnostics()

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [*self.trackers.values(), *self.density_diagnostics]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.inversion_model(inputs, training=training)

    def compute_loss_terms(
        self, gravity: tf.Tensor, truth: tf.Tensor, *, training: bool
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        prediction = self.inversion_model(gravity, training=training)
        cfg = self.loss_config
        density = self.density_loss_function(truth, prediction)
        profile = depth_profile_mse(truth, prediction, epsilon=cfg.epsilon)
        center = z_center_mse(truth, prediction, epsilon=cfg.epsilon)
        depth = profile + cfg.alpha_center * center
        total = cfg.lambda_density * density + cfg.lambda_depth * depth
        return prediction, density, profile, center, depth, total

    def _update(self, truth: tf.Tensor, terms: tuple[tf.Tensor, ...]) -> None:
        prediction, density, profile, center, depth, total = terms
        for name, value in (
            ("loss", total), ("density_loss", density), ("depth_profile_loss", profile),
            ("z_center_loss", center), ("depth_loss", depth),
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
