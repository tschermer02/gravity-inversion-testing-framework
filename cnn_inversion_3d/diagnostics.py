from __future__ import annotations

from typing import Any, Literal

import tensorflow as tf


class PredictionExtremum(tf.keras.metrics.Metric):
    """
    Track the minimum or maximum prediction over a complete epoch.

    Parameters
    ----------
    mode
        Extremum to track.
    name
        Metric name reported by Keras.
    """

    def __init__(
        self,
        *,
        mode: Literal["minimum", "maximum"],
        name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            **kwargs,
        )

        if mode not in {
            "minimum",
            "maximum",
        }:
            raise ValueError(
                "mode must be either 'minimum' or 'maximum'."
            )

        self.mode = mode
        initial_value = (
            float("inf")
            if mode == "minimum"
            else float("-inf")
        )
        self.extremum = self.add_weight(
            name="extremum",
            initializer=tf.keras.initializers.Constant(
                initial_value
            ),
        )

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Update the epoch extremum."""

        del y_true, sample_weight
        batch_extremum = (
            tf.reduce_min(y_pred)
            if self.mode == "minimum"
            else tf.reduce_max(y_pred)
        )

        if self.mode == "minimum":
            self.extremum.assign(
                tf.minimum(
                    self.extremum,
                    batch_extremum,
                )
            )
        else:
            self.extremum.assign(
                tf.maximum(
                    self.extremum,
                    batch_extremum,
                )
            )

    def result(self) -> tf.Tensor:
        """Return the epoch extremum."""

        return self.extremum

    def get_config(self) -> dict[str, Any]:
        """Return the serializable metric configuration."""

        config = super().get_config()
        config.update(
            {
                "mode": self.mode,
            }
        )
        return config


class PredictionMoments(tf.keras.metrics.Metric):
    """
    Track the global mean or standard deviation of predictions.

    Parameters
    ----------
    statistic
        Moment to report.
    name
        Metric name reported by Keras.
    """

    def __init__(
        self,
        *,
        statistic: Literal["mean", "standard_deviation"],
        name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            **kwargs,
        )

        if statistic not in {
            "mean",
            "standard_deviation",
        }:
            raise ValueError(
                "statistic must be either 'mean' or "
                "'standard_deviation'."
            )

        self.statistic = statistic
        self.value_sum = self.add_weight(
            name="value_sum",
            initializer="zeros",
        )
        self.square_sum = self.add_weight(
            name="square_sum",
            initializer="zeros",
        )
        self.value_count = self.add_weight(
            name="value_count",
            initializer="zeros",
        )

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Accumulate prediction moments."""

        del y_true, sample_weight
        values = tf.cast(
            y_pred,
            self.dtype,
        )
        self.value_sum.assign_add(
            tf.reduce_sum(values)
        )
        self.square_sum.assign_add(
            tf.reduce_sum(
                tf.square(values)
            )
        )
        self.value_count.assign_add(
            tf.cast(
                tf.size(values),
                self.dtype,
            )
        )

    def result(self) -> tf.Tensor:
        """Return the requested prediction moment."""

        mean = tf.math.divide_no_nan(
            self.value_sum,
            self.value_count,
        )

        if self.statistic == "mean":
            return mean

        mean_square = tf.math.divide_no_nan(
            self.square_sum,
            self.value_count,
        )
        variance = tf.maximum(
            mean_square - tf.square(mean),
            tf.cast(
                0.0,
                self.dtype,
            ),
        )
        return tf.sqrt(variance)

    def get_config(self) -> dict[str, Any]:
        """Return the serializable metric configuration."""

        config = super().get_config()
        config.update(
            {
                "statistic": self.statistic,
            }
        )
        return config


class PredictionAboveThreshold(tf.keras.metrics.Metric):
    """
    Track the fraction of predicted voxels above a threshold.

    Parameters
    ----------
    threshold
        Strict lower bound used to count a predicted voxel.
    name
        Metric name reported by Keras.
    """

    def __init__(
        self,
        *,
        threshold: float = 1.0e-4,
        name: str = "prediction_fraction_above_1e_4",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            **kwargs,
        )

        if threshold < 0.0:
            raise ValueError(
                "threshold must not be negative."
            )

        self.threshold = float(
            threshold
        )
        self.above_count = self.add_weight(
            name="above_count",
            initializer="zeros",
        )
        self.value_count = self.add_weight(
            name="value_count",
            initializer="zeros",
        )

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Accumulate threshold counts."""

        del y_true, sample_weight
        above = tf.cast(
            y_pred > self.threshold,
            self.dtype,
        )
        self.above_count.assign_add(
            tf.reduce_sum(above)
        )
        self.value_count.assign_add(
            tf.cast(
                tf.size(y_pred),
                self.dtype,
            )
        )

    def result(self) -> tf.Tensor:
        """Return the above-threshold fraction."""

        return tf.math.divide_no_nan(
            self.above_count,
            self.value_count,
        )

    def get_config(self) -> dict[str, Any]:
        """Return the serializable metric configuration."""

        config = super().get_config()
        config.update(
            {
                "threshold": self.threshold,
            }
        )
        return config


class MaskedPredictionMean(tf.keras.metrics.Metric):
    """
    Track mean prediction in true body or background voxels.

    Parameters
    ----------
    region
        True-density region over which to average predictions.
    name
        Metric name reported by Keras.
    """

    def __init__(
        self,
        *,
        region: Literal["body", "background"],
        name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            **kwargs,
        )

        if region not in {
            "body",
            "background",
        }:
            raise ValueError(
                "region must be either 'body' or 'background'."
            )

        self.region = region
        self.prediction_sum = self.add_weight(
            name="prediction_sum",
            initializer="zeros",
        )
        self.voxel_count = self.add_weight(
            name="voxel_count",
            initializer="zeros",
        )

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Accumulate masked predictions."""

        del sample_weight
        body_mask = tf.cast(
            y_true > 0.0,
            self.dtype,
        )
        mask = (
            body_mask
            if self.region == "body"
            else 1.0 - body_mask
        )
        prediction = tf.cast(
            y_pred,
            self.dtype,
        )
        self.prediction_sum.assign_add(
            tf.reduce_sum(
                prediction * mask
            )
        )
        self.voxel_count.assign_add(
            tf.reduce_sum(mask)
        )

    def result(self) -> tf.Tensor:
        """Return the masked mean prediction."""

        return tf.math.divide_no_nan(
            self.prediction_sum,
            self.voxel_count,
        )

    def get_config(self) -> dict[str, Any]:
        """Return the serializable metric configuration."""

        config = super().get_config()
        config.update(
            {
                "region": self.region,
            }
        )
        return config


class BalancedMSEComponent(tf.keras.metrics.Metric):
    """
    Track an unweighted body or background MSE component.

    This diagnostic follows ``BalancedDensityMSE`` by averaging masked
    error within each sample and then averaging samples.

    Parameters
    ----------
    region
        Error component to report.
    name
        Metric name reported by Keras.
    """

    def __init__(
        self,
        *,
        region: Literal["body", "background"],
        name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            **kwargs,
        )

        if region not in {
            "body",
            "background",
        }:
            raise ValueError(
                "region must be either 'body' or 'background'."
            )

        self.region = region
        self.error_sum = self.add_weight(
            name="error_sum",
            initializer="zeros",
        )
        self.sample_count = self.add_weight(
            name="sample_count",
            initializer="zeros",
        )

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Accumulate masked squared error."""

        del sample_weight
        true_density = tf.cast(
            y_true,
            self.dtype,
        )
        predicted_density = tf.cast(
            y_pred,
            self.dtype,
        )
        body_mask = tf.cast(
            true_density > 0.0,
            self.dtype,
        )
        mask = (
            body_mask
            if self.region == "body"
            else 1.0 - body_mask
        )
        sample_error_sum = tf.reduce_sum(
            tf.square(
                predicted_density
                - true_density
            )
            * mask,
            axis=(1, 2, 3, 4),
        )
        sample_voxel_count = tf.reduce_sum(
            mask,
            axis=(1, 2, 3, 4),
        )
        sample_mse = tf.math.divide_no_nan(
            sample_error_sum,
            sample_voxel_count,
        )
        self.error_sum.assign_add(
            tf.reduce_sum(
                sample_mse
            )
        )
        self.sample_count.assign_add(
            tf.cast(
                tf.shape(sample_mse)[0],
                self.dtype,
            )
        )

    def result(self) -> tf.Tensor:
        """Return the masked MSE component."""

        return tf.math.divide_no_nan(
            self.error_sum,
            self.sample_count,
        )

    def get_config(self) -> dict[str, Any]:
        """Return the serializable metric configuration."""

        config = super().get_config()
        config.update(
            {
                "region": self.region,
            }
        )
        return config


class CollapseDetectionCallback(tf.keras.callbacks.Callback):
    """
    Detect consecutive epochs with near-zero validation predictions.

    Parameters
    ----------
    threshold
        Retained for backward-compatible configuration reporting. Collapse is
        detected from the robust multi-metric criterion documented below.
    patience
        Number of consecutive collapsed epochs required for detection.
    stop_training
        Whether detection should stop model fitting.
        monitor
        Retained for backward-compatible result metadata.
    """

    def __init__(
        self,
        *,
        threshold: float = 1.0e-5,
        patience: int = 2,
        stop_training: bool = False,
        monitor: str = "val_prediction_maximum",
    ) -> None:
        super().__init__()

        if threshold < 0.0:
            raise ValueError(
                "threshold must not be negative."
            )

        if patience < 1:
            raise ValueError(
                "patience must be at least one."
            )

        self.threshold = float(
            threshold
        )
        self.patience = int(
            patience
        )
        self.stop_training = bool(
            stop_training
        )
        self.monitor = monitor
        self.consecutive_epochs = 0
        self.collapse_epoch: int | None = None
        self.collapse_logs: dict[str, float] | None = None

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        """Inspect validation diagnostics after an epoch."""

        logs = logs or {}
        required = (
            "val_body_prediction_mean",
            "val_prediction_fraction_above_1e_4",
            "val_prediction_mean",
            "val_prediction_maximum",
        )
        if not any(key in logs for key in required):
            return
        body_mean = float(logs.get(required[0], float("inf")))
        occupied_fraction = float(logs.get(required[1], float("inf")))
        prediction_mean = float(logs.get(required[2], float("inf")))
        prediction_maximum = float(logs.get(required[3], float("inf")))
        collapsed = (
            body_mean < 1.0e-4
            or occupied_fraction < 1.0e-4
            or (prediction_mean < 1.0e-6 and prediction_maximum < 1.0e-2)
        )

        if collapsed:
            self.consecutive_epochs += 1
        else:
            self.consecutive_epochs = 0

        if (
            self.consecutive_epochs < self.patience
            or self.collapse_epoch is not None
        ):
            return

        self.collapse_epoch = epoch + 1
        self.collapse_logs = {
            key: float(metric_value)
            for key, metric_value in logs.items()
            if isinstance(
                metric_value,
                int | float,
            )
        }

        print(
            "\nWARNING: likely all-zero prediction collapse detected "
            f"at epoch {self.collapse_epoch}: "
            "the robust validation criterion remained true for "
            f"{self.patience} consecutive epochs."
        )

        if self.stop_training:
            self.model.stop_training = True

    def result(self) -> dict[str, Any]:
        """
        Return serializable collapse-detection results.

        Returns
        -------
        dict
            Callback configuration and any detected collapse event.
        """

        return {
            "monitor": self.monitor,
            "threshold": self.threshold,
            "criterion": (
                "val_body_prediction_mean < 1e-4 OR "
                "val_prediction_fraction_above_1e_4 < 1e-4 OR "
                "(val_prediction_mean < 1e-6 AND "
                "val_prediction_maximum < 1e-2)"
            ),
            "patience": self.patience,
            "stop_training": self.stop_training,
            "detected": (
                self.collapse_epoch is not None
            ),
            "collapse_epoch": self.collapse_epoch,
            "diagnostics": self.collapse_logs,
        }


def build_prediction_diagnostics(
    *,
    threshold: float = 1.0e-4,
) -> list[tf.keras.metrics.Metric]:
    """
    Build reusable prediction-collapse diagnostic metrics.

    Parameters
    ----------
    threshold
        Threshold for the predicted-voxel fraction metric.

    Returns
    -------
    list of tensorflow.keras.metrics.Metric
        Metrics suitable for ``Model.compile``.
    """

    return [
        PredictionExtremum(
            mode="minimum",
            name="prediction_minimum",
        ),
        PredictionExtremum(
            mode="maximum",
            name="prediction_maximum",
        ),
        PredictionMoments(
            statistic="mean",
            name="prediction_mean",
        ),
        PredictionMoments(
            statistic="standard_deviation",
            name="prediction_standard_deviation",
        ),
        PredictionAboveThreshold(
            threshold=threshold,
        ),
        MaskedPredictionMean(
            region="body",
            name="body_prediction_mean",
        ),
        MaskedPredictionMean(
            region="background",
            name="background_prediction_mean",
        ),
        BalancedMSEComponent(
            region="body",
            name="body_mse_component",
        ),
        BalancedMSEComponent(
            region="background",
            name="background_mse_component",
        ),
    ]
