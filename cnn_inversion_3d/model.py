from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from cnn_inversion_3d.dataset import (
    DENSITY_SHAPE,
    GRAVITY_SHAPE,
    SINGLE_PLANE_GRAVITY_SHAPE,
)
from cnn_inversion_3d.diagnostics import (
    build_prediction_diagnostics,
)

from typing import Literal

VerticalExpansion = Literal[
    "repeat",
    "transpose",
]


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration for the baseline 3D gravity-inversion CNN.

    The baseline model intentionally uses:

    - no explicit regularization
    - no dropout
    - no physics loss
    - no smoothness loss
    - one positive density output channel
    """

    base_filters: int = 8
    learning_rate: float = 1.0e-3
    output_activation: str = "sigmoid"
    body_loss_fraction: float = 0.5
    vertical_expansion: VerticalExpansion = "repeat"

    def validate(self) -> None:
        """Validate model parameters."""

        if self.base_filters < 1:
            raise ValueError(
                "base_filters must be at least one."
            )

        if self.learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be greater than zero."
            )

        if self.output_activation not in {
            "sigmoid",
            "relu",
            "linear",
        }:
            raise ValueError(
                "output_activation must be one of: "
                "'sigmoid', 'relu', or 'linear'."
            )
        if not 0.0 < self.body_loss_fraction < 1.0:
            raise ValueError(
                "body_loss_fraction must be between zero and one."
            )

        if self.vertical_expansion not in {
            "repeat",
            "transpose",
        }:
            raise ValueError(
                "vertical_expansion must be either "
                "'repeat' or 'transpose'."
            )

def convolution_block(
    inputs: tf.Tensor,
    *,
    filters: int,
    name: str,
) -> tf.Tensor:
    """
    Apply two same-padded 3D convolution layers.

    Parameters
    ----------
    inputs
        Input feature tensor.
    filters
        Number of filters in both convolution layers.
    name
        Layer-name prefix.

    Returns
    -------
    tensorflow.Tensor
        Output feature tensor.
    """

    x = tf.keras.layers.Conv3D(
        filters=filters,
        kernel_size=(3, 3, 3),
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name=f"{name}_conv_1",
    )(
        inputs
    )

    x = tf.keras.layers.Conv3D(
        filters=filters,
        kernel_size=(3, 3, 3),
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name=f"{name}_conv_2",
    )(
        x
    )

    return x


def build_baseline_model(
    config: ModelConfig | None = None,
) -> tf.keras.Model:
    """
    Build the baseline 3D gravity-to-density CNN.

    Input shape
    -----------
    ``(8, 64, 64, 1)``

    Output shape
    ------------
    ``(24, 64, 64, 1)``

    Architecture
    ------------
    The input is downsampled twice:

    ``8 x 64 x 64``
        -> ``4 x 32 x 32``
        -> ``2 x 16 x 16``

    It is then upsampled back to:

    ``8 x 64 x 64``

    Finally, the receiver-depth dimension is expanded by a factor of
    three:

    ``8 x 64 x 64``
        -> ``24 x 64 x 64``

    Parameters
    ----------
    config
        Optional model configuration.

    Returns
    -------
    tensorflow.keras.Model
        Uncompiled Keras model.
    """

    if config is None:
        config = ModelConfig()

    config.validate()

    base_filters = config.base_filters

    inputs = tf.keras.Input(
        shape=GRAVITY_SHAPE,
        name="gravity_volume",
    )

    encoder_1 = convolution_block(
        inputs,
        filters=base_filters,
        name="encoder_1",
    )

    pooled_1 = tf.keras.layers.MaxPool3D(
        pool_size=(2, 2, 2),
        name="pool_1",
    )(
        encoder_1
    )

    encoder_2 = convolution_block(
        pooled_1,
        filters=base_filters * 2,
        name="encoder_2",
    )

    pooled_2 = tf.keras.layers.MaxPool3D(
        pool_size=(2, 2, 2),
        name="pool_2",
    )(
        encoder_2
    )

    bottleneck = convolution_block(
        pooled_2,
        filters=base_filters * 4,
        name="bottleneck",
    )

    upsample_1 = tf.keras.layers.UpSampling3D(
        size=(2, 2, 2),
        name="upsample_1",
    )(
        bottleneck
    )

    upsample_1 = tf.keras.layers.Conv3D(
        filters=base_filters * 2,
        kernel_size=(2, 2, 2),
        padding="same",
        activation="relu",
        name="upsample_1_projection",
    )(
        upsample_1
    )

    merged_1 = tf.keras.layers.Concatenate(
        axis=-1,
        name="skip_connection_1",
    )(
        [
            upsample_1,
            encoder_2,
        ]
    )

    decoder_1 = convolution_block(
        merged_1,
        filters=base_filters * 2,
        name="decoder_1",
    )

    upsample_2 = tf.keras.layers.UpSampling3D(
        size=(2, 2, 2),
        name="upsample_2",
    )(
        decoder_1
    )

    upsample_2 = tf.keras.layers.Conv3D(
        filters=base_filters,
        kernel_size=(2, 2, 2),
        padding="same",
        activation="relu",
        name="upsample_2_projection",
    )(
        upsample_2
    )

    merged_2 = tf.keras.layers.Concatenate(
        axis=-1,
        name="skip_connection_2",
    )(
        [
            upsample_2,
            encoder_1,
        ]
    )

    decoder_2 = convolution_block(
        merged_2,
        filters=base_filters,
        name="decoder_2",
    )

    if config.vertical_expansion == "repeat":
        depth_expansion = tf.keras.layers.UpSampling3D(
            size=(3, 1, 1),
            name="expand_receiver_depth_to_model_depth",
        )(
            decoder_2
        )
    elif config.vertical_expansion == "transpose":
        depth_expansion = tf.keras.layers.Conv3DTranspose(
            filters=base_filters,
            kernel_size=(3, 3, 3),
            strides=(3, 1, 1),
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name="expand_receiver_depth_to_model_depth",
        )(
            decoder_2
        )
    else:
        raise ValueError(
            "Unsupported vertical expansion method: "
            f"{config.vertical_expansion!r}."
        )

    refined = convolution_block(
        depth_expansion,
        filters=base_filters,
        name="density_refinement",
    )

    outputs = tf.keras.layers.Conv3D(
        filters=1,
        kernel_size=(1, 1, 1),
        padding="same",
        activation=config.output_activation,
        name="recovered_density",
    )(
        refined
    )

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="baseline_fwd3d_inversion_cnn",
    )

    expected_output_shape = (
        None,
        *DENSITY_SHAPE,
    )

    if model.output_shape != expected_output_shape:
        raise RuntimeError(
            "Model output shape is incorrect. "
            f"Expected {expected_output_shape}, "
            f"received {model.output_shape}."
        )

    return model


def _convolution_2d_block(
    inputs: tf.Tensor,
    *,
    filters: int,
    name: str,
) -> tf.Tensor:
    """Apply two same-padded 2D convolutions for surface encoding."""

    x = inputs
    for index in (1, 2):
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name=f"{name}_conv_{index}",
        )(x)
    return x


def build_single_plane_model(
    config: ModelConfig | None = None,
) -> tf.keras.Model:
    """
    Build the additive 2D-surface-encoder to 3D-density-decoder CNN.

    The complete configured 81 x 81 surface is linearly resampled to a
    64 x 64 lateral feature grid, encoded to 16 x 16, decoded to 64 x 64,
    expanded to six learned-depth feature planes, and transposed-convolved
    in depth from 6 to 12 to 24 cells.
    """

    if config is None:
        config = ModelConfig(vertical_expansion="transpose")
    config.validate()
    filters = config.base_filters
    inputs = tf.keras.Input(
        shape=SINGLE_PLANE_GRAVITY_SHAPE,
        name="surface_gravity_gz",
    )
    lateral_grid = tf.keras.layers.Resizing(
        64,
        64,
        interpolation="bilinear",
        name="resample_complete_observation_plane_to_lateral_grid",
    )(inputs)
    encoder_1 = _convolution_2d_block(
        lateral_grid, filters=filters, name="surface_encoder_1"
    )
    pooled_1 = tf.keras.layers.MaxPool2D(2, name="surface_pool_1")(
        encoder_1
    )
    encoder_2 = _convolution_2d_block(
        pooled_1, filters=filters * 2, name="surface_encoder_2"
    )
    pooled_2 = tf.keras.layers.MaxPool2D(2, name="surface_pool_2")(
        encoder_2
    )
    bottleneck = _convolution_2d_block(
        pooled_2, filters=filters * 4, name="surface_bottleneck"
    )
    up_1 = tf.keras.layers.UpSampling2D(2, name="surface_upsample_1")(
        bottleneck
    )
    up_1 = tf.keras.layers.Conv2D(
        filters * 2, 2, padding="same", activation="relu"
    )(up_1)
    decoded_1 = _convolution_2d_block(
        tf.keras.layers.Concatenate()([up_1, encoder_2]),
        filters=filters * 2,
        name="surface_decoder_1",
    )
    up_2 = tf.keras.layers.UpSampling2D(2, name="surface_upsample_2")(
        decoded_1
    )
    up_2 = tf.keras.layers.Conv2D(
        filters, 2, padding="same", activation="relu"
    )(up_2)
    decoded_2 = _convolution_2d_block(
        tf.keras.layers.Concatenate()([up_2, encoder_1]),
        filters=filters,
        name="surface_decoder_2",
    )
    depth_seed = tf.keras.layers.Reshape(
        (1, 64, 64, filters), name="surface_to_depth_seed"
    )(decoded_2)
    depth_seed = tf.keras.layers.UpSampling3D(
        size=(6, 1, 1), name="seed_six_depth_planes"
    )(depth_seed)
    depth_12 = tf.keras.layers.Conv3DTranspose(
        filters=filters,
        kernel_size=3,
        strides=(2, 1, 1),
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="decode_depth_6_to_12",
    )(depth_seed)
    depth_24 = tf.keras.layers.Conv3DTranspose(
        filters=filters,
        kernel_size=3,
        strides=(2, 1, 1),
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="decode_depth_12_to_24",
    )(depth_12)
    refined = convolution_block(
        depth_24, filters=filters, name="single_plane_density_refinement"
    )
    outputs = tf.keras.layers.Conv3D(
        1,
        1,
        padding="same",
        activation=config.output_activation,
        name="recovered_density",
    )(refined)
    model = tf.keras.Model(inputs, outputs, name="single_plane_inversion_cnn")
    if model.output_shape != (None, *DENSITY_SHAPE):
        raise RuntimeError(f"Unexpected single-plane output: {model.output_shape}")
    return model


@tf.keras.utils.register_keras_serializable(
    package="gravity_inversion"
)
class BalancedDensityMSE(tf.keras.losses.Loss):
    """
    Balanced mean-squared-error loss for sparse density models.
    """

    body_fraction: float

    def __init__(
        self,
        *,
        body_fraction: float = 0.5,
        name: str = "balanced_density_mse",
    ) -> None:
        super().__init__(
            name=name,
            reduction="sum_over_batch_size",
        )

        if not 0.0 < body_fraction < 1.0:
            raise ValueError(
                "body_fraction must be between zero and one."
            )

        self.body_fraction = float(
            body_fraction
        )

    def call(
        self,
        y_true,
        y_pred,
    ):
        """
        Calculate equally controlled body and background errors.
        """

        true_density = tf.cast(
            y_true,
            tf.float32,
        )

        predicted_density = tf.cast(
            y_pred,
            tf.float32,
        )

        squared_error = tf.square(
            predicted_density
            - true_density
        )

        body_mask = tf.cast(
            true_density > 0.0,
            tf.float32,
        )

        background_mask = (
            1.0
            - body_mask
        )

        body_error_sum = tf.reduce_sum(
            squared_error
            * body_mask,
            axis=(1, 2, 3, 4),
        )

        body_cell_count = tf.reduce_sum(
            body_mask,
            axis=(1, 2, 3, 4),
        )

        background_error_sum = tf.reduce_sum(
            squared_error
            * background_mask,
            axis=(1, 2, 3, 4),
        )

        background_cell_count = tf.reduce_sum(
            background_mask,
            axis=(1, 2, 3, 4),
        )

        body_mse = tf.math.divide_no_nan(
            body_error_sum,
            body_cell_count,
        )

        background_mse = tf.math.divide_no_nan(
            background_error_sum,
            background_cell_count,
        )

        background_fraction = (
            1.0
            - self.body_fraction
        )

        sample_loss = (
            self.body_fraction
            * body_mse
            + background_fraction
            * background_mse
        )

        return tf.reduce_mean(
            sample_loss
        )

    def get_config(
        self,
    ) -> dict[str, float | str]:
        """
        Return the serializable loss configuration.
        """

        config = super().get_config()

        config.update(
            {
                "body_fraction": (
                    self.body_fraction
                ),
            }
        )

        return config


def compile_baseline_model(
    model: tf.keras.Model,
    config: ModelConfig | None = None,
) -> None:
    """
    Compile the baseline model.

    Training uses balanced body/background mean squared error with no
    explicit regularization. Ordinary mean squared error and mean
    absolute error are included only as reporting metrics.

    Parameters
    ----------
    model
        Model returned by ``build_baseline_model``.
    config
        Optional model configuration.
    """

    if config is None:
        config = ModelConfig()

    config.validate()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate,
    )

    balanced_loss = BalancedDensityMSE(
        body_fraction=(
            config.body_loss_fraction
        ),
    )

    model.compile(
        optimizer=optimizer,
        loss=balanced_loss,
        metrics=[
            tf.keras.metrics.MeanSquaredError(
                name="density_mse"
            ),
            tf.keras.metrics.MeanAbsoluteError(
                name="density_mae"
            ),
            *build_prediction_diagnostics(),
        ],
    )

def count_trainable_parameters(
    model: tf.keras.Model,
) -> int:
    """
    Return the number of trainable model parameters.
    """

    return int(
        sum(
            variable.shape.num_elements()
            for variable in model.trainable_weights
        )
    )


def main() -> None:
    """
    Build, compile, and test one forward pass.
    """

    config = ModelConfig(
        base_filters=8,
        learning_rate=1.0e-3,
        output_activation="sigmoid",
    )

    model = build_baseline_model(
        config
    )

    compile_baseline_model(
        model,
        config,
    )

    test_gravity = tf.random.uniform(
        shape=(
            2,
            *GRAVITY_SHAPE,
        ),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
        seed=20260727,
    )

    predicted_density = model(
        test_gravity,
        training=False,
    )

    expected_prediction_shape = (
        2,
        *DENSITY_SHAPE,
    )

    if (
        tuple(
            predicted_density.shape
        )
        != expected_prediction_shape
    ):
        raise AssertionError(
            "Prediction shape is incorrect. "
            f"Expected {expected_prediction_shape}, "
            f"received {predicted_density.shape}."
        )

    if not bool(
        tf.reduce_all(
            tf.math.is_finite(
                predicted_density
            )
        )
    ):
        raise AssertionError(
            "Prediction contains NaN or infinite values."
        )

    if config.output_activation == "sigmoid":
        if not bool(
            tf.reduce_all(
                predicted_density >= 0.0
            )
        ):
            raise AssertionError(
                "Sigmoid prediction contains values below zero."
            )

        if not bool(
            tf.reduce_all(
                predicted_density <= 1.0
            )
        ):
            raise AssertionError(
                "Sigmoid prediction contains values above one."
            )

    print()
    print("Baseline 3D CNN model test")
    print("=" * 27)
    print(
        f"Model input shape: "
        f"{model.input_shape}"
    )
    print(
        f"Model output shape: "
        f"{model.output_shape}"
    )
    print(
        f"Test prediction shape: "
        f"{predicted_density.shape}"
    )
    print(
        f"Trainable parameters: "
        f"{count_trainable_parameters(model):,}"
    )
    print(
        f"Prediction range: "
        f"{float(tf.reduce_min(predicted_density)):.8e} "
        f"to "
        f"{float(tf.reduce_max(predicted_density)):.8e}"
    )
    print()
    print("Model construction: PASSED")
    print("Model compilation: PASSED")
    print("Input shape check: PASSED")
    print("Output shape check: PASSED")
    print("Forward-pass finite-value check: PASSED")


if __name__ == "__main__":
    main()
