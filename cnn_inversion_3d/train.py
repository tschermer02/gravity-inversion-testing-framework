from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cnn_inversion_3d.dataset import (
    GRAVITY_SHAPE,
    SINGLE_PLANE_GRAVITY_SHAPE,
    build_training_datasets,
    find_repository_root,
)
from cnn_inversion_3d.diagnostics import (
    CollapseDetectionCallback,
)
from cnn_inversion_3d.differentiable_gravity import (
    DifferentiableSinglePlaneGz,
    PhysicsConsistencyTrainingModel,
    SaveBestInversionModel,
    excess_occupied_volume_fraction_mse,
    soft_occupied_fraction,
)
from cnn_inversion_3d.model import (
    ModelConfig,
    VerticalExpansion,
    build_baseline_model,
    build_asymmetric_2d_unet_model,
    build_e10_sensitivity_unet_model,
    build_single_plane_model,
    build_single_plane_learned_depth_seed_model,
    compile_baseline_model,
    count_trainable_parameters,
)
from cnn_inversion_3d.e10_training import (
    E10LossConfig,
    E10TrainingModel,
    build_e10_sensitivity_weights,
)
from cnn_inversion_3d.e09a_training import E09ADepthLossConfig, E09ATrainingModel
from cnn_inversion_3d.e09b_training import (
    E09BLossConfig, E09BTrainingModel, build_e09b_sensitivity_weights,
)
from cnn_inversion_3d.e09b911_training import E09B911LossConfig, E09B911TrainingModel
from cnn_inversion_3d.e09c_training import E09CLossConfig, E09CTrainingModel

from cnn_inversion_3d.normalization import (
    GravityScaleMethod,
    load_gravity_normalization,
)


@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration for baseline 3D CNN training.

    Training intentionally uses:

    - no explicit regularization
    - no dropout
    - no physics-based loss
    - no observational noise
    - balanced body/background mean squared error
    """

    dataset_directory: Path = Path(
        "datasets/fwd3d_smoke_test"
    )

    output_directory: Path = Path(
        "training_outputs/fwd3d_smoke_test"
    )

    batch_size: int = 2
    epochs: int = 3

    body_loss_fraction: float = 0.5

    gravity_scale: float = 1.0

    gravity_scale_summary: Path | None = None
    gravity_scale_method: GravityScaleMethod = "percentile_99"

    density_scale: float = 1.0

    random_seed: int = 20260727

    base_filters: int = 8
    learning_rate: float = 1.0e-3
    output_activation: str = "sigmoid"
    vertical_expansion: VerticalExpansion = "repeat"
    architecture: Literal[
        "multi_height_3d",
        "single_plane_2d3d",
        "single_plane_2d3d_learned_depth_seed",
        "single_plane_asymmetric_2d_unet",
        "single_plane_asymmetric_2d_unet_depth_loss",
        "single_plane_asymmetric_2d_unet_sensitivity_loss",
        "single_plane_asymmetric_2d_unet_extent_loss",
        "single_plane_e10_sensitivity_unet",
    ] = (
        "multi_height_3d"
    )
    gravity_loss_weight: float = 0.0
    volume_loss_weight: float = 0.0
    volume_threshold: float = 0.1
    volume_sharpness: float = 60.0
    e10_lambda_shape: float = 1.0
    e10_lambda_sensitivity: float = 1.0
    e10_lambda_physics: float = 1.0e-4
    e10_sensitivity_gamma: float = 0.25
    e10_occupancy_threshold: float = 0.1
    e10_occupancy_sharpness: float = 10.0
    e10_ablation: Literal["A", "B", "C"] | None = None
    e09a_lambda_density: float = 1.0
    e09a_lambda_depth: float = 1.0
    e09a_alpha_center: float = 1.0
    e09a_epsilon: float = 1.0e-8
    e09b_lambda_sensitivity: float = 1.0
    e09b_sensitivity_gamma: float = 0.5
    e09b_weight_min: float = 0.5
    e09b_weight_max: float = 5.0
    e09b_lambda_amplitude: float = 0.0
    e09b_small_body_weighting: bool = False
    e09b_volume_gamma: float = 0.5
    e09b_sample_weight_min: float = 0.5
    e09b_sample_weight_max: float = 2.0
    e09b_training_median_body_volume_cells: float = 1.0
    e09b_training_weight_mean: float = 1.0
    e09b_lambda_body_density: float = 0.0
    e09b_lambda_gravity: float = 0.0
    e09c_lambda_extent: float = 1.0
    e09c_top_quantile: float = 0.05
    e09c_bottom_quantile: float = 0.95
    e09c_boundary_sharpness: float = 100.0

    collapse_threshold: float = 1.0e-5
    collapse_patience: int = 2
    stop_on_collapse: bool = False

    overwrite: bool = False

    def validate(self) -> None:
        """Validate training settings."""

        if self.batch_size < 1:
            raise ValueError(
                "batch_size must be at least one."
            )

        if self.epochs < 1:
            raise ValueError(
                "epochs must be at least one."
            )

        if self.gravity_scale <= 0.0:
            raise ValueError(
                "gravity_scale must be greater than zero."
            )

        if self.density_scale <= 0.0:
            raise ValueError(
                "density_scale must be greater than zero."
            )

        if self.base_filters < 1:
            raise ValueError(
                "base_filters must be at least one."
            )

        if self.learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be greater than zero."
            )

        if self.gravity_scale_method not in {
            "absolute_maximum",
            "percentile_99",
            "standard_deviation",
        }:
            raise ValueError(
                "gravity_scale_method must be one of: "
                "'absolute_maximum', 'percentile_99', or "
                "'standard_deviation'."
            )

        if self.vertical_expansion not in {
            "repeat",
            "transpose",
        }:
            raise ValueError(
                "vertical_expansion must be either "
                "'repeat' or 'transpose'."
            )
        if self.architecture not in {
            "multi_height_3d",
            "single_plane_2d3d",
            "single_plane_2d3d_learned_depth_seed",
            "single_plane_asymmetric_2d_unet",
            "single_plane_asymmetric_2d_unet_depth_loss",
            "single_plane_asymmetric_2d_unet_sensitivity_loss",
            "single_plane_asymmetric_2d_unet_extent_loss",
            "single_plane_e10_sensitivity_unet",
        }:
            raise ValueError("Unsupported architecture.")
        if self.gravity_loss_weight < 0.0:
            raise ValueError("gravity_loss_weight must not be negative.")
        if self.volume_loss_weight < 0.0:
            raise ValueError("volume_loss_weight must not be negative.")
        if not 0.0 < self.volume_threshold < 1.0:
            raise ValueError("volume_threshold must be between zero and one.")
        if self.volume_sharpness <= 0.0:
            raise ValueError("volume_sharpness must be greater than zero.")
        if (
            (self.gravity_loss_weight > 0.0 or self.volume_loss_weight > 0.0)
            and self.architecture
            not in {
                "single_plane_2d3d_learned_depth_seed",
                "single_plane_asymmetric_2d_unet",
                "single_plane_asymmetric_2d_unet_depth_loss",
                "single_plane_asymmetric_2d_unet_sensitivity_loss",
                "single_plane_e10_sensitivity_unet",
            }
        ):
            raise ValueError(
                "Physics/volume consistency training requires the E06 "
                "learned-depth-seed or E09 asymmetric U-Net architecture."
            )

        if self.collapse_threshold < 0.0:
            raise ValueError(
                "collapse_threshold must not be negative."
            )

        if self.collapse_patience < 1:
            raise ValueError(
                "collapse_patience must be at least one."
            )
        if self.architecture == "single_plane_e10_sensitivity_unet":
            E10LossConfig(
                lambda_shape=self.e10_lambda_shape,
                lambda_sensitivity=self.e10_lambda_sensitivity,
                lambda_physics=self.e10_lambda_physics,
                sensitivity_gamma=self.e10_sensitivity_gamma,
                occupancy_threshold=self.e10_occupancy_threshold,
                occupancy_sharpness=self.e10_occupancy_sharpness,
                body_fraction=self.body_loss_fraction,
            ).validate()
        elif self.e10_ablation is not None:
            raise ValueError("--e10-ablation requires the E10 architecture.")
        if self.architecture in {
            "single_plane_asymmetric_2d_unet_depth_loss",
            "single_plane_asymmetric_2d_unet_sensitivity_loss",
        }:
            E09ADepthLossConfig(
                lambda_density=self.e09a_lambda_density,
                lambda_depth=self.e09a_lambda_depth,
                alpha_center=self.e09a_alpha_center,
                epsilon=self.e09a_epsilon,
                body_fraction=self.body_loss_fraction,
            ).validate()
        if self.architecture == "single_plane_asymmetric_2d_unet_sensitivity_loss":
            E09B911LossConfig(
                lambda_density=self.e09a_lambda_density,
                lambda_depth=self.e09a_lambda_depth,
                alpha_center=self.e09a_alpha_center,
                lambda_sensitivity=self.e09b_lambda_sensitivity,
                sensitivity_gamma=self.e09b_sensitivity_gamma,
                sensitivity_weight_min=self.e09b_weight_min,
                sensitivity_weight_max=self.e09b_weight_max,
                lambda_amplitude=self.e09b_lambda_amplitude,
                small_body_weighting=self.e09b_small_body_weighting,
                volume_gamma=self.e09b_volume_gamma,
                sample_weight_min=self.e09b_sample_weight_min,
                sample_weight_max=self.e09b_sample_weight_max,
                lambda_body_density=self.e09b_lambda_body_density,
                lambda_gravity=self.e09b_lambda_gravity,
                epsilon=self.e09a_epsilon,
                body_fraction=self.body_loss_fraction,
            ).validate()
        if self.architecture == "single_plane_asymmetric_2d_unet_extent_loss":
            E09CLossConfig(lambda_density=self.e09a_lambda_density,lambda_depth=self.e09a_lambda_depth,alpha_center=self.e09a_alpha_center,
                lambda_sensitivity=self.e09b_lambda_sensitivity,sensitivity_gamma=self.e09b_sensitivity_gamma,
                sensitivity_weight_min=self.e09b_weight_min,sensitivity_weight_max=self.e09b_weight_max,
                lambda_extent=self.e09c_lambda_extent,top_quantile=self.e09c_top_quantile,bottom_quantile=self.e09c_bottom_quantile,
                boundary_sharpness=self.e09c_boundary_sharpness,epsilon=self.e09a_epsilon,body_fraction=self.body_loss_fraction).validate()


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the baseline 3D gravity-inversion CNN."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Dataset directory. Relative paths are interpreted "
            "from the repository root."
        ),
    )

    parser.add_argument(
        "--gravity-scale-summary",
        type=Path,
        default=None,
        help=(
            "Training-only gravity-distribution JSON file. When supplied, "
            "the selected statistic overrides --gravity-scale."
        ),
    )

    parser.add_argument(
        "--gravity-scale-method",
        choices=(
            "absolute_maximum",
            "percentile_99",
            "standard_deviation",
        ),
        default=None,
        help=(
            "Statistic loaded from --gravity-scale-summary. "
            "Default: percentile_99."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Training-output directory. Relative paths are interpreted "
            "from the repository root."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size.",
    )

    parser.add_argument(
        "--base-filters",
        type=int,
        default=None,
        help="Number of filters in the first convolution block.",
    )

    parser.add_argument(
        "--architecture",
        choices=(
            "multi_height_3d",
            "single_plane_2d3d",
            "single_plane_2d3d_learned_depth_seed",
            "single_plane_asymmetric_2d_unet",
            "single_plane_asymmetric_2d_unet_depth_loss",
            "single_plane_asymmetric_2d_unet_sensitivity_loss",
            "single_plane_asymmetric_2d_unet_extent_loss",
            "single_plane_e10_sensitivity_unet",
        ),
        default=None,
        help="Additive model/data path. Default preserves multi-height 3D.",
    )

    parser.add_argument(
        "--gravity-loss-weight",
        type=float,
        default=None,
        help=(
            "Differentiable relative-L2 gravity loss weight. Default 0 "
            "preserves all existing training objectives; canonical E07 uses 0.001."
        ),
    )

    parser.add_argument(
        "--volume-loss-weight", type=float, default=None,
        help="Weight for one-sided excess-volume loss. Default 0 preserves historical runs; corrected E08 uses 0.01.",
    )
    parser.add_argument(
        "--volume-threshold", type=float, default=None,
        help="Density midpoint for soft occupancy. Default: 0.1 g/cm^3.",
    )
    parser.add_argument(
        "--volume-sharpness", type=float, default=None,
        help="Sigmoid sharpness for soft occupancy. Default: 60.",
    )
    parser.add_argument("--e10-lambda-shape", type=float, default=None)
    parser.add_argument("--e10-lambda-sensitivity", type=float, default=None)
    parser.add_argument("--e10-lambda-physics", type=float, default=None)
    parser.add_argument("--e10-sensitivity-gamma", type=float, default=None)
    parser.add_argument("--e10-occupancy-threshold", type=float, default=None)
    parser.add_argument("--e10-occupancy-sharpness", type=float, default=None)
    parser.add_argument("--e09a-lambda-density", type=float, default=None)
    parser.add_argument("--e09a-lambda-depth", type=float, default=None)
    parser.add_argument("--e09a-alpha-center", type=float, default=None)
    parser.add_argument("--e09a-epsilon", type=float, default=None)
    parser.add_argument("--e09b-lambda-sensitivity", type=float, default=None)
    parser.add_argument("--e09b-sensitivity-gamma", type=float, default=None)
    parser.add_argument("--e09b-weight-min", type=float, default=None)
    parser.add_argument("--e09b-weight-max", type=float, default=None)
    parser.add_argument("--e09b-lambda-amplitude", type=float, default=None)
    parser.add_argument("--e09b-small-body-weighting", action="store_true", default=None)
    parser.add_argument("--e09b-volume-gamma", type=float, default=None)
    parser.add_argument("--e09b-sample-weight-min", type=float, default=None)
    parser.add_argument("--e09b-sample-weight-max", type=float, default=None)
    parser.add_argument("--e09b-lambda-body-density", type=float, default=None)
    parser.add_argument("--e09b-lambda-gravity", type=float, default=None)
    parser.add_argument("--e09c-lambda-extent", type=float, default=None)
    parser.add_argument("--e09c-top-quantile", type=float, default=None)
    parser.add_argument("--e09c-bottom-quantile", type=float, default=None)
    parser.add_argument("--e09c-boundary-sharpness", type=float, default=None)
    parser.add_argument(
        "--e10-ablation", choices=("A", "B", "C"), default=None,
        help="Controlled E10 loss preset; architecture is identical for A/B/C.",
    )

    parser.add_argument(
        "--vertical-expansion",
        choices=(
            "repeat",
            "transpose",
        ),
        default=None,
        help=(
            "Layer used to expand receiver depth from 8 to 24. "
            "Use 'repeat' for UpSampling3D or 'transpose' for "
            "Conv3DTranspose."
        ),
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed used by Python, NumPy, TensorFlow, and "
            "training-dataset shuffling."
        ),
    )

    parser.add_argument(
        "--collapse-threshold",
        type=float,
        default=None,
        help=(
            "Validation prediction maximum below which an epoch is "
            "considered collapsed."
        ),
    )

    parser.add_argument(
        "--collapse-patience",
        type=int,
        default=None,
        help=(
            "Consecutive collapsed epochs required before reporting "
            "collapse."
        ),
    )

    parser.add_argument(
        "--stop-on-collapse",
        action="store_true",
        help=(
            "Stop training after collapse detection. By default, "
            "collapse is reported without stopping."
        ),
    )

    parser.add_argument(
        "--gravity-scale",
        type=float,
        default=None,
        help="Constant used to normalize gravity values.",
    )

    parser.add_argument(
        "--density-scale",
        type=float,
        default=None,
        help="Constant used to normalize density values.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing training-output directory.",
    )

    return parser


def build_inversion_model_for_architecture(
    architecture: str, model_config: ModelConfig
) -> tf.keras.Model:
    """Build the inference network selected by the training architecture."""

    if architecture == "single_plane_e10_sensitivity_unet":
        return build_e10_sensitivity_unet_model(model_config)
    if architecture in {
        "single_plane_asymmetric_2d_unet",
        "single_plane_asymmetric_2d_unet_depth_loss",
        "single_plane_asymmetric_2d_unet_sensitivity_loss",
        "single_plane_asymmetric_2d_unet_extent_loss",
    }:
        return build_asymmetric_2d_unet_model(model_config)
    if architecture == "single_plane_2d3d_learned_depth_seed":
        return build_single_plane_learned_depth_seed_model(model_config)
    if architecture == "single_plane_2d3d":
        return build_single_plane_model(model_config)
    return build_baseline_model(model_config)


def apply_arguments(
    *,
    config: TrainingConfig,
    arguments: argparse.Namespace,
) -> TrainingConfig:
    """Return a configuration updated from command-line arguments."""

    values = asdict(config)

    if arguments.dataset is not None:
        values["dataset_directory"] = (
            arguments.dataset
        )

    if arguments.output is not None:
        values["output_directory"] = (
            arguments.output
        )

    if arguments.gravity_scale_summary is not None:
        values["gravity_scale_summary"] = (
            arguments.gravity_scale_summary
        )

    if arguments.gravity_scale_method is not None:
        values["gravity_scale_method"] = (
            arguments.gravity_scale_method
        )

    if arguments.epochs is not None:
        values["epochs"] = arguments.epochs

    if arguments.batch_size is not None:
        values["batch_size"] = (
            arguments.batch_size
        )

    if arguments.base_filters is not None:
        values["base_filters"] = (
            arguments.base_filters
        )

    if arguments.vertical_expansion is not None:
        values["vertical_expansion"] = (
            arguments.vertical_expansion
        )
    if arguments.architecture is not None:
        values["architecture"] = arguments.architecture
    e09a_options_supplied = any(
        getattr(arguments, name) is not None
        for name in (
            "e09a_lambda_density", "e09a_lambda_depth",
            "e09a_alpha_center", "e09a_epsilon",
        )
    )
    selected_architecture = values.get("architecture", config.architecture)
    if (
        e09a_options_supplied
        and selected_architecture not in {
            "single_plane_asymmetric_2d_unet_depth_loss",
            "single_plane_asymmetric_2d_unet_sensitivity_loss",
            "single_plane_asymmetric_2d_unet_extent_loss",
        }
    ):
        raise ValueError(
            "E09A loss options require --architecture "
            "single_plane_asymmetric_2d_unet_depth_loss; the ordinary E09 "
            "architecture selector would ignore depth supervision."
        )
    e09b_options_supplied = any(
        getattr(arguments, name) is not None
        for name in (
            "e09b_lambda_sensitivity", "e09b_sensitivity_gamma",
            "e09b_weight_min", "e09b_weight_max",
            "e09b_lambda_amplitude", "e09b_small_body_weighting",
            "e09b_volume_gamma", "e09b_sample_weight_min", "e09b_sample_weight_max",
            "e09b_lambda_body_density", "e09b_lambda_gravity",
        )
    )
    if (
        e09b_options_supplied
        and selected_architecture not in {"single_plane_asymmetric_2d_unet_sensitivity_loss","single_plane_asymmetric_2d_unet_extent_loss"}
    ):
        raise ValueError(
            "E09B sensitivity options require --architecture "
            "single_plane_asymmetric_2d_unet_sensitivity_loss."
        )
    if arguments.gravity_loss_weight is not None:
        values["gravity_loss_weight"] = arguments.gravity_loss_weight
    if arguments.volume_loss_weight is not None:
        values["volume_loss_weight"] = arguments.volume_loss_weight
    if arguments.volume_threshold is not None:
        values["volume_threshold"] = arguments.volume_threshold
    if arguments.volume_sharpness is not None:
        values["volume_sharpness"] = arguments.volume_sharpness
    if arguments.e10_ablation is not None:
        values["e10_ablation"] = arguments.e10_ablation
        preset = E10LossConfig.for_ablation(arguments.e10_ablation)
        values.update({
            "e10_lambda_shape": preset.lambda_shape,
            "e10_lambda_sensitivity": preset.lambda_sensitivity,
            "e10_lambda_physics": preset.lambda_physics,
            "e10_sensitivity_gamma": preset.sensitivity_gamma,
            "e10_occupancy_threshold": preset.occupancy_threshold,
            "e10_occupancy_sharpness": preset.occupancy_sharpness,
        })
    for argument_name, config_name in (
        ("e10_lambda_shape", "e10_lambda_shape"),
        ("e10_lambda_sensitivity", "e10_lambda_sensitivity"),
        ("e10_lambda_physics", "e10_lambda_physics"),
        ("e10_sensitivity_gamma", "e10_sensitivity_gamma"),
        ("e10_occupancy_threshold", "e10_occupancy_threshold"),
        ("e10_occupancy_sharpness", "e10_occupancy_sharpness"),
        ("e09a_lambda_density", "e09a_lambda_density"),
        ("e09a_lambda_depth", "e09a_lambda_depth"),
        ("e09a_alpha_center", "e09a_alpha_center"),
        ("e09a_epsilon", "e09a_epsilon"),
        ("e09b_lambda_sensitivity", "e09b_lambda_sensitivity"),
        ("e09b_sensitivity_gamma", "e09b_sensitivity_gamma"),
        ("e09b_weight_min", "e09b_weight_min"),
        ("e09b_weight_max", "e09b_weight_max"),
        ("e09b_lambda_amplitude", "e09b_lambda_amplitude"),
        ("e09b_small_body_weighting", "e09b_small_body_weighting"),
        ("e09b_volume_gamma", "e09b_volume_gamma"),
        ("e09b_sample_weight_min", "e09b_sample_weight_min"),
        ("e09b_sample_weight_max", "e09b_sample_weight_max"),
        ("e09b_lambda_body_density", "e09b_lambda_body_density"),
        ("e09b_lambda_gravity", "e09b_lambda_gravity"),
        ("e09c_lambda_extent", "e09c_lambda_extent"),("e09c_top_quantile", "e09c_top_quantile"),
        ("e09c_bottom_quantile", "e09c_bottom_quantile"),("e09c_boundary_sharpness", "e09c_boundary_sharpness"),
    ):
        value = getattr(arguments, argument_name)
        if value is not None:
            values[config_name] = value

    if arguments.learning_rate is not None:
        values["learning_rate"] = (
            arguments.learning_rate
        )

    if arguments.seed is not None:
        values["random_seed"] = (
            arguments.seed
        )

    if arguments.collapse_threshold is not None:
        values["collapse_threshold"] = (
            arguments.collapse_threshold
        )

    if arguments.collapse_patience is not None:
        values["collapse_patience"] = (
            arguments.collapse_patience
        )

    if arguments.stop_on_collapse:
        values["stop_on_collapse"] = True

    if arguments.gravity_scale is not None:
        values["gravity_scale"] = (
            arguments.gravity_scale
        )

    if arguments.density_scale is not None:
        values["density_scale"] = (
            arguments.density_scale
        )

    if arguments.overwrite:
        values["overwrite"] = True

    return TrainingConfig(**values)


def resolve_path(
    *,
    repository_root: Path,
    path: Path,
) -> Path:
    """Resolve a path relative to the repository root."""

    if not path.is_absolute():
        path = repository_root / path

    return path.resolve()


def resolve_gravity_scale(
    *,
    config: TrainingConfig,
    repository_root: Path,
) -> tuple[float, Path | None]:
    """
    Resolve the global gravity scale used for all dataset splits.

    An explicit training-distribution summary takes precedence over the
    numeric ``gravity_scale`` configuration value.

    Parameters
    ----------
    config
        Training configuration.
    repository_root
        Repository root used to resolve relative paths.

    Returns
    -------
    tuple
        Resolved positive gravity scale and optional source JSON path.
    """

    if config.gravity_scale_summary is None:
        return (
            config.gravity_scale,
            None,
        )

    summary_path = resolve_path(
        repository_root=repository_root,
        path=config.gravity_scale_summary,
    )

    normalization = load_gravity_normalization(
        summary_path=summary_path,
        method=config.gravity_scale_method,
    )

    return (
        normalization.scale,
        normalization.source_path,
    )


def prepare_output_directory(
    *,
    output_directory: Path,
    overwrite: bool,
) -> None:
    """Create the training-output directory."""

    if output_directory.exists():
        if not overwrite:
            raise FileExistsError(
                "Training-output directory already exists:\n"
                f"{output_directory}\n"
                "Use --overwrite to replace it."
            )

        import shutil

        shutil.rmtree(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )


def save_history_csv(
    *,
    history: tf.keras.callbacks.History,
    output_path: Path,
) -> None:
    """Save Keras training history as CSV."""

    history_data = history.history

    metric_names = list(
        history_data.keys()
    )

    epoch_count = len(
        history_data[metric_names[0]]
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.writer(output_file)

        writer.writerow(
            ["epoch", *metric_names]
        )

        for epoch_index in range(epoch_count):
            writer.writerow(
                [
                    epoch_index + 1,
                    *[
                        history_data[name][epoch_index]
                        for name in metric_names
                    ],
                ]
            )


def save_history_figure(
    *,
    history: tf.keras.callbacks.History,
    output_path: Path,
) -> None:
    """Save training and validation loss curves."""

    training_loss = history.history[
        "loss"
    ]

    validation_loss = history.history[
        "val_loss"
    ]

    epochs = range(
        1,
        len(training_loss) + 1,
    )

    figure, axis = plt.subplots(
        figsize=(8.0, 5.0)
    )

    axis.plot(
        epochs,
        training_loss,
        marker="o",
        label="Training loss",
    )

    axis.plot(
        epochs,
        validation_loss,
        marker="o",
        label="Validation loss",
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Density MSE")
    axis.set_title(
        "Baseline 3D CNN training history"
    )
    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_training_metadata(
    *,
    output_path: Path,
    config: TrainingConfig,
    model: tf.keras.Model,
    split_counts: dict[str, int],
    elapsed_seconds: float,
    final_metrics: dict[str, float],
    history: tf.keras.callbacks.History,
    collapse_detection: dict[str, Any],
    pretraining_loss_scales: dict[str, Any] | None = None,
) -> None:
    """
    Save training metadata as JSON.

    Parameters
    ----------
    output_path
        JSON output path.
    config
        Fully resolved training configuration.
    model
        Trained Keras model.
    split_counts
        Number of samples in each dataset split.
    elapsed_seconds
        Total training duration in seconds.
    final_metrics
        Test metrics calculated using the best saved model.
    history
        Keras history containing epoch-level diagnostics.
    collapse_detection
        Serializable collapse callback result.
    """

    training_configuration = asdict(
        config
    )

    training_configuration[
        "dataset_directory"
    ] = str(
        config.dataset_directory
    )

    training_configuration[
        "output_directory"
    ] = str(
        config.output_directory
    )

    training_configuration[
        "gravity_scale_summary"
    ] = (
        str(
            config.gravity_scale_summary
        )
        if config.gravity_scale_summary is not None
        else None
    )

    metadata: dict[str, Any] = {
        "experiment_name": config.output_directory.name,
        "training_configuration": (
            training_configuration
        ),
        "model_name": model.name,
        "optimizer": "Adam",
        "model_input_shape": list(
            model.input_shape
        ),
        "model_output_shape": list(
            model.output_shape
        ),
        "trainable_parameters": (
            count_trainable_parameters(
                model
            )
        ),
        "split_counts": split_counts,
        "elapsed_seconds": elapsed_seconds,
        "final_metrics": final_metrics,
        "final_epoch_diagnostics": {
            name: float(values[-1])
            for name, values in history.history.items()
            if values
        },
        "collapse_detection": collapse_detection,
        "pretraining_loss_scales": pretraining_loss_scales,
        "best_epoch": int(
            np.argmin(history.history["val_loss"]) + 1
        ),
        "best_validation_loss": float(
            np.min(history.history["val_loss"])
        ),
        "reproducibility": {
            "seed": config.random_seed,
            "seeded_libraries": [
                "python.random",
                "numpy",
                "tensorflow",
                "tf.data training shuffle",
            ],
            "bitwise_determinism_guaranteed": False,
            "note": (
                "Seeds provide practical repeatability, but TensorFlow "
                "kernels, hardware, threading, and oneDNN may still "
                "produce nondeterministic or numerically different results."
            ),
        },
        "loss": {
            "name": (
                "e10_soft_iou_plus_sensitivity_balanced_mse_plus_data_weighted_gravity"
                if config.architecture == "single_plane_e10_sensitivity_unet"
                else "e09c_e09b_plus_soft_vertical_extent"
                if config.architecture == "single_plane_asymmetric_2d_unet_extent_loss"
                else "e09b_e09a_plus_integrated_sensitivity_compensated_mse"
                if config.architecture == "single_plane_asymmetric_2d_unet_sensitivity_loss"
                else "e09a_balanced_density_mse_plus_depth_supervision"
                if config.architecture == "single_plane_asymmetric_2d_unet_depth_loss"
                else
                "balanced_density_mse_plus_global_normalized_gravity_mse_plus_excess_volume_loss"
                if config.volume_loss_weight > 0.0
                else "balanced_density_mse_plus_global_normalized_gravity_mse"
                if config.gravity_loss_weight > 0.0
                else "balanced_density_mse"
            ),
            "body_fraction": (
                config.body_loss_fraction
            ),
            "gravity_loss_weight": config.gravity_loss_weight,
            "gravity_loss_definition": (
                "mean(square((G_true - F(rho_pred)) / Wd)) in mGal"
                if config.architecture == "single_plane_e10_sensitivity_unet"
                else "E09B + lambda_extent * (top_boundary_mse + bottom_boundary_mse + thickness_mse) / 3"
                if config.architecture == "single_plane_asymmetric_2d_unet_extent_loss"
                else "mean(square(F(rho_pred)/gravity_scale - "
                "G_true/gravity_scale)) over all batch samples and pixels"
                if config.gravity_loss_weight > 0.0
                else None
            ),
            "background_fraction": (
                1.0
                - config.body_loss_fraction
            ),
            "gravity_loss_normalization": (
                "Wd=sqrt(diag(GG^T)); input Gz restored to mGal using training-set global percentile_99"
                if config.architecture == "single_plane_e10_sensitivity_unet"
                else "training_set_global_percentile_99"
                if config.gravity_loss_weight > 0.0
                else None
            ),
            "gravity_scale_mgal": config.gravity_scale,
            "per_sample_gravity_loss_normalization": False,
            "volume_loss_weight": config.volume_loss_weight,
            "volume_threshold": config.volume_threshold,
            "volume_sigmoid_sharpness": config.volume_sharpness,
            "volume_loss_definition": (
                "mean_batch(relu(mean_voxels(zero_floor_soft_occ(rho_pred)) - "
                "mean_voxels(rho_true>0))^2)"
                if config.volume_loss_weight > 0.0 else None
            ),
            "loss_equation": (
                "lambda_shape * soft_iou_loss + lambda_sensitivity * "
                "sensitivity_balanced_50_50_mse + lambda_physics * "
                "Wd_inverse_gravity_mse"
                if config.architecture == "single_plane_e10_sensitivity_unet"
                else "lambda_density * BalancedDensityMSE + lambda_depth * "
                "(normalized_depth_profile_mse + alpha_center * normalized_z_center_mse) + "
                "lambda_sensitivity * integrated_sensitivity_compensated_mse + "
                "lambda_amplitude * true_body_region_density_amplitude_mse; optional "
                "lambda_body_density * true_body_voxel_density_mse + "
                "lambda_gravity * global_normalized_gravity_mse; optional "
                "training-only normalized true-body-volume sample weighting"
                if config.architecture == "single_plane_asymmetric_2d_unet_sensitivity_loss"
                else "lambda_density * BalancedDensityMSE + lambda_depth * "
                "(normalized_depth_profile_mse + alpha_center * normalized_z_center_mse)"
                if config.architecture == "single_plane_asymmetric_2d_unet_depth_loss"
                else "BalancedDensityMSE + gravity_loss_weight * global_normalized_gravity_mse "
                "+ volume_loss_weight * excess_occupied_volume_fraction_loss"
            ),
            "e10": (
                {
                    "ablation": config.e10_ablation,
                    "lambda_shape": config.e10_lambda_shape,
                    "lambda_sensitivity": config.e10_lambda_sensitivity,
                    "lambda_physics": config.e10_lambda_physics,
                    "sensitivity_gamma": config.e10_sensitivity_gamma,
                    "occupancy_threshold": config.e10_occupancy_threshold,
                    "occupancy_sharpness": config.e10_occupancy_sharpness,
                    "body_fraction": config.body_loss_fraction,
                    "background_fraction": 1.0 - config.body_loss_fraction,
                    "gravity_units": "mGal before Wd normalization",
                    "density_order": "z,y,x",
                    "gravity_order": "y,x",
                }
                if config.architecture == "single_plane_e10_sensitivity_unet"
                else None
            ),
            "e09a": (
                {
                    "lambda_density": config.e09a_lambda_density,
                    "lambda_depth": config.e09a_lambda_depth,
                    "alpha_center": config.e09a_alpha_center,
                    "epsilon": config.e09a_epsilon,
                    "depth_coordinates_m": [5.0 + 10.0 * index for index in range(24)],
                    "density_order": "z,y,x",
                }
                if config.architecture in {
                    "single_plane_asymmetric_2d_unet_depth_loss",
                    "single_plane_asymmetric_2d_unet_sensitivity_loss",
                    "single_plane_asymmetric_2d_unet_extent_loss",
                }
                else None
            ),
            "e09b": (
                {
                    "lambda_density": config.e09a_lambda_density,
                    "lambda_depth": config.e09a_lambda_depth,
                    "alpha_center": config.e09a_alpha_center,
                    "lambda_sensitivity": config.e09b_lambda_sensitivity,
                    "sensitivity_gamma": config.e09b_sensitivity_gamma,
                    "sensitivity_weight_min": config.e09b_weight_min,
                    "sensitivity_weight_max": config.e09b_weight_max,
                    "lambda_amplitude": config.e09b_lambda_amplitude,
                    "density_amplitude_definition": (
                        "square(mean(prediction over true body mask) - "
                        "mean(truth over true body mask))"
                    ),
                    "small_body_weighting": config.e09b_small_body_weighting,
                    "volume_gamma": config.e09b_volume_gamma,
                    "sample_weight_min": config.e09b_sample_weight_min,
                    "sample_weight_max": config.e09b_sample_weight_max,
                    "training_median_body_volume_cells": config.e09b_training_median_body_volume_cells,
                    "training_raw_clipped_weight_mean": config.e09b_training_weight_mean,
                    "volume_statistics_source": "train_manifest.csv only",
                    "validation_and_test_sample_weighting": False,
                    "lambda_body_density": config.e09b_lambda_body_density,
                    "body_density_definition": (
                        "sum(true_body_mask * square(prediction-truth)) / "
                        "(sum(true_body_mask)+epsilon)"
                    ),
                    "lambda_gravity": config.e09b_lambda_gravity,
                    "gravity_loss_definition": (
                        "existing global_normalized_gravity_mse using fixed "
                        "DifferentiableSinglePlaneGz"
                    ),
                    "forward_operator_trainable": False,
                    "epsilon": config.e09a_epsilon,
                    "sensitivity_normalization": "bounded clipped weights with exact global mean one",
                    "sensitivity_definition": "S_k = sqrt(sum_i A_ik^2)",
                    "weight_definition": "w_k proportional to (S_max/(S_k+epsilon))^gamma",
                    "scientific_scope": (
                        "supervised ML sensitivity compensation; not classical "
                        "Tikhonov Wm=S regularization"
                    ),
                    "density_order": "z,y,x",
                }
                if config.architecture in {"single_plane_asymmetric_2d_unet_sensitivity_loss","single_plane_asymmetric_2d_unet_extent_loss"}
                else None
            ),
            "e09c": ({"lambda_extent":config.e09c_lambda_extent,"top_quantile":config.e09c_top_quantile,
                "bottom_quantile":config.e09c_bottom_quantile,"boundary_sharpness":config.e09c_boundary_sharpness,
                "boundary_method":"soft normalized-Z-mass CDF quantiles on physical cell edges"}
                if config.architecture=="single_plane_asymmetric_2d_unet_extent_loss" else None),
        },
        "explicit_regularization": (
            "one_sided_zero_floor_excess_occupied_volume_constraint"
            if config.volume_loss_weight > 0.0 else None
        ),
        "observational_noise": None,
    }

    if config.architecture.startswith("single_plane_2d3d"):
        transformations = [
            "81 x 81 x 1 surface Gz",
            "64 x 64 x 8 full-plane resampled surface features",
            "32 x 32 x 16 encoder features",
            "16 x 16 x 32 bottleneck features",
            "64 x 64 x 8 decoded lateral features",
        ]
        if config.architecture == "single_plane_2d3d_learned_depth_seed":
            transformations.extend(
                [
                    "64 x 64 x 48 learned depth-channel features",
                    "6 x 64 x 64 x 8 learned depth seed",
                ]
            )
        else:
            transformations.append(
                "6 x 64 x 64 x 8 repeated identical depth seed"
            )
        transformations.extend([
            "12 x 64 x 64 x 8 depth decoder",
            "24 x 64 x 64 x 8 depth decoder",
            "24 x 64 x 64 x 1 density output",
        ])
        metadata["dimensional_transformations"] = transformations
    elif config.architecture in {
        "single_plane_asymmetric_2d_unet",
        "single_plane_asymmetric_2d_unet_depth_loss",
        "single_plane_asymmetric_2d_unet_sensitivity_loss",
        "single_plane_asymmetric_2d_unet_extent_loss",
    }:
        metadata["dimensional_transformations"] = [
            "81 x 81 x 1 complete surface Gz",
            "96 x 96 x 1 deterministic zero-padded surface",
            "96 -> 48 -> 24 -> 12 2D U-Net encoder",
            "12 -> 24 -> 48 -> 96 2D U-Net decoder with skip connections",
            "96 x 96 learned valid-convolution transform to 64 x 64",
            "64 x 64 x 24 directly supervised physical depth channels",
            "24 x 64 x 64 x 1 deterministic canonical permutation/reshape",
        ]
    elif config.architecture == "single_plane_e10_sensitivity_unet":
        metadata["dimensional_transformations"] = [
            "81 x 81 x 1 complete surface Gz",
            "128 x 128 x 1 deterministic zero-padded surface",
            "128 x 128 x 8 encoder features",
            "64 x 64 x 16 encoder features",
            "32 x 32 x 32 encoder features",
            "16 x 16 x 64 bottleneck",
            "32 x 32 x 32 decoder features with skip",
            "64 x 64 x 16 decoder features with skip",
            "128 x 128 x 8 decoder features with highest-resolution skip",
            "64 x 64 x 8 learned stride-two spatial projection",
            "64 x 64 x 24 directly supervised physical depth channels",
            "24 x 64 x 64 x 1 initial canonical density volume",
            "two 3 x 3 x 3 Conv3D layers and residual 1 x 1 x 1 correction",
            "24 x 64 x 64 x 1 bounded refined density output",
        ]

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=2,
        )


def run_volume_regime_diagnostics(
    *, threshold: float, sharpness: float, volume_loss_weight: float
) -> dict[str, Any]:
    """Evaluate corrected volume behavior on tiny artificial tensors only."""

    threshold_values = tf.Variable([0.05, 0.10, 0.15], dtype=tf.float32)
    with tf.GradientTape() as tape:
        occupancies = soft_occupied_fraction(
            threshold_values[:, None, None, None, None],
            threshold=threshold,
            sharpness=sharpness,
        )
    threshold_gradients = tape.gradient(occupancies, threshold_values)

    truth = tf.constant([[[[[1.0], [1.0], [0.0], [0.0]]]]], tf.float32)
    cases = {
        "zero": tf.zeros_like(truth),
        "matching": tf.constant([[[[[1.0], [1.0], [0.0], [0.0]]]]], tf.float32),
        "excess": tf.constant([[[[[1.0], [1.0], [0.2], [0.0]]]]], tf.float32),
        "deficit": tf.constant([[[[[1.0], [0.0], [0.0], [0.0]]]]], tf.float32),
    }
    results: dict[str, Any] = {
        "threshold_regime_density_values": threshold_values.numpy().tolist(),
        "threshold_regime_soft_occupancy": occupancies.numpy().tolist(),
        "threshold_regime_soft_occupancy_gradients": threshold_gradients.numpy().tolist(),
        "soft_occupancy_at_zero": float(
            soft_occupied_fraction(
                tf.zeros((1, 1, 1, 1, 1), tf.float32),
                threshold=threshold, sharpness=sharpness,
            ).numpy()[0]
        ),
    }
    for name, values in cases.items():
        prediction = tf.Variable(values)
        with tf.GradientTape() as tape:
            terms = excess_occupied_volume_fraction_mse(
                truth, prediction, threshold=threshold, sharpness=sharpness
            )
        gradient = tape.gradient(terms[0], prediction)
        results[f"{name}_excess_volume_loss"] = float(terms[0].numpy())
        results[f"{name}_weighted_excess_volume_loss"] = float(
            volume_loss_weight * terms[0].numpy()
        )
        results[f"{name}_excess_occupied_fraction"] = float(terms[3].numpy())
        results[f"{name}_gradient_norm"] = float(tf.linalg.global_norm([gradient]).numpy())
    excess_prediction = tf.Variable(cases["excess"])
    with tf.GradientTape() as tape:
        excess_loss = excess_occupied_volume_fraction_mse(
            truth, excess_prediction, threshold=threshold, sharpness=sharpness
        )[0]
    excess_gradient = tape.gradient(excess_loss, excess_prediction)
    results["excess_example_gradient_at_extra_voxel"] = float(
        excess_gradient.numpy()[0, 0, 0, 2, 0]
    )
    return results


def run_physics_preflight(
    model: PhysicsConsistencyTrainingModel,
    gravity: tf.Tensor,
    density: tf.Tensor,
    *,
    learning_rate: float,
) -> dict[str, Any]:
    """Measure E07/E08 loss and gradient scales without mutating the model."""

    variables = model.inversion_model.trainable_variables
    with tf.GradientTape(persistent=True) as tape:
        terms = model.compute_loss_terms(gravity, density, training=False)
        density_loss = terms[1]
        gravity_loss = terms[2]
        weighted_gravity_loss = terms[3]
        volume_loss = terms[5]
        weighted_volume_loss = terms[6]
    density_gradients = tape.gradient(density_loss, variables)
    gravity_gradients = tape.gradient(gravity_loss, variables)
    weighted_gradients = tape.gradient(weighted_gravity_loss, variables)
    volume_gradients = tape.gradient(volume_loss, variables)
    weighted_volume_gradients = tape.gradient(weighted_volume_loss, variables)
    del tape

    def gradient_norm(values: list[tf.Tensor | None]) -> float:
        finite_values = [value for value in values if value is not None]
        return float(tf.linalg.global_norm(finite_values).numpy())

    density_gradient_norm = gradient_norm(density_gradients)
    gravity_gradient_norm = gradient_norm(gravity_gradients)
    weighted_gradient_norm = gradient_norm(weighted_gradients)
    volume_gradient_norm = gradient_norm(volume_gradients)
    weighted_volume_gradient_norm = gradient_norm(weighted_volume_gradients)

    diagnostic_inversion = tf.keras.models.clone_model(model.inversion_model)
    diagnostic_inversion.set_weights(model.inversion_model.get_weights())
    diagnostic_model = PhysicsConsistencyTrainingModel(
        diagnostic_inversion,
        DifferentiableSinglePlaneGz(),
        gravity_scale=model.gravity_scale,
        gravity_loss_weight=model.gravity_loss_weight,
        body_loss_fraction=model.density_loss_function.body_fraction,
        volume_loss_weight=model.volume_loss_weight,
        volume_threshold=model.volume_threshold,
        volume_sharpness=model.volume_sharpness,
    )
    diagnostic_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )
    before = diagnostic_inversion(gravity, training=False)
    diagnostic_model.train_on_batch(gravity, density)
    after = diagnostic_inversion(gravity, training=False)
    body_mask = tf.cast(density > 0.0, before.dtype)

    def body_mean(values: tf.Tensor) -> float:
        return float(
            tf.math.divide_no_nan(
                tf.reduce_sum(values * body_mask), tf.reduce_sum(body_mask)
            ).numpy()
        )

    ratio = (
        weighted_gradient_norm / density_gradient_norm
        if density_gradient_norm > 0.0
        else float("inf")
    )
    volume_ratio = (
        weighted_volume_gradient_norm / density_gradient_norm
        if density_gradient_norm > 0.0
        else float("inf")
    )
    return {
        "density_loss": float(density_loss.numpy()),
        "gravity_loss": float(gravity_loss.numpy()),
        "raw_gravity_loss": float(gravity_loss.numpy()),
        "weighted_gravity_loss": float(weighted_gravity_loss.numpy()),
        "raw_excess_volume_loss": float(volume_loss.numpy()),
        "weighted_excess_volume_loss": float(weighted_volume_loss.numpy()),
        "total_loss": float(terms[4].numpy()),
        "density_gradient_norm": density_gradient_norm,
        "gravity_gradient_norm": gravity_gradient_norm,
        "weighted_gravity_gradient_norm": weighted_gradient_norm,
        "excess_volume_gradient_norm": volume_gradient_norm,
        "weighted_excess_volume_gradient_norm": weighted_volume_gradient_norm,
        "weighted_gravity_to_density_gradient_ratio": ratio,
        "weighted_volume_to_density_gradient_ratio": volume_ratio,
        "true_occupied_fraction": float(terms[7].numpy()),
        "predicted_soft_occupied_fraction": float(terms[8].numpy()),
        "soft_volume_fraction_ratio": float(
            tf.math.divide_no_nan(terms[8], terms[7]).numpy()
        ),
        "excess_occupied_fraction": float(terms[9].numpy()),
        "volume_regime_diagnostics": run_volume_regime_diagnostics(
            threshold=model.volume_threshold,
            sharpness=model.volume_sharpness,
            volume_loss_weight=model.volume_loss_weight,
        ),
        "prediction_mean_before": float(tf.reduce_mean(before).numpy()),
        "prediction_mean_after": float(tf.reduce_mean(after).numpy()),
        "prediction_max_before": float(tf.reduce_max(before).numpy()),
        "prediction_max_after": float(tf.reduce_max(after).numpy()),
        "body_prediction_mean_before": body_mean(before),
        "body_prediction_mean_after": body_mean(after),
    }


def run_e10_preflight(
    model: E10TrainingModel,
    gravity: tf.Tensor,
    density: tf.Tensor,
) -> dict[str, Any]:
    """Verify E10 shapes, finite losses, and gradients before fitting."""

    variables = model.inversion_model.trainable_variables
    with tf.GradientTape(persistent=True) as tape:
        terms = model.compute_loss_terms(gravity, density, training=False)
    names = ("iou", "sensitivity", "gravity", "total")
    losses = dict(zip(names, terms[1:]))

    def norm(loss: tf.Tensor) -> float:
        gradients = tape.gradient(loss, variables)
        values = [value for value in gradients if value is not None]
        return float(tf.linalg.global_norm(values).numpy())

    gradient_norms = {f"{name}_gradient_norm": norm(loss) for name, loss in losses.items()}
    cfg = model.loss_config
    weighted_gradient_norms = {
        "weighted_shape_gradient_norm": cfg.lambda_shape * gradient_norms["iou_gradient_norm"],
        "weighted_density_gradient_norm": cfg.lambda_sensitivity * gradient_norms["sensitivity_gradient_norm"],
        "weighted_physics_gradient_norm": cfg.lambda_physics * gradient_norms["gravity_gradient_norm"],
    }
    del tape
    prediction = terms[0]
    expected_input = (gravity.shape[0], 81, 81, 1)
    expected_density = (gravity.shape[0], 24, 64, 64, 1)
    if tuple(gravity.shape) != expected_input:
        raise ValueError(f"E10 input shape mismatch: {gravity.shape}")
    if tuple(prediction.shape) != expected_density:
        raise ValueError(f"E10 density shape mismatch: {prediction.shape}")
    values = [float(loss.numpy()) for loss in losses.values()]
    if not np.all(
        np.isfinite(
            values
            + list(gradient_norms.values())
            + list(weighted_gradient_norms.values())
        )
    ):
        raise ValueError("E10 preflight contains nonfinite losses or gradients.")
    return {
        "input_gravity_shape": list(gravity.shape),
        "padded_input_shape": [gravity.shape[0], 128, 128, 1],
        "depth_channel_shape": [gravity.shape[0], 64, 64, 24],
        "density_volume_shape": list(prediction.shape),
        "iou_loss": float(losses["iou"].numpy()),
        "sensitivity_loss": float(losses["sensitivity"].numpy()),
        "gravity_loss": float(losses["gravity"].numpy()),
        "total_loss": float(losses["total"].numpy()),
        **gradient_norms,
        **weighted_gradient_norms,
        "sensitivity_balancing_enabled": cfg.sensitivity_enabled,
        "physics_contribution_enabled": cfg.lambda_physics > 0.0,
        "all_terms_differentiable": all(
            gradient_norms[f"{name}_gradient_norm"] > 0.0
            for name in ("iou", "sensitivity", "gravity")
        ),
        "density_order": "batch,z,y,x,channel",
        "gravity_order": "batch,y,x,channel",
    }


def run_e09a_preflight(
    model: E09ATrainingModel, gravity: tf.Tensor, density: tf.Tensor
) -> dict[str, Any]:
    """Report finite raw and weighted E09A component gradient norms."""

    variables = model.inversion_model.trainable_variables
    with tf.GradientTape(persistent=True) as tape:
        terms = model.compute_loss_terms(gravity, density, training=False)
    losses = {
        "density": terms[1], "depth_profile": terms[2],
        "z_center": terms[3], "depth": terms[4], "total": terms[5],
    }

    def norm(loss: tf.Tensor) -> float:
        gradients = [g for g in tape.gradient(loss, variables) if g is not None]
        return float(tf.linalg.global_norm(gradients).numpy())

    norms = {f"{name}_gradient_norm": norm(loss) for name, loss in losses.items()}
    del tape
    cfg = model.loss_config
    weighted = {
        "weighted_density_gradient_norm": cfg.lambda_density * norms["density_gradient_norm"],
        "weighted_depth_profile_gradient_norm": cfg.lambda_depth * norms["depth_profile_gradient_norm"],
        "weighted_z_center_gradient_norm": (
            cfg.lambda_depth * cfg.alpha_center * norms["z_center_gradient_norm"]
        ),
    }
    prediction = terms[0]
    values = [float(value.numpy()) for value in losses.values()]
    if tuple(prediction.shape[1:]) != (24, 64, 64, 1):
        raise ValueError(f"E09A density shape mismatch: {prediction.shape}")
    if not np.all(np.isfinite(values + list(norms.values()) + list(weighted.values()))):
        raise ValueError("E09A preflight contains NaN or Inf.")
    return {
        "input_shape": list(gravity.shape), "output_shape": list(prediction.shape),
        **{f"{name}_loss": float(value.numpy()) for name, value in losses.items()},
        **norms, **weighted,
        "all_terms_differentiable": all(
            norms[f"{name}_gradient_norm"] > 0.0
            for name in ("density", "depth_profile", "z_center")
        ),
        "density_order": "batch,z,y,x,channel",
        "depth_coordinates_m": [5.0 + 10.0 * index for index in range(24)],
    }


def save_e09a_loss_history_figure(
    history: tf.keras.callbacks.History, output_path: Path
) -> None:
    """Plot separately logged E09A training and validation loss terms."""

    figure, axis = plt.subplots(figsize=(9, 5))
    for name in ("loss", "density_loss", "depth_profile_loss", "z_center_loss", "depth_loss"):
        if name in history.history:
            axis.plot(history.history[name], label=name)
        if f"val_{name}" in history.history:
            axis.plot(history.history[f"val_{name}"], linestyle="--", label=f"val_{name}")
    axis.set(xlabel="Epoch", ylabel="Loss", title="E09A depth-supervision losses")
    axis.grid(alpha=0.3); axis.legend(ncol=2)
    figure.tight_layout(); figure.savefig(output_path, dpi=180); plt.close(figure)


def run_e09b_preflight(
    model: E09BTrainingModel, gravity: tf.Tensor, density: tf.Tensor
) -> dict[str, Any]:
    """Calculate raw/weighted E09B component gradients before fitting."""

    variables = model.inversion_model.trainable_variables
    with tf.GradientTape(persistent=True) as tape:
        terms = model.compute_loss_terms(gravity, density, training=False)
    losses = {
        "density": terms[1], "depth_profile": terms[2], "z_center": terms[3],
        "depth": terms[4], "sensitivity": terms[5], "amplitude": terms[6],
        "total": terms[7],
    }

    def norm(loss: tf.Tensor) -> float:
        gradients = [value for value in tape.gradient(loss, variables) if value is not None]
        return float(tf.linalg.global_norm(gradients).numpy())

    norms = {f"{name}_gradient_norm": norm(loss) for name, loss in losses.items()}
    del tape
    cfg = model.loss_config
    weighted = {
        "weighted_density_gradient_norm": cfg.lambda_density * norms["density_gradient_norm"],
        "weighted_depth_gradient_norm": cfg.lambda_depth * norms["depth_gradient_norm"],
        "weighted_sensitivity_gradient_norm": (
            cfg.lambda_sensitivity * norms["sensitivity_gradient_norm"]
        ),
        "weighted_amplitude_gradient_norm": (
            cfg.lambda_amplitude * norms["amplitude_gradient_norm"]
        ),
    }
    density_norm = max(weighted["weighted_density_gradient_norm"], cfg.epsilon)
    ratio = weighted["weighted_sensitivity_gradient_norm"] / density_norm
    values = [float(value.numpy()) for value in losses.values()]
    if not np.all(np.isfinite(values + list(norms.values()) + list(weighted.values()) + [ratio])):
        raise ValueError("E09B preflight contains NaN or Inf.")
    return {
        **{f"{name}_loss": float(value.numpy()) for name, value in losses.items()},
        **norms, **weighted,
        "weighted_sensitivity_to_density_gradient_ratio": ratio,
        "sensitivity_gradient_nonzero": norms["sensitivity_gradient_norm"] > 0.0,
        "sensitivity_overwhelms_density_by_orders_of_magnitude": ratio >= 100.0,
        "input_shape": list(gravity.shape), "output_shape": list(terms[0].shape),
        "density_order": "batch,z,y,x,channel",
    }


def training_body_volume_statistics(
    dataset_directory: Path, *, gamma: float, minimum: float, maximum: float
) -> dict[str, Any]:
    """Derive body-volume weighting constants from the training manifest only."""

    manifest = dataset_directory / "train_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    volumes = np.asarray(
        [float(row["width_x"]) * float(row["width_y"]) * float(row["thickness_z"]) for row in rows],
        dtype=np.float64,
    )
    if volumes.size == 0 or not np.all(np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError("Training manifest contains invalid body volumes.")
    median = float(np.median(volumes))
    clipped = np.clip(np.power(median / volumes, gamma), minimum, maximum)
    normalized = np.clip(clipped / np.mean(clipped), minimum, maximum)
    return {
        "source": str(manifest),
        "split": "training_only",
        "sample_count": int(volumes.size),
        "median_body_volume_cells": median,
        "minimum_body_volume_cells": float(np.min(volumes)),
        "maximum_body_volume_cells": float(np.max(volumes)),
        "raw_clipped_weight_mean": float(np.mean(clipped)),
        "normalized_weight_mean": float(np.mean(normalized)),
        "normalized_weight_minimum": float(np.min(normalized)),
        "normalized_weight_maximum": float(np.max(normalized)),
        "gamma": gamma,
        "clip_minimum": minimum,
        "clip_maximum": maximum,
    }


def run_e09b911_preflight(
    model: E09B911TrainingModel, gravity: tf.Tensor, density: tf.Tensor
) -> dict[str, Any]:
    """Verify finite E09B-9/10/11 terms and trainable-network gradients."""

    variables = model.inversion_model.trainable_variables
    with tf.GradientTape(persistent=True) as tape:
        terms = model.compute_loss_terms(gravity, density, training=False)
    names = ("density", "depth_profile", "z_center", "depth", "sensitivity",
             "amplitude", "body_density", "gravity", "weighted_gravity", "total")
    losses = dict(zip(names, terms[1:]))
    gradient_names = ("body_density", "gravity", "total")
    norms = {}
    for name in gradient_names:
        gradients = [value for value in tape.gradient(losses[name], variables) if value is not None]
        norms[f"{name}_gradient_norm"] = float(tf.linalg.global_norm(gradients).numpy())
    del tape
    values = {f"{name}_loss": float(value.numpy()) for name, value in losses.items()}
    if not np.all(np.isfinite([*values.values(), *norms.values()])):
        raise ValueError("E09B-9/10/11 preflight contains NaN or Inf.")
    return {**values, **norms, "input_shape": list(gravity.shape),
            "output_shape": list(terms[0].shape),
            "forward_operator_trainable_variables": len(model.forward_operator.trainable_variables),
            "density_order": "batch,z,y,x,channel"}
def run_e09c_preflight(
    model: E09CTrainingModel, gravity: tf.Tensor, density: tf.Tensor
) -> dict[str, Any]:
    variables = model.inversion_model.trainable_variables
    with tf.GradientTape(persistent=True) as tape:
        terms = model.compute_loss_terms(gravity, density, training=False)
    names = (
        "density", "depth_profile", "z_center", "depth", "sensitivity",
        "top_boundary", "bottom_boundary", "thickness", "extent", "total",
    )
    losses = dict(zip(names, terms[1:]))
    norms = {}
    for name, loss in losses.items():
        gradients = [
            value for value in tape.gradient(loss, variables) if value is not None
        ]
        norms[f"{name}_gradient_norm"] = float(
            tf.linalg.global_norm(gradients).numpy()
        )
    del tape
    values = {f"{name}_loss": float(loss.numpy()) for name, loss in losses.items()}
    if not np.all(np.isfinite(list(values.values()) + list(norms.values()))):
        raise ValueError("E09C preflight contains NaN or Inf.")
    return {
        **values,
        **norms,
        "weighted_extent_gradient_norm": (
            model.loss_config.lambda_extent * norms["extent_gradient_norm"]
        ),
        "input_shape": list(gravity.shape),
        "output_shape": list(terms[0].shape),
        "density_order": "batch,z,y,x,channel",
    }


def save_e09b_sensitivity_diagnostics(
    sensitivity: np.ndarray, weights: np.ndarray, output_directory: Path
) -> None:
    """Save fixed E09B sensitivity arrays, statistics, and depth profiles."""

    np.save(output_directory / "e09b_integrated_sensitivity.npy", sensitivity)
    np.save(output_directory / "e09b_sensitivity_weights.npy", weights)
    sensitivity_by_depth = np.mean(sensitivity, axis=(1, 2))
    weight_by_depth = np.mean(weights, axis=(1, 2))
    diagnostics = {
        "sensitivity_min": float(np.min(sensitivity)),
        "sensitivity_max": float(np.max(sensitivity)),
        "sensitivity_mean": float(np.mean(sensitivity)),
        "weight_min": float(np.min(weights)), "weight_max": float(np.max(weights)),
        "weight_mean": float(np.mean(weights)),
        "sensitivity_mean_by_depth": sensitivity_by_depth.tolist(),
        "weight_mean_by_depth": weight_by_depth.tolist(),
        "depth_bin_centers_m": (5.0 + 10.0 * np.arange(24)).tolist(),
        "deeper_cells_generally_receive_larger_weights": bool(
            np.mean(weight_by_depth[-4:]) > np.mean(weight_by_depth[:4])
        ),
        "density_order": "z,y,x",
        "sensitivity_definition": "sqrt(sum_i A_ik^2)",
        "interpretation": (
            "Supervised inverse integrated-sensitivity compensation; not a claim "
            "to reproduce classical Tikhonov Wm=S regularization."
        ),
    }
    (output_directory / "e09b_sensitivity_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    depth_m = np.asarray(diagnostics["depth_bin_centers_m"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, constrained_layout=True)
    axes[0].plot(sensitivity_by_depth, depth_m, marker="o")
    axes[0].set(xlabel="Mean integrated sensitivity", ylabel="Depth (m)", title="S vs depth")
    axes[1].plot(weight_by_depth, depth_m, marker="o")
    axes[1].set(xlabel="Mean compensation weight", title="w vs depth")
    for axis in axes:
        axis.invert_yaxis(); axis.grid(alpha=0.3)
    figure.savefig(output_directory / "e09b_sensitivity_vs_depth.png", dpi=180)
    plt.close(figure)


def save_e09b_loss_history_figure(
    history: tf.keras.callbacks.History, output_path: Path
) -> None:
    """Plot separately logged E09B loss components."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for name in (
        "loss", "density_loss", "depth_profile_loss", "z_center_loss",
        "depth_loss", "sensitivity_loss", "density_amplitude_loss",
        "body_density_loss", "gravity_loss", "weighted_gravity_loss",
        "gravity_rmse", "gravity_correlation",
        "extent_loss", "top_boundary_loss", "bottom_boundary_loss", "thickness_loss",
    ):
        if name in history.history:
            axis.plot(history.history[name], label=name)
        if f"val_{name}" in history.history:
            axis.plot(history.history[f"val_{name}"], linestyle="--", label=f"val_{name}")
    axis.set(xlabel="Epoch", ylabel="Loss", title="E09B loss components")
    axis.grid(alpha=0.3); axis.legend(ncol=2)
    figure.tight_layout(); figure.savefig(output_path, dpi=180); plt.close(figure)


def save_e10_weight_visualization(
    sensitivity: np.ndarray,
    weights: np.ndarray,
    output_directory: Path,
) -> None:
    """Save E10 fixed arrays and compact depth-profile visualization."""

    np.save(output_directory / "e10_integrated_sensitivity.npy", sensitivity)
    np.save(output_directory / "e10_sensitivity_weights.npy", weights)
    depth_m = (np.arange(sensitivity.shape[0]) + 0.5) * 10.0
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes[0, 0].plot(np.mean(sensitivity, axis=(1, 2)), depth_m, marker="o")
    axes[0, 0].set(xlabel="Mean integrated sensitivity", ylabel="Depth (m)", title="S(z)")
    axes[0, 1].plot(np.mean(weights, axis=(1, 2)), depth_m, marker="o")
    axes[0, 1].set(xlabel="Mean inverse-sensitivity weight", title="w_sens(z)")
    for axis in axes[0]:
        axis.invert_yaxis(); axis.grid(alpha=0.3)
    sensitivity_image = axes[1, 0].imshow(
        sensitivity[:, sensitivity.shape[1] // 2, :], origin="upper", aspect="auto",
        extent=(0, 640, 240, 0), cmap="viridis",
    )
    axes[1, 0].set(xlabel="X (m)", ylabel="Depth (m)", title="Central X-Z sensitivity")
    figure.colorbar(sensitivity_image, ax=axes[1, 0], shrink=0.8)
    weight_image = axes[1, 1].imshow(
        weights[:, weights.shape[1] // 2, :], origin="upper", aspect="auto",
        extent=(0, 640, 240, 0), cmap="magma",
    )
    axes[1, 1].set(xlabel="X (m)", ylabel="Depth (m)", title="Central X-Z sensitivity weight")
    figure.colorbar(weight_image, ax=axes[1, 1], shrink=0.8)
    figure.savefig(output_directory / "e10_sensitivity_weights.png", dpi=180)
    plt.close(figure)


def save_e10_loss_history_figure(
    history: tf.keras.callbacks.History, output_path: Path
) -> None:
    """Plot E10 total and individual train/validation loss components."""

    figure, axis = plt.subplots(figsize=(9, 5))
    for name in ("loss", "iou_loss", "sensitivity_loss", "gravity_loss"):
        if name in history.history:
            axis.plot(history.history[name], label=name)
        validation_name = f"val_{name}"
        if validation_name in history.history:
            axis.plot(history.history[validation_name], linestyle="--", label=validation_name)
    axis.set(xlabel="Epoch", ylabel="Loss", title="E10 loss components")
    axis.grid(alpha=0.3); axis.legend(ncol=2)
    figure.tight_layout(); figure.savefig(output_path, dpi=180); plt.close(figure)


def main() -> None:
    """Train the baseline 3D CNN."""

    parser = build_argument_parser()
    arguments = parser.parse_args()

    config = apply_arguments(
        config=TrainingConfig(),
        arguments=arguments,
    )

    config.validate()

    repository_root = (
        find_repository_root()
    )

    dataset_directory = resolve_path(
        repository_root=repository_root,
        path=config.dataset_directory,
    )

    output_directory = resolve_path(
        repository_root=repository_root,
        path=config.output_directory,
    )

    resolved_gravity_scale, gravity_scale_source = (
        resolve_gravity_scale(
            config=config,
            repository_root=repository_root,
        )
    )

    prepare_output_directory(
        output_directory=output_directory,
        overwrite=config.overwrite,
    )

    random.seed(
        config.random_seed
    )
    np.random.seed(
        config.random_seed
    )
    tf.keras.utils.set_random_seed(
        config.random_seed
    )

    (
        training_dataset,
        validation_dataset,
        test_dataset,
        split_counts,
    ) = build_training_datasets(
        dataset_directory=dataset_directory,
        batch_size=config.batch_size,
        gravity_scale=resolved_gravity_scale,
        density_scale=config.density_scale,
        random_seed=config.random_seed,
        gravity_shape=(
            SINGLE_PLANE_GRAVITY_SHAPE
            if config.architecture.startswith("single_plane")
            else GRAVITY_SHAPE
        ),
    )

    model_config = ModelConfig(
        base_filters=config.base_filters,
        learning_rate=config.learning_rate,
        output_activation=(
            config.output_activation
        ),
        body_loss_fraction=(
            config.body_loss_fraction
        ),
        vertical_expansion=(
            config.vertical_expansion
        ),
    )

    model = build_inversion_model_for_architecture(config.architecture, model_config)
    inversion_model = model
    e10_training = config.architecture == "single_plane_e10_sensitivity_unet"
    e09a_training = config.architecture == "single_plane_asymmetric_2d_unet_depth_loss"
    e09b_training = config.architecture == "single_plane_asymmetric_2d_unet_sensitivity_loss"
    extended_e09b = False
    e09c_training = config.architecture == "single_plane_asymmetric_2d_unet_extent_loss"
    physics_training = (
        e10_training
        or config.gravity_loss_weight > 0.0
        or config.volume_loss_weight > 0.0
    )
    pretraining_loss_scales: dict[str, Any] | None = None
    if e09c_training:
        sensitivity,sensitivity_weights=build_e09b_sensitivity_weights(gamma=config.e09b_sensitivity_gamma,epsilon=config.e09a_epsilon,weight_min=config.e09b_weight_min,weight_max=config.e09b_weight_max)
        save_e09b_sensitivity_diagnostics(sensitivity,sensitivity_weights,output_directory)
        cfg=E09CLossConfig(lambda_density=config.e09a_lambda_density,lambda_depth=config.e09a_lambda_depth,alpha_center=config.e09a_alpha_center,
            lambda_sensitivity=config.e09b_lambda_sensitivity,sensitivity_gamma=config.e09b_sensitivity_gamma,sensitivity_weight_min=config.e09b_weight_min,
            sensitivity_weight_max=config.e09b_weight_max,lambda_extent=config.e09c_lambda_extent,top_quantile=config.e09c_top_quantile,
            bottom_quantile=config.e09c_bottom_quantile,boundary_sharpness=config.e09c_boundary_sharpness,epsilon=config.e09a_epsilon,body_fraction=config.body_loss_fraction)
        model=E09CTrainingModel(inversion_model,sensitivity_weights,loss_config=cfg); model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate))
        sample_gravity,sample_density=next(iter(training_dataset)); pretraining_loss_scales=run_e09c_preflight(model,sample_gravity,sample_density)
        (output_directory/"e09c_pretraining_diagnostics.json").write_text(json.dumps(pretraining_loss_scales,indent=2),encoding="utf-8")
        print("E09C gradient norms:",pretraining_loss_scales)
    elif e09b_training:
        sensitivity, sensitivity_weights = build_e09b_sensitivity_weights(
            gamma=config.e09b_sensitivity_gamma,
            epsilon=config.e09a_epsilon,
            weight_min=config.e09b_weight_min,
            weight_max=config.e09b_weight_max,
        )
        save_e09b_sensitivity_diagnostics(
            sensitivity, sensitivity_weights, output_directory
        )
        volume_statistics = training_body_volume_statistics(
            dataset_directory,
            gamma=config.e09b_volume_gamma,
            minimum=config.e09b_sample_weight_min,
            maximum=config.e09b_sample_weight_max,
        )
        if config.e09b_small_body_weighting:
            (output_directory / "e09b_training_volume_weights.json").write_text(
                json.dumps(volume_statistics, indent=2), encoding="utf-8"
            )
        extended_e09b = (
            config.e09b_lambda_body_density > 0.0
            or config.e09b_lambda_gravity > 0.0
        )
        loss_config_class = E09B911LossConfig if extended_e09b else E09BLossConfig
        e09b_loss_config = loss_config_class(
            lambda_density=config.e09a_lambda_density,
            lambda_depth=config.e09a_lambda_depth,
            alpha_center=config.e09a_alpha_center,
            lambda_sensitivity=config.e09b_lambda_sensitivity,
            sensitivity_gamma=config.e09b_sensitivity_gamma,
            sensitivity_weight_min=config.e09b_weight_min,
            sensitivity_weight_max=config.e09b_weight_max,
            lambda_amplitude=config.e09b_lambda_amplitude,
            small_body_weighting=config.e09b_small_body_weighting,
            volume_gamma=config.e09b_volume_gamma,
            sample_weight_min=config.e09b_sample_weight_min,
            sample_weight_max=config.e09b_sample_weight_max,
            training_median_body_volume_cells=volume_statistics["median_body_volume_cells"],
            training_weight_mean=volume_statistics["raw_clipped_weight_mean"],
            **({
                "lambda_body_density": config.e09b_lambda_body_density,
                "lambda_gravity": config.e09b_lambda_gravity,
            } if extended_e09b else {}),
            epsilon=config.e09a_epsilon,
            body_fraction=config.body_loss_fraction,
        )
        model = (
            E09B911TrainingModel(
                inversion_model, sensitivity_weights, DifferentiableSinglePlaneGz(),
                gravity_scale=resolved_gravity_scale, loss_config=e09b_loss_config,
            )
            if extended_e09b
            else E09BTrainingModel(
                inversion_model, sensitivity_weights, loss_config=e09b_loss_config
            )
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.learning_rate),
            jit_compile=False if extended_e09b else "auto",
        )
        sample_gravity, sample_density = next(iter(training_dataset))
        pretraining_loss_scales = (
            run_e09b911_preflight(model, sample_gravity, sample_density)
            if extended_e09b
            else run_e09b_preflight(model, sample_gravity, sample_density)
        )
        print("E09B pre-training diagnostics:", pretraining_loss_scales)
        (output_directory / "e09b_pretraining_diagnostics.json").write_text(
            json.dumps(pretraining_loss_scales, indent=2), encoding="utf-8"
        )
    elif e09a_training:
        e09a_loss_config = E09ADepthLossConfig(
            lambda_density=config.e09a_lambda_density,
            lambda_depth=config.e09a_lambda_depth,
            alpha_center=config.e09a_alpha_center,
            epsilon=config.e09a_epsilon,
            body_fraction=config.body_loss_fraction,
        )
        model = E09ATrainingModel(inversion_model, loss_config=e09a_loss_config)
        model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate))
        sample_gravity, sample_density = next(iter(training_dataset))
        pretraining_loss_scales = run_e09a_preflight(model, sample_gravity, sample_density)
        print("E09A raw gradient norms:", {
            key: pretraining_loss_scales[key] for key in (
                "density_gradient_norm", "depth_profile_gradient_norm", "z_center_gradient_norm"
            )
        })
        print("E09A weighted gradient norms:", {
            key: pretraining_loss_scales[key] for key in (
                "weighted_density_gradient_norm", "weighted_depth_profile_gradient_norm",
                "weighted_z_center_gradient_norm"
            )
        })
        (output_directory / "e09a_pretraining_diagnostics.json").write_text(
            json.dumps(pretraining_loss_scales, indent=2), encoding="utf-8"
        )
    elif e10_training:
        sensitivity, sensitivity_weights, data_weights = build_e10_sensitivity_weights(
            gamma=config.e10_sensitivity_gamma
        )
        save_e10_weight_visualization(
            sensitivity, sensitivity_weights, output_directory
        )
        np.save(output_directory / "e10_data_weights.npy", data_weights)
        e10_loss_config = E10LossConfig(
            lambda_shape=config.e10_lambda_shape,
            lambda_sensitivity=config.e10_lambda_sensitivity,
            lambda_physics=config.e10_lambda_physics,
            sensitivity_gamma=config.e10_sensitivity_gamma,
            occupancy_threshold=config.e10_occupancy_threshold,
            occupancy_sharpness=config.e10_occupancy_sharpness,
            body_fraction=config.body_loss_fraction,
        )
        model = E10TrainingModel(
            inversion_model,
            DifferentiableSinglePlaneGz(),
            sensitivity_weights,
            data_weights,
            gravity_scale=resolved_gravity_scale,
            loss_config=e10_loss_config,
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate))
        sample_gravity, sample_density = next(iter(training_dataset))
        pretraining_loss_scales = run_e10_preflight(
            model, sample_gravity, sample_density
        )
        print("E10 raw gradient norms:", {
            name: pretraining_loss_scales[name] for name in (
                "iou_gradient_norm", "sensitivity_gradient_norm", "gravity_gradient_norm"
            )
        })
        print("E10 weighted gradient norms:", {
            name: pretraining_loss_scales[name] for name in (
                "weighted_shape_gradient_norm", "weighted_density_gradient_norm",
                "weighted_physics_gradient_norm"
            )
        })
        (output_directory / "e10_pretraining_diagnostics.json").write_text(
            json.dumps(pretraining_loss_scales, indent=2), encoding="utf-8"
        )
    elif physics_training:
        model = PhysicsConsistencyTrainingModel(
            inversion_model,
            DifferentiableSinglePlaneGz(),
            gravity_scale=resolved_gravity_scale,
            gravity_loss_weight=config.gravity_loss_weight,
            body_loss_fraction=config.body_loss_fraction,
            volume_loss_weight=config.volume_loss_weight,
            volume_threshold=config.volume_threshold,
            volume_sharpness=config.volume_sharpness,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.learning_rate
            )
        )
        sample_gravity, sample_density = next(iter(training_dataset))
        pretraining_loss_scales = run_physics_preflight(
            model,
            sample_gravity,
            sample_density,
            learning_rate=config.learning_rate,
        )
        mean_collapsed = (
            pretraining_loss_scales["prediction_mean_after"]
            < 0.1 * pretraining_loss_scales["prediction_mean_before"]
        )
        maximum_collapsed = (
            pretraining_loss_scales["prediction_max_after"]
            < 0.1 * pretraining_loss_scales["prediction_max_before"]
        )
        gradient_failed = (
            pretraining_loss_scales[
                "weighted_gravity_to_density_gradient_ratio"
            ] > 5.0
        )
        volume_gradient_failed = (
            pretraining_loss_scales[
                "weighted_volume_to_density_gradient_ratio"
            ] > 5.0
        )
        (
            output_directory / "pretraining_loss_scales.json"
        ).write_text(
            json.dumps(
                {
                    **pretraining_loss_scales,
                    "gravity_loss_weight": config.gravity_loss_weight,
                    "volume_loss_weight": config.volume_loss_weight,
                    "volume_threshold": config.volume_threshold,
                    "volume_sigmoid_sharpness": config.volume_sharpness,
                    "maximum_permitted_weighted_gradient_ratio": 5.0,
                    "gradient_safety_passed": not (
                        gradient_failed or volume_gradient_failed
                    ),
                    "one_step_collapse_detected": (
                        mean_collapsed or maximum_collapsed
                    ),
                    "preflight_passed": not (
                        gradient_failed or volume_gradient_failed
                        or mean_collapsed or maximum_collapsed
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if gradient_failed or volume_gradient_failed or mean_collapsed or maximum_collapsed:
            raise RuntimeError(
                "E07/E08 pre-training safety check failed: a weighted auxiliary "
                "gradient exceeds five times the density gradient or the "
                "copy-only optimizer step collapsed prediction magnitude. "
                f"Observed {pretraining_loss_scales}. Training stopped "
                "without changing lambda_gravity."
            )
    else:
        compile_baseline_model(model, model_config)

    best_model_path = (
        output_directory
        / "best_model.keras"
    )

    final_model_path = (
        output_directory
        / "final_model.keras"
    )

    collapse_callback = CollapseDetectionCallback(
        threshold=config.collapse_threshold,
        patience=config.collapse_patience,
        stop_training=(config.stop_on_collapse or physics_training),
    )

    checkpoint_callback: tf.keras.callbacks.Callback = (
        SaveBestInversionModel(inversion_model, best_model_path)
        if (physics_training or e09a_training or e09b_training or e09c_training)
        else tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        )
    )
    callbacks = [
        checkpoint_callback,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            mode="min",
            restore_best_weights=False,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        collapse_callback,
    ]

    print()
    print("Training baseline 3D CNN")
    print("=" * 26)
    print(
        f"Dataset: {dataset_directory}"
    )
    print(
        f"Output: {output_directory}"
    )
    print(
        f"Split counts: {split_counts}"
    )
    print(
        f"Batch size: {config.batch_size}"
    )
    print(
        f"Epochs: {config.epochs}"
    )
    print(
        f"Random seed: {config.random_seed}"
    )
    print(
        f"Trainable parameters: "
        f"{count_trainable_parameters(model):,}"
    )
    print(
        "Gravity scale: "
        f"{resolved_gravity_scale:.12e}"
    )

    print(
        "Gravity scale method: "
        f"{config.gravity_scale_method}"
    )
    print(
        "Collapse detection: robust near-zero validation criterion for "
        f"{config.collapse_patience} epoch(s)"
    )
    if pretraining_loss_scales is not None:
        print(f"Pre-training loss scales: {pretraining_loss_scales}")

    if gravity_scale_source is not None:
        print(
            "Gravity scale source: "
            f"{gravity_scale_source}"
        )
    print()

    training_start = perf_counter()

    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed_seconds = (
        perf_counter()
        - training_start
    )

    inversion_model.save(
        final_model_path
    )
    (
        output_directory / "model_architecture.json"
    ).write_text(inversion_model.to_json(indent=2), encoding="utf-8")

    best_model = tf.keras.models.load_model(
        str(best_model_path),
        compile=False,
    )

    compile_baseline_model(
        best_model,
        model_config,
        jit_compile=False if extended_e09b else "auto",
    )

    test_results = best_model.evaluate(
        test_dataset,
        return_dict=True,
        verbose=1,
    )

    if not isinstance(test_results, dict):
        raise RuntimeError(
            "Expected evaluate(return_dict=True) to return a dictionary."
        )

    final_metrics = {
        name: float(value)
        for name, value in test_results.items()
    }

    history_csv_path = (
        output_directory
        / "training_history.csv"
    )

    history_figure_path = (
        output_directory
        / "training_history.png"
    )

    metadata_path = (
        output_directory
        / "training_metadata.json"
    )

    save_history_csv(
        history=history,
        output_path=history_csv_path,
    )

    save_history_figure(
        history=history,
        output_path=history_figure_path,
    )
    if e10_training:
        save_e10_loss_history_figure(
            history, output_directory / "e10_loss_components.png"
        )
    if e09a_training:
        save_e09a_loss_history_figure(history, output_directory / "e09a_loss_components.png")
    if e09b_training:
        save_e09b_loss_history_figure(history, output_directory / "e09b_loss_components.png")
    if e09c_training:
        save_e09b_loss_history_figure(history, output_directory / "e09c_loss_components.png")

    resolved_config = TrainingConfig(
        dataset_directory=dataset_directory,
        output_directory=output_directory,
        batch_size=config.batch_size,
        epochs=config.epochs,
        body_loss_fraction=(
            config.body_loss_fraction
        ),
        gravity_scale=(
            resolved_gravity_scale
        ),
        gravity_scale_summary=(
            gravity_scale_source
        ),
        gravity_scale_method=(
            config.gravity_scale_method
        ),
        density_scale=(
            config.density_scale
        ),
        random_seed=(
            config.random_seed
        ),
        base_filters=(
            config.base_filters
        ),
        learning_rate=(
            config.learning_rate
        ),
        output_activation=(
            config.output_activation
        ),
        vertical_expansion=(
            config.vertical_expansion
        ),
        architecture=config.architecture,
        gravity_loss_weight=config.gravity_loss_weight,
        volume_loss_weight=config.volume_loss_weight,
        volume_threshold=config.volume_threshold,
        volume_sharpness=config.volume_sharpness,
        e10_lambda_shape=config.e10_lambda_shape,
        e10_lambda_sensitivity=config.e10_lambda_sensitivity,
        e10_lambda_physics=config.e10_lambda_physics,
        e10_sensitivity_gamma=config.e10_sensitivity_gamma,
        e10_occupancy_threshold=config.e10_occupancy_threshold,
        e10_occupancy_sharpness=config.e10_occupancy_sharpness,
        e10_ablation=config.e10_ablation,
        e09a_lambda_density=config.e09a_lambda_density,
        e09a_lambda_depth=config.e09a_lambda_depth,
        e09a_alpha_center=config.e09a_alpha_center,
        e09a_epsilon=config.e09a_epsilon,
        e09b_lambda_sensitivity=config.e09b_lambda_sensitivity,
        e09b_sensitivity_gamma=config.e09b_sensitivity_gamma,
        e09b_weight_min=config.e09b_weight_min,
        e09b_weight_max=config.e09b_weight_max,
        e09b_lambda_amplitude=config.e09b_lambda_amplitude,
        e09b_small_body_weighting=config.e09b_small_body_weighting,
        e09b_volume_gamma=config.e09b_volume_gamma,
        e09b_sample_weight_min=config.e09b_sample_weight_min,
        e09b_sample_weight_max=config.e09b_sample_weight_max,
        e09b_training_median_body_volume_cells=(
            volume_statistics["median_body_volume_cells"]
            if e09b_training else config.e09b_training_median_body_volume_cells
        ),
        e09b_training_weight_mean=(
            volume_statistics["raw_clipped_weight_mean"]
            if e09b_training else config.e09b_training_weight_mean
        ),
        e09b_lambda_body_density=config.e09b_lambda_body_density,
        e09b_lambda_gravity=config.e09b_lambda_gravity,
        e09c_lambda_extent=config.e09c_lambda_extent,
        e09c_top_quantile=config.e09c_top_quantile,
        e09c_bottom_quantile=config.e09c_bottom_quantile,
        e09c_boundary_sharpness=config.e09c_boundary_sharpness,
        collapse_threshold=(
            config.collapse_threshold
        ),
        collapse_patience=(
            config.collapse_patience
        ),
        stop_on_collapse=(
            config.stop_on_collapse or physics_training
        ),
        overwrite=(
            config.overwrite
        ),
    )

    save_training_metadata(
        output_path=metadata_path,
        config=resolved_config,
        model=best_model,
        split_counts=split_counts,
        elapsed_seconds=elapsed_seconds,
        final_metrics=final_metrics,
        history=history,
        collapse_detection=(
            collapse_callback.result()
        ),
        pretraining_loss_scales=pretraining_loss_scales,
    )

    print()
    print("Training complete")
    print("=" * 17)
    print(
        f"Elapsed time: "
        f"{elapsed_seconds:.2f} seconds"
    )
    print(
        f"Test metrics: {final_metrics}"
    )
    print(
        f"Best model: {best_model_path}"
    )
    print(
        f"Final model: {final_model_path}"
    )
    print(
        f"History CSV: {history_csv_path}"
    )
    print(
        f"History figure: "
        f"{history_figure_path}"
    )
    print(
        f"Metadata: {metadata_path}"
    )


if __name__ == "__main__":
    main()
