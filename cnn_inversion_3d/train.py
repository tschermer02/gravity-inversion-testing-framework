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
)
from cnn_inversion_3d.model import (
    ModelConfig,
    VerticalExpansion,
    build_baseline_model,
    build_single_plane_model,
    build_single_plane_learned_depth_seed_model,
    compile_baseline_model,
    count_trainable_parameters,
)

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
    ] = (
        "multi_height_3d"
    )
    gravity_loss_weight: float = 0.0

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
        }:
            raise ValueError("Unsupported architecture.")
        if self.gravity_loss_weight < 0.0:
            raise ValueError("gravity_loss_weight must not be negative.")
        if (
            self.gravity_loss_weight > 0.0
            and self.architecture
            != "single_plane_2d3d_learned_depth_seed"
        ):
            raise ValueError(
                "Gravity consistency training requires the E06 "
                "single_plane_2d3d_learned_depth_seed architecture."
            )

        if self.collapse_threshold < 0.0:
            raise ValueError(
                "collapse_threshold must not be negative."
            )

        if self.collapse_patience < 1:
            raise ValueError(
                "collapse_patience must be at least one."
            )


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
            "preserves all existing training objectives; E07 uses 0.1."
        ),
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
    if arguments.gravity_loss_weight is not None:
        values["gravity_loss_weight"] = arguments.gravity_loss_weight

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
    pretraining_loss_scales: dict[str, float] | None = None,
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
        "training_configuration": (
            training_configuration
        ),
        "model_name": model.name,
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
                "balanced_density_mse_plus_global_normalized_gravity_mse"
                if config.gravity_loss_weight > 0.0
                else "balanced_density_mse"
            ),
            "body_fraction": (
                config.body_loss_fraction
            ),
            "gravity_loss_weight": config.gravity_loss_weight,
            "gravity_loss_definition": (
                "mean(square(F(rho_pred)/gravity_scale - "
                "G_true/gravity_scale)) over all batch samples and pixels"
                if config.gravity_loss_weight > 0.0
                else None
            ),
            "background_fraction": (
                1.0
                - config.body_loss_fraction
            ),
            "gravity_loss_normalization": (
                "training_set_global_percentile_99"
                if config.gravity_loss_weight > 0.0
                else None
            ),
            "gravity_scale_mgal": config.gravity_scale,
            "per_sample_gravity_loss_normalization": False,
        },
        "explicit_regularization": None,
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

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=2,
        )


def run_physics_preflight(
    model: PhysicsConsistencyTrainingModel,
    gravity: tf.Tensor,
    density: tf.Tensor,
    *,
    learning_rate: float,
) -> dict[str, float]:
    """Measure E07 loss/gradient scale and a non-mutating optimizer step."""

    variables = model.inversion_model.trainable_variables
    with tf.GradientTape(persistent=True) as tape:
        terms = model.compute_loss_terms(gravity, density, training=False)
        density_loss = terms[1]
        gravity_loss = terms[2]
        weighted_gravity_loss = terms[3]
    density_gradients = tape.gradient(density_loss, variables)
    gravity_gradients = tape.gradient(gravity_loss, variables)
    weighted_gradients = tape.gradient(weighted_gravity_loss, variables)
    del tape

    def gradient_norm(values: list[tf.Tensor | None]) -> float:
        finite_values = [value for value in values if value is not None]
        return float(tf.linalg.global_norm(finite_values).numpy())

    density_gradient_norm = gradient_norm(density_gradients)
    gravity_gradient_norm = gradient_norm(gravity_gradients)
    weighted_gradient_norm = gradient_norm(weighted_gradients)

    diagnostic_inversion = tf.keras.models.clone_model(model.inversion_model)
    diagnostic_inversion.set_weights(model.inversion_model.get_weights())
    diagnostic_model = PhysicsConsistencyTrainingModel(
        diagnostic_inversion,
        DifferentiableSinglePlaneGz(),
        gravity_scale=model.gravity_scale,
        gravity_loss_weight=model.gravity_loss_weight,
        body_loss_fraction=model.density_loss_function.body_fraction,
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
    return {
        "density_loss": float(density_loss.numpy()),
        "gravity_loss": float(gravity_loss.numpy()),
        "weighted_gravity_loss": float(weighted_gravity_loss.numpy()),
        "total_loss": float(terms[4].numpy()),
        "density_gradient_norm": density_gradient_norm,
        "gravity_gradient_norm": gravity_gradient_norm,
        "weighted_gravity_gradient_norm": weighted_gradient_norm,
        "weighted_gravity_to_density_gradient_ratio": ratio,
        "prediction_mean_before": float(tf.reduce_mean(before).numpy()),
        "prediction_mean_after": float(tf.reduce_mean(after).numpy()),
        "prediction_max_before": float(tf.reduce_max(before).numpy()),
        "prediction_max_after": float(tf.reduce_max(after).numpy()),
        "body_prediction_mean_before": body_mean(before),
        "body_prediction_mean_after": body_mean(after),
    }


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
            if config.architecture.startswith("single_plane_2d3d")
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

    if config.architecture == "single_plane_2d3d_learned_depth_seed":
        model = build_single_plane_learned_depth_seed_model(model_config)
    elif config.architecture == "single_plane_2d3d":
        model = build_single_plane_model(model_config)
    else:
        model = build_baseline_model(model_config)
    inversion_model = model
    physics_training = config.gravity_loss_weight > 0.0
    pretraining_loss_scales: dict[str, float] | None = None
    if physics_training:
        model = PhysicsConsistencyTrainingModel(
            inversion_model,
            DifferentiableSinglePlaneGz(),
            gravity_scale=resolved_gravity_scale,
            gravity_loss_weight=config.gravity_loss_weight,
            body_loss_fraction=config.body_loss_fraction,
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
        (
            output_directory / "pretraining_loss_scales.json"
        ).write_text(
            json.dumps(
                {
                    **pretraining_loss_scales,
                    "gravity_loss_weight": config.gravity_loss_weight,
                    "maximum_permitted_weighted_gradient_ratio": 5.0,
                    "gradient_safety_passed": not gradient_failed,
                    "one_step_collapse_detected": (
                        mean_collapsed or maximum_collapsed
                    ),
                    "preflight_passed": not (
                        gradient_failed or mean_collapsed or maximum_collapsed
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if gradient_failed or mean_collapsed or maximum_collapsed:
            raise RuntimeError(
                "E07 pre-training safety check failed: weighted gravity "
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
        if physics_training
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
