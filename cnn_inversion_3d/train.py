from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import tensorflow as tf

from cnn_inversion_3d.dataset import (
    build_training_datasets,
    find_repository_root,
)
from cnn_inversion_3d.model import (
    ModelConfig,
    build_baseline_model,
    compile_baseline_model,
    count_trainable_parameters,
)


@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration for baseline 3D CNN training.

    The initial smoke test intentionally uses:

    - no explicit regularization
    - no dropout
    - no physics-based loss
    - no observational noise
    - plain mean squared error
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
    density_scale: float = 1.0

    random_seed: int = 20260727

    base_filters: int = 8
    learning_rate: float = 1.0e-3
    output_activation: str = "sigmoid"

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
        "--learning-rate",
        type=float,
        default=None,
        help="Adam learning rate.",
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

    if arguments.learning_rate is not None:
        values["learning_rate"] = (
            arguments.learning_rate
        )

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
) -> None:
    """Save training metadata as JSON."""

    metadata: dict[str, Any] = {
        "training_configuration": {
            **asdict(config),
            "dataset_directory": str(
                config.dataset_directory
            ),
            "output_directory": str(
                config.output_directory
            ),
        },
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
        "loss": "mean_squared_error",
        "explicit_regularization": None,
        "observational_noise": None,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=2,
        )


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

    prepare_output_directory(
        output_directory=output_directory,
        overwrite=config.overwrite,
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
        gravity_scale=config.gravity_scale,
        density_scale=config.density_scale,
        random_seed=config.random_seed,
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
    )

    model = build_baseline_model(
        model_config
    )

    compile_baseline_model(
        model,
        model_config,
    )

    best_model_path = (
        output_directory
        / "best_model.keras"
    )

    final_model_path = (
        output_directory
        / "final_model.keras"
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            mode="min",
            restore_best_weights=False,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
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
        f"Trainable parameters: "
        f"{count_trainable_parameters(model):,}"
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

    model.save(
        final_model_path
    )

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
        gravity_scale=config.gravity_scale,
        density_scale=config.density_scale,
        random_seed=config.random_seed,
        base_filters=config.base_filters,
        learning_rate=config.learning_rate,
        output_activation=(
            config.output_activation
        ),
        overwrite=config.overwrite,
        body_loss_fraction=(
            config.body_loss_fraction
        ),
    )

    save_training_metadata(
        output_path=metadata_path,
        config=resolved_config,
        model=best_model,
        split_counts=split_counts,
        elapsed_seconds=elapsed_seconds,
        final_metrics=final_metrics,
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