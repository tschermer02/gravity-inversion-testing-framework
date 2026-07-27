from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf
import math


GRAVITY_SHAPE = (
    8,
    64,
    64,
    1,
)

DENSITY_SHAPE = (
    24,
    64,
    64,
    1,
)


@dataclass(frozen=True)
class DatasetLoaderConfig:
    """
    Configuration for loading gravity-density training pairs.

    Parameters
    ----------
    batch_size
        Number of examples in each TensorFlow batch.
    shuffle
        Whether to shuffle the sample paths.
    shuffle_seed
        Deterministic seed used when shuffling.
    cache
        Whether to cache decoded samples in memory.
    prefetch
        Whether TensorFlow should prepare future batches while the
        current batch is being processed.
    gravity_scale
        Constant used to normalize gravity. A value of one leaves the
        gravity data unchanged.
    density_scale
        Constant used to normalize density. The current baseline density
        contrast is at most 1.0 g/cm3, so the default leaves it unchanged.
    """

    batch_size: int = 2
    shuffle: bool = False
    shuffle_seed: int = 20260727
    cache: bool = False
    prefetch: bool = True

    gravity_scale: float = 1.0
    density_scale: float = 1.0

    def validate(self) -> None:
        """Validate loader settings."""

        if self.batch_size < 1:
            raise ValueError(
                "batch_size must be at least one."
            )

        if self.gravity_scale <= 0.0:
            raise ValueError(
                "gravity_scale must be greater than zero."
            )

        if self.density_scale <= 0.0:
            raise ValueError(
                "density_scale must be greater than zero."
            )


def find_repository_root() -> Path:
    """
    Return the repository root.
    """

    return Path(__file__).resolve().parents[1]


def resolve_dataset_directory(
    dataset_directory: str | Path,
) -> Path:
    """
    Resolve a dataset path from the repository root.

    Parameters
    ----------
    dataset_directory
        Absolute dataset path or a path relative to the repository root.

    Returns
    -------
    pathlib.Path
        Resolved dataset directory.
    """

    path = Path(
        dataset_directory
    )

    if not path.is_absolute():
        path = (
            find_repository_root()
            / path
        )

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            "Dataset directory does not exist:\n"
            f"{path}"
        )

    return path


def read_manifest_paths(
    *,
    dataset_directory: Path,
    manifest_name: str,
) -> list[Path]:
    """
    Read sample paths from one split manifest.

    Parameters
    ----------
    dataset_directory
        Root directory of the generated dataset.
    manifest_name
        Manifest filename, such as ``train_manifest.csv``.

    Returns
    -------
    list of pathlib.Path
        Absolute sample paths in manifest order.
    """

    manifest_path = (
        dataset_directory
        / manifest_name
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Split manifest does not exist:\n"
            f"{manifest_path}"
        )

    sample_paths: list[Path] = []

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        reader = csv.DictReader(
            manifest_file
        )

        if reader.fieldnames is None:
            raise ValueError(
                f"{manifest_name} contains no header."
            )

        if "relative_path" not in reader.fieldnames:
            raise ValueError(
                f"{manifest_name} does not contain a "
                "'relative_path' column."
            )

        for row in reader:
            relative_path = row.get(
                "relative_path"
            )

            if relative_path is None:
                raise ValueError(
                    f"{manifest_name} contains a row without "
                    "a relative path."
                )

            sample_path = (
                dataset_directory
                / relative_path
            ).resolve()

            if not sample_path.exists():
                raise FileNotFoundError(
                    "Sample referenced by the manifest "
                    "does not exist:\n"
                    f"{sample_path}"
                )

            sample_paths.append(
                sample_path
            )

    if not sample_paths:
        raise ValueError(
            f"{manifest_name} contains no sample rows."
        )

    return sample_paths


def load_npz_sample(
    sample_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one gravity-density pair from disk.

    The channel dimension is appended here so TensorFlow receives
    channels-last 3D tensors:

    - gravity: ``(8, 64, 64, 1)``
    - density: ``(24, 64, 64, 1)``

    Parameters
    ----------
    sample_path
        Path to one generated ``.npz`` sample.

    Returns
    -------
    tuple of numpy.ndarray
        Gravity input and density target as float32 arrays.
    """

    path = Path(
        sample_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Sample does not exist:\n{path}"
        )

    with np.load(
        path
    ) as sample:
        if "gravity" not in sample:
            raise KeyError(
                f"{path.name} has no 'gravity' array."
            )

        if "density" not in sample:
            raise KeyError(
                f"{path.name} has no 'density' array."
            )

        gravity = np.asarray(
            sample["gravity"],
            dtype=np.float32,
        )

        density = np.asarray(
            sample["density"],
            dtype=np.float32,
        )

    expected_gravity_without_channel = (
        GRAVITY_SHAPE[:-1]
    )

    expected_density_without_channel = (
        DENSITY_SHAPE[:-1]
    )

    if (
        gravity.shape
        != expected_gravity_without_channel
    ):
        raise ValueError(
            f"{path.name}: expected gravity shape "
            f"{expected_gravity_without_channel}, "
            f"received {gravity.shape}."
        )

    if (
        density.shape
        != expected_density_without_channel
    ):
        raise ValueError(
            f"{path.name}: expected density shape "
            f"{expected_density_without_channel}, "
            f"received {density.shape}."
        )

    if not np.all(
        np.isfinite(gravity)
    ):
        raise ValueError(
            f"{path.name}: gravity contains invalid values."
        )

    if not np.all(
        np.isfinite(density)
    ):
        raise ValueError(
            f"{path.name}: density contains invalid values."
        )

    gravity = gravity[
        ...,
        np.newaxis,
    ]

    density = density[
        ...,
        np.newaxis,
    ]

    return (
        np.ascontiguousarray(
            gravity,
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            density,
            dtype=np.float32,
        ),
    )


def _sample_generator(
    *,
    sample_paths: list[Path],
    gravity_scale: float,
    density_scale: float,
) -> Iterator[
    tuple[np.ndarray, np.ndarray]
]:
    """
    Yield normalized samples for TensorFlow.
    """

    for sample_path in sample_paths:
        gravity, density = load_npz_sample(
            sample_path
        )

        gravity = (
            gravity
            / np.float32(
                gravity_scale
            )
        )

        density = (
            density
            / np.float32(
                density_scale
            )
        )

        yield (
            np.ascontiguousarray(
                gravity,
                dtype=np.float32,
            ),
            np.ascontiguousarray(
                density,
                dtype=np.float32,
            ),
        )


def build_dataset(
    *,
    dataset_directory: str | Path,
    manifest_name: str,
    config: DatasetLoaderConfig,
) -> tuple[tf.data.Dataset, int]:
    """
    Build one TensorFlow dataset from a split manifest.

    Parameters
    ----------
    dataset_directory
        Root directory of the generated dataset.
    manifest_name
        Split manifest filename.
    config
        Dataset loading options.

    Returns
    -------
    tuple
        TensorFlow dataset and number of unbatched samples.
    """

    config.validate()

    resolved_directory = (
        resolve_dataset_directory(
            dataset_directory
        )
    )

    sample_paths = read_manifest_paths(
        dataset_directory=resolved_directory,
        manifest_name=manifest_name,
    )

    output_signature = (
        tf.TensorSpec(
            shape=GRAVITY_SHAPE,
            dtype=tf.float32,
            name="gravity",
        ),
        tf.TensorSpec(
            shape=DENSITY_SHAPE,
            dtype=tf.float32,
            name="density",
        ),
    )

    dataset = (
        tf.data.Dataset.from_generator(
            lambda: _sample_generator(
                sample_paths=sample_paths,
                gravity_scale=(
                    config.gravity_scale
                ),
                density_scale=(
                    config.density_scale
                ),
            ),
            output_signature=output_signature,
        )
    )

    if config.shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(
                sample_paths
            ),
            seed=config.shuffle_seed,
            reshuffle_each_iteration=True,
        )

    if config.cache:
        dataset = dataset.cache()

    dataset = dataset.batch(
            config.batch_size,
            drop_remainder=False,
        )

    number_of_batches = math.ceil(
         len(sample_paths)
        / config.batch_size
    )

    dataset = dataset.apply(
        tf.data.experimental.assert_cardinality(
            number_of_batches
        )
     )

    if config.prefetch:
        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

    return (
        dataset,
        len(sample_paths),
    )


def build_training_datasets(
    *,
    dataset_directory: str | Path,
    batch_size: int,
    gravity_scale: float = 1.0,
    density_scale: float = 1.0,
    random_seed: int = 20260727,
) -> tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    tf.data.Dataset,
    dict[str, int],
]:
    """
    Build training, validation, and test TensorFlow datasets.

    Returns
    -------
    tuple
        Training dataset, validation dataset, test dataset, and sample
        counts for each split.
    """

    training_dataset, training_count = (
        build_dataset(
            dataset_directory=dataset_directory,
            manifest_name=(
                "train_manifest.csv"
            ),
            config=DatasetLoaderConfig(
                batch_size=batch_size,
                shuffle=True,
                shuffle_seed=random_seed,
                gravity_scale=gravity_scale,
                density_scale=density_scale,
            ),
        )
    )

    validation_dataset, validation_count = (
        build_dataset(
            dataset_directory=dataset_directory,
            manifest_name=(
                "validation_manifest.csv"
            ),
            config=DatasetLoaderConfig(
                batch_size=batch_size,
                shuffle=False,
                gravity_scale=gravity_scale,
                density_scale=density_scale,
            ),
        )
    )

    test_dataset, test_count = (
        build_dataset(
            dataset_directory=dataset_directory,
            manifest_name=(
                "test_manifest.csv"
            ),
            config=DatasetLoaderConfig(
                batch_size=batch_size,
                shuffle=False,
                gravity_scale=gravity_scale,
                density_scale=density_scale,
            ),
        )
    )

    counts = {
        "train": training_count,
        "validation": validation_count,
        "test": test_count,
    }

    return (
        training_dataset,
        validation_dataset,
        test_dataset,
        counts,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the loader smoke-test arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Load the FWD3D dataset splits and inspect one "
            "TensorFlow batch."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "datasets/fwd3d_smoke_test"
        ),
        help=(
            "Dataset directory. Relative paths are interpreted "
            "from the repository root."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="TensorFlow batch size.",
    )

    return parser


def main() -> None:
    """
    Run a TensorFlow dataset-loader smoke test.
    """

    parser = build_argument_parser()
    arguments = parser.parse_args()

    (
        training_dataset,
        validation_dataset,
        test_dataset,
        counts,
    ) = build_training_datasets(
        dataset_directory=arguments.dataset,
        batch_size=arguments.batch_size,
    )

    training_batch = next(
        iter(
            training_dataset
        )
    )

    gravity_batch, density_batch = (
        training_batch
    )

    expected_gravity_rank = 5
    expected_density_rank = 5

    if len(
        gravity_batch.shape
    ) != expected_gravity_rank:
        raise AssertionError(
            "Gravity batch must have rank five."
        )

    if len(
        density_batch.shape
    ) != expected_density_rank:
        raise AssertionError(
            "Density batch must have rank five."
        )

    if tuple(
        gravity_batch.shape[1:]
    ) != GRAVITY_SHAPE:
        raise AssertionError(
            "Unexpected gravity sample shape: "
            f"{gravity_batch.shape[1:]}"
        )

    if tuple(
        density_batch.shape[1:]
    ) != DENSITY_SHAPE:
        raise AssertionError(
            "Unexpected density sample shape: "
            f"{density_batch.shape[1:]}"
        )

    if not bool(
        tf.reduce_all(
            tf.math.is_finite(
                gravity_batch
            )
        )
    ):
        raise AssertionError(
            "Gravity batch contains invalid values."
        )

    if not bool(
        tf.reduce_all(
            tf.math.is_finite(
                density_batch
            )
        )
    ):
        raise AssertionError(
            "Density batch contains invalid values."
        )

    print()
    print("TensorFlow dataset-loader test")
    print("=" * 30)
    print(
        f"Split counts: {counts}"
    )
    print(
        f"Gravity batch shape: "
        f"{gravity_batch.shape}"
    )
    print(
        f"Density batch shape: "
        f"{density_batch.shape}"
    )
    print(
        f"Gravity dtype: "
        f"{gravity_batch.dtype.name}"
    )
    print(
        f"Density dtype: "
        f"{density_batch.dtype.name}"
    )
    print(
        f"Gravity range in batch: "
        f"{float(tf.reduce_min(gravity_batch)):.8e} "
        f"to "
        f"{float(tf.reduce_max(gravity_batch)):.8e}"
    )
    print(
        f"Density range in batch: "
        f"{float(tf.reduce_min(density_batch)):.8e} "
        f"to "
        f"{float(tf.reduce_max(density_batch)):.8e}"
    )
    print()
    print("Training dataset load: PASSED")
    print("Validation dataset load: PASSED")
    print("Test dataset load: PASSED")
    print("Input shape check: PASSED")
    print("Target shape check: PASSED")
    print("Finite-value check: PASSED")

    # Make sure the other two datasets can also produce a batch.
    next(
        iter(
            validation_dataset
        )
    )

    next(
        iter(
            test_dataset
        )
    )


if __name__ == "__main__":
    main()