from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

from cnn_inversion_3d.dataset import (
    DatasetLoaderConfig,
    build_dataset,
    build_training_datasets,
    load_npz_sample,
    read_manifest_paths,
)


DATASET_DIRECTORY = Path(
    "datasets/fwd3d_matlab_edge_rectangular_baseline"
)


def first_batch(
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the first gravity and density batch as NumPy arrays.

    Parameters
    ----------
    dataset
        TensorFlow dataset yielding gravity-density pairs.

    Returns
    -------
    tuple of numpy.ndarray
        Gravity batch and density batch.
    """

    gravity_batch, density_batch = next(
        iter(dataset)
    )

    return (
        gravity_batch.numpy(),
        density_batch.numpy(),
    )


def load_first_manifest_sample_path(
    manifest_name: str,
) -> Path:
    """
    Return the first sample path listed in a manifest.

    Parameters
    ----------
    manifest_name
        Manifest filename inside the dataset directory.

    Returns
    -------
    pathlib.Path
        Absolute path to the first listed sample.
    """

    manifest_path = (
        DATASET_DIRECTORY
        / manifest_name
    )

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        row = next(
            csv.DictReader(
                manifest_file
            )
        )

    return (
        DATASET_DIRECTORY
        / row["relative_path"]
    ).resolve()


def test_global_gravity_scale_is_applied_to_all_splits() -> None:
    """
    Verify that one global scale is accepted by every dataset split.

    This test checks shapes, finite values, and split counts. Exact
    numerical scaling is tested separately using an unshuffled dataset.
    """

    gravity_scale = 0.01

    (
        training_dataset,
        validation_dataset,
        test_dataset,
        split_counts,
    ) = build_training_datasets(
        dataset_directory=DATASET_DIRECTORY,
        batch_size=1,
        gravity_scale=gravity_scale,
        density_scale=1.0,
        random_seed=20260727,
    )

    assert split_counts == {
        "train": 800,
        "validation": 100,
        "test": 100,
    }

    for dataset in (
        training_dataset,
        validation_dataset,
        test_dataset,
    ):
        gravity_batch, density_batch = first_batch(
            dataset
        )

        assert gravity_batch.shape == (
            1,
            8,
            64,
            64,
            1,
        )

        assert density_batch.shape == (
            1,
            24,
            64,
            64,
            1,
        )

        assert np.all(
            np.isfinite(
                gravity_batch
            )
        )

        assert np.all(
            np.isfinite(
                density_batch
            )
        )


def test_loader_divides_gravity_by_requested_scale() -> None:
    """
    Verify exact global gravity scaling using an unshuffled split.

    The dataset is deliberately built with ``shuffle=False`` so its first
    TensorFlow sample corresponds to the first row of the manifest.
    """

    gravity_scale = 0.02

    training_dataset, training_count = build_dataset(
        dataset_directory=DATASET_DIRECTORY,
        manifest_name="train_manifest.csv",
        config=DatasetLoaderConfig(
            batch_size=1,
            shuffle=False,
            cache=False,
            prefetch=False,
            gravity_scale=gravity_scale,
            density_scale=1.0,
        ),
    )

    assert training_count == 800

    normalized_gravity, normalized_density = first_batch(
        training_dataset
    )

    sample_path = load_first_manifest_sample_path(
        "train_manifest.csv"
    )

    raw_gravity, raw_density = load_npz_sample(
        sample_path
    )

    expected_gravity = (
        raw_gravity
        / np.float32(
            gravity_scale
        )
    )

    np.testing.assert_allclose(
        normalized_gravity[0],
        expected_gravity,
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    np.testing.assert_allclose(
        normalized_density[0],
        raw_density,
        rtol=0.0,
        atol=0.0,
    )


def test_validation_loader_uses_same_global_scale() -> None:
    """
    Verify exact scaling for the validation split.
    """

    gravity_scale = 0.02

    validation_dataset, validation_count = build_dataset(
        dataset_directory=DATASET_DIRECTORY,
        manifest_name="validation_manifest.csv",
        config=DatasetLoaderConfig(
            batch_size=1,
            shuffle=False,
            cache=False,
            prefetch=False,
            gravity_scale=gravity_scale,
            density_scale=1.0,
        ),
    )

    assert validation_count == 100

    normalized_gravity, normalized_density = first_batch(
        validation_dataset
    )

    sample_path = load_first_manifest_sample_path(
        "validation_manifest.csv"
    )

    raw_gravity, raw_density = load_npz_sample(
        sample_path
    )

    np.testing.assert_allclose(
        normalized_gravity[0],
        raw_gravity / np.float32(gravity_scale),
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    np.testing.assert_allclose(
        normalized_density[0],
        raw_density,
        rtol=0.0,
        atol=0.0,
    )


def test_test_loader_uses_same_global_scale() -> None:
    """
    Verify exact scaling for the test split.
    """

    gravity_scale = 0.02

    test_dataset, test_count = build_dataset(
        dataset_directory=DATASET_DIRECTORY,
        manifest_name="test_manifest.csv",
        config=DatasetLoaderConfig(
            batch_size=1,
            shuffle=False,
            cache=False,
            prefetch=False,
            gravity_scale=gravity_scale,
            density_scale=1.0,
        ),
    )

    assert test_count == 100

    normalized_gravity, normalized_density = first_batch(
        test_dataset
    )

    sample_path = load_first_manifest_sample_path(
        "test_manifest.csv"
    )

    raw_gravity, raw_density = load_npz_sample(
        sample_path
    )

    np.testing.assert_allclose(
        normalized_gravity[0],
        raw_gravity / np.float32(gravity_scale),
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    np.testing.assert_allclose(
        normalized_density[0],
        raw_density,
        rtol=0.0,
        atol=0.0,
    )


def test_loader_does_not_use_per_sample_maximum_scaling() -> None:
    """
    Verify that normalization preserves amplitude differences.

    Dividing every sample by one common constant preserves the ratio
    between their maximum gravity amplitudes. Per-sample maximum
    normalization would instead make both maxima approximately one.
    """

    gravity_scale = 0.02

    sample_paths = read_manifest_paths(
        dataset_directory=(
            DATASET_DIRECTORY.resolve()
        ),
        manifest_name="train_manifest.csv",
    )

    first_raw_gravity, _ = load_npz_sample(
        sample_paths[0]
    )

    second_raw_gravity, _ = load_npz_sample(
        sample_paths[1]
    )

    raw_first_maximum = float(
        np.max(
            np.abs(
                first_raw_gravity
            )
        )
    )

    raw_second_maximum = float(
        np.max(
            np.abs(
                second_raw_gravity
            )
        )
    )

    assert raw_first_maximum > 0.0
    assert raw_second_maximum > 0.0

    dataset, _ = build_dataset(
        dataset_directory=DATASET_DIRECTORY,
        manifest_name="train_manifest.csv",
        config=DatasetLoaderConfig(
            batch_size=2,
            shuffle=False,
            cache=False,
            prefetch=False,
            gravity_scale=gravity_scale,
            density_scale=1.0,
        ),
    )

    normalized_gravity, _ = first_batch(
        dataset
    )

    normalized_first_maximum = float(
        np.max(
            np.abs(
                normalized_gravity[0]
            )
        )
    )

    normalized_second_maximum = float(
        np.max(
            np.abs(
                normalized_gravity[1]
            )
        )
    )

    np.testing.assert_allclose(
        normalized_first_maximum,
        raw_first_maximum / gravity_scale,
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    np.testing.assert_allclose(
        normalized_second_maximum,
        raw_second_maximum / gravity_scale,
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    raw_amplitude_ratio = (
        raw_first_maximum
        / raw_second_maximum
    )

    normalized_amplitude_ratio = (
        normalized_first_maximum
        / normalized_second_maximum
    )

    np.testing.assert_allclose(
        normalized_amplitude_ratio,
        raw_amplitude_ratio,
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    assert not (
        np.isclose(
            normalized_first_maximum,
            1.0,
        )
        and np.isclose(
            normalized_second_maximum,
            1.0,
        )
    )


def test_density_scale_is_applied_exactly_once() -> None:
    """
    Verify density normalization independently of gravity scaling.
    """

    gravity_scale = 0.02
    density_scale = 2.0

    dataset, _ = build_dataset(
        dataset_directory=DATASET_DIRECTORY,
        manifest_name="train_manifest.csv",
        config=DatasetLoaderConfig(
            batch_size=1,
            shuffle=False,
            cache=False,
            prefetch=False,
            gravity_scale=gravity_scale,
            density_scale=density_scale,
        ),
    )

    normalized_gravity, normalized_density = first_batch(
        dataset
    )

    sample_path = load_first_manifest_sample_path(
        "train_manifest.csv"
    )

    raw_gravity, raw_density = load_npz_sample(
        sample_path
    )

    np.testing.assert_allclose(
        normalized_gravity[0],
        raw_gravity / np.float32(gravity_scale),
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    np.testing.assert_allclose(
        normalized_density[0],
        raw_density / np.float32(density_scale),
        rtol=1.0e-6,
        atol=1.0e-7,
    )