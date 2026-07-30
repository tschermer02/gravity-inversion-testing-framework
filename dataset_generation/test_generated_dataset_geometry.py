from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from dataset_generation.matlab_grid import MatlabCompatibleGridSpec


DATASET_DIRECTORY = Path(
    "datasets/fwd3d_matlab_edge_smoke_test"
)


def load_manifest_rows(
    manifest_path: Path,
) -> list[dict[str, str]]:
    """
    Load all rows from a generated dataset manifest.

    Parameters
    ----------
    manifest_path
        Path to the dataset manifest.

    Returns
    -------
    list of dict
        Manifest rows.
    """

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        return list(
            csv.DictReader(
                manifest_file
            )
        )


def test_smoke_test_dataset_exists() -> None:
    """
    Verify that the generated smoke-test dataset is present.
    """

    assert DATASET_DIRECTORY.exists()

    assert (
        DATASET_DIRECTORY
        / "manifest.csv"
    ).exists()

    assert (
        DATASET_DIRECTORY
        / "metadata.json"
    ).exists()

    assert (
        DATASET_DIRECTORY
        / "samples"
    ).exists()


def test_manifest_contains_five_samples() -> None:
    """
    Verify the smoke-test manifest length.
    """

    rows = load_manifest_rows(
        DATASET_DIRECTORY
        / "manifest.csv"
    )

    assert len(rows) == 5


def test_manifest_physical_depths_match_indices() -> None:
    """
    Verify physical depth metadata against MATLAB edge semantics.
    """

    grid = MatlabCompatibleGridSpec()

    rows = load_manifest_rows(
        DATASET_DIRECTORY
        / "manifest.csv"
    )

    for row in rows:
        z_start = int(
            row["z_start"]
        )

        z_end = int(
            row["z_end"]
        )

        top_depth_index = int(
            row["top_depth_index"]
        )

        bottom_depth_index = int(
            row["bottom_depth_index"]
        )

        top_depth_m = float(
            row["top_depth_m"]
        )

        bottom_depth_m = float(
            row["bottom_depth_m"]
        )

        center_depth_m = float(
            row["center_depth_m"]
        )

        thickness_z_m = float(
            row["thickness_z_m"]
        )

        assert top_depth_index == z_start
        assert bottom_depth_index == z_end - 1

        assert top_depth_m == (
            grid.z_edge_from_index(
                z_start
            )
        )

        assert bottom_depth_m == (
            grid.z_edge_from_index(
                z_end
            )
        )

        assert center_depth_m == (
            top_depth_m
            + bottom_depth_m
        ) / 2.0

        assert thickness_z_m == (
            bottom_depth_m
            - top_depth_m
        )


def test_saved_density_matches_manifest_geometry() -> None:
    """
    Verify that occupied density voxels match each manifest body.
    """

    rows = load_manifest_rows(
        DATASET_DIRECTORY
        / "manifest.csv"
    )

    expected_shape = (
        24,
        64,
        64,
    )

    for row in rows:
        sample_path = (
            DATASET_DIRECTORY
            / row["relative_path"]
        )

        with np.load(
            sample_path
        ) as sample:
            density = np.asarray(
                sample["density"],
                dtype=np.float32,
            )

        assert density.shape == expected_shape

        x_start = int(
            row["x_start"]
        )
        x_end = int(
            row["x_end"]
        )

        y_start = int(
            row["y_start"]
        )
        y_end = int(
            row["y_end"]
        )

        z_start = int(
            row["z_start"]
        )
        z_end = int(
            row["z_end"]
        )

        density_contrast = float(
            row["density_contrast"]
        )

        expected_mask = np.zeros(
            expected_shape,
            dtype=bool,
        )

        expected_mask[
            z_start:z_end,
            y_start:y_end,
            x_start:x_end,
        ] = True

        actual_mask = density != 0.0

        np.testing.assert_array_equal(
            actual_mask,
            expected_mask,
        )

        np.testing.assert_allclose(
            density[expected_mask],
            density_contrast,
            rtol=1.0e-6,
            atol=1.0e-7,
        )

        assert np.all(
            density[~expected_mask]
            == 0.0
        )


def test_saved_gravity_shapes_and_values() -> None:
    """
    Verify gravity-volume shapes and finite values.
    """

    rows = load_manifest_rows(
        DATASET_DIRECTORY
        / "manifest.csv"
    )

    expected_shape = (
        8,
        64,
        64,
    )

    for row in rows:
        sample_path = (
            DATASET_DIRECTORY
            / row["relative_path"]
        )

        with np.load(
            sample_path
        ) as sample:
            gravity = np.asarray(
                sample["gravity"],
                dtype=np.float32,
            )

        assert gravity.shape == expected_shape

        assert np.all(
            np.isfinite(
                gravity
            )
        )

        assert not np.all(
            gravity == 0.0
        )