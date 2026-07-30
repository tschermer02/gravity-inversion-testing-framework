from __future__ import annotations

from pathlib import Path

import numpy as np

from forward_modeling.matlab_fwd3d.gravity import (
    calculate_gravity,
)
from forward_modeling.matlab_fwd3d.grid import (
    ModelGrid,
)


REPOSITORY_ROOT = Path(
    __file__
).resolve().parents[2]

MATLAB_OBSERVATION_PATH = (
    REPOSITORY_ROOT
    / "matlab_code"
    / "FWD3D"
    / "example1"
    / "FWD3D"
    / "obsData1.dat"
)


def build_matlab_reference_grid() -> ModelGrid:
    """
    Build the exact density grid from the original MATLAB example.

    MATLAB model bounds are physical cell edges:

    - X: -100 to 100 m
    - Y: -100 to 100 m
    - Z: 50 to 100 m

    Cell dimensions are 10 x 10 x 10 m.

    Returns
    -------
    ModelGrid
        MATLAB-compatible 20 x 20 x 5 model grid.
    """

    return ModelGrid.from_bounds(
        bounds=(
            -100.0,
            100.0,
            -100.0,
            100.0,
            50.0,
            100.0,
        ),
        dx=10.0,
        dy=10.0,
        dz=10.0,
    )


def load_matlab_gravity_reference(
    path: Path,
    *,
    channel: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load receiver coordinates and gravity values from MATLAB output.

    Parameters
    ----------
    path
        Path to MATLAB ``obsData1.dat``.
    channel
        Gravity channel to extract. Channel 4 is Gz.

    Returns
    -------
    tuple of numpy.ndarray
        Receiver coordinates with shape ``(N, 3)`` and reference gravity
        values with shape ``(N,)``.

    Raises
    ------
    FileNotFoundError
        If the MATLAB reference file does not exist.
    ValueError
        If the file has an unexpected format or contains no requested
        channel rows.
    """

    if not path.exists():
        raise FileNotFoundError(
            "MATLAB gravity reference file was not found:\n"
            f"{path}"
        )

    reference_table = np.loadtxt(
        path,
        dtype=np.float64,
    )

    if (
        reference_table.ndim != 2
        or reference_table.shape[1] < 5
    ):
        raise ValueError(
            "Expected the MATLAB reference file to contain at least "
            "five columns: X, Y, Z, channel, and calculated value."
        )

    channel_mask = (
        reference_table[:, 3]
        == float(channel)
    )

    if not np.any(channel_mask):
        raise ValueError(
            f"MATLAB reference contains no channel {channel} rows."
        )

    channel_rows = reference_table[
        channel_mask
    ]

    receiver_points = np.ascontiguousarray(
        channel_rows[:, 0:3],
        dtype=np.float64,
    )

    reference_values = np.ascontiguousarray(
        channel_rows[:, 4],
        dtype=np.float64,
    )

    return (
        receiver_points,
        reference_values,
    )


def test_matlab_reference_file_structure() -> None:
    """
    Verify the original MATLAB gravity-output structure.
    """

    reference_table = np.loadtxt(
        MATLAB_OBSERVATION_PATH,
        dtype=np.float64,
    )

    assert reference_table.shape == (
        847,
        5,
    )

    channels = np.unique(
        reference_table[:, 3]
    )

    np.testing.assert_array_equal(
        channels,
        np.array(
            [
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
            ],
            dtype=np.float64,
        ),
    )


def test_matlab_reference_gz_receiver_geometry() -> None:
    """
    Verify the receiver coordinates used for MATLAB Gz.
    """

    receiver_points, _ = (
        load_matlab_gravity_reference(
            MATLAB_OBSERVATION_PATH,
            channel=4,
        )
    )

    assert receiver_points.shape == (
        121,
        3,
    )

    np.testing.assert_array_equal(
        np.unique(
            receiver_points[:, 0]
        ),
        np.arange(
            -250.0,
            251.0,
            50.0,
            dtype=np.float64,
        ),
    )

    np.testing.assert_array_equal(
        np.unique(
            receiver_points[:, 1]
        ),
        np.arange(
            -250.0,
            251.0,
            50.0,
            dtype=np.float64,
        ),
    )

    np.testing.assert_array_equal(
        np.unique(
            receiver_points[:, 2]
        ),
        np.array(
            [-30.0],
            dtype=np.float64,
        ),
    )


def test_python_gz_matches_full_matlab_receiver_plane() -> None:
    """
    Compare Python Gz against all 121 original MATLAB receiver values.
    """

    grid = build_matlab_reference_grid()

    density = np.ones(
        grid.model_shape,
        dtype=np.float64,
    )

    receiver_points, reference_values = (
        load_matlab_gravity_reference(
            MATLAB_OBSERVATION_PATH,
            channel=4,
        )
    )

    result = calculate_gravity(
        model=density,
        model_grid=grid,
        receiver_points=receiver_points,
        channels=(4,),
        receiver_chunk_size=32,
    )

    python_values = result.channel_values(
        4
    )

    assert python_values.shape == (
        121,
    )

    absolute_error = np.abs(
        python_values
        - reference_values
    )

    maximum_absolute_error = float(
        np.max(
            absolute_error
        )
    )

    mean_absolute_error = float(
        np.mean(
            absolute_error
        )
    )

    print()
    print(
        "Maximum absolute error:",
        f"{maximum_absolute_error:.12e}",
    )
    print(
        "Mean absolute error:",
        f"{mean_absolute_error:.12e}",
    )

    np.testing.assert_allclose(
        python_values,
        reference_values,
        rtol=1.0e-7,
        atol=1.0e-8,
    )


def test_python_gz_matches_matlab_central_receiver() -> None:
    """
    Compare the Python and MATLAB Gz values at X=0, Y=0, Z=-30 m.
    """

    grid = build_matlab_reference_grid()

    density = np.ones(
        grid.model_shape,
        dtype=np.float64,
    )

    receiver_points, reference_values = (
        load_matlab_gravity_reference(
            MATLAB_OBSERVATION_PATH,
            channel=4,
        )
    )

    central_mask = (
        np.isclose(
            receiver_points[:, 0],
            0.0,
        )
        & np.isclose(
            receiver_points[:, 1],
            0.0,
        )
        & np.isclose(
            receiver_points[:, 2],
            -30.0,
        )
    )

    assert np.count_nonzero(
        central_mask
    ) == 1

    central_receiver = receiver_points[
        central_mask
    ]

    matlab_value = float(
        reference_values[
            central_mask
        ][0]
    )

    result = calculate_gravity(
        model=density,
        model_grid=grid,
        receiver_points=central_receiver,
        channels=(4,),
        receiver_chunk_size=1,
    )

    python_value = float(
        result.channel_values(
            4
        )[0]
    )

    absolute_error = abs(
        python_value
        - matlab_value
    )

    print()
    print(
        "MATLAB central Gz:",
        f"{matlab_value:.12f}",
    )
    print(
        "Python central Gz:",
        f"{python_value:.12f}",
    )
    print(
        "Absolute error:",
        f"{absolute_error:.12e}",
    )

    assert absolute_error < 1.0e-8