from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


def load_matlab_gravity_table(
    path: str | Path,
) -> FloatArray:
    """
    Load a MATLAB ``obsData1.dat`` gravity table.

    Expected columns are:

    ``X, Y, Z, channel_code, gravity_value``

    Parameters
    ----------
    path
        Path to the MATLAB output file.

    Returns
    -------
    numpy.ndarray
        Gravity table with five columns.
    """

    input_path = Path(path).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find MATLAB output file:\n{input_path}"
        )

    table = np.loadtxt(
        input_path,
        dtype=np.float64,
    )

    if table.ndim == 1:
        table = table.reshape(1, -1)

    if table.ndim != 2 or table.shape[1] < 5:
        raise ValueError(
            "MATLAB gravity output must contain at least five columns: "
            "X, Y, Z, channel, and value."
        )

    table = table[:, :5]

    if not np.all(np.isfinite(table)):
        raise ValueError(
            "MATLAB gravity output contains NaN or infinite values."
        )

    return table


def save_gravity_table(
    path: str | Path,
    table: npt.ArrayLike,
) -> Path:
    """
    Save a MATLAB-compatible gravity table.

    Parameters
    ----------
    path
        Destination path.
    table
        Table with columns X, Y, Z, channel, and value.

    Returns
    -------
    pathlib.Path
        Resolved output path.
    """

    output_path = Path(path).expanduser().resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_array = np.asarray(
        table,
        dtype=np.float64,
    )

    if (
        table_array.ndim != 2
        or table_array.shape[1] != 5
    ):
        raise ValueError(
            "Gravity table must have shape (N, 5)."
        )

    np.savetxt(
        output_path,
        table_array,
        fmt=[
            "% .7e",
            "% .7e",
            "% .7e",
            "% .7e",
            "% .7e",
        ],
    )

    return output_path


def sort_gravity_table(
    table: npt.ArrayLike,
) -> FloatArray:
    """
    Sort a gravity table by X, Y, Z, and channel.

    This makes Python and MATLAB tables directly comparable even if the
    original receiver enumeration differs.
    """

    table_array = np.asarray(
        table,
        dtype=np.float64,
    )

    if (
        table_array.ndim != 2
        or table_array.shape[1] < 5
    ):
        raise ValueError(
            "Gravity table must contain at least five columns."
        )

    sort_indices = np.lexsort(
        (
            table_array[:, 3],
            table_array[:, 2],
            table_array[:, 1],
            table_array[:, 0],
        )
    )

    return table_array[
        sort_indices
    ]