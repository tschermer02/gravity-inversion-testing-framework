from __future__ import annotations

import numpy as np

from forward_modeling.matlab_fwd3d.grid import ModelGrid
from synthetic_models.common.grid import GridSpec


def model_grid_from_grid_spec(
    grid: GridSpec,
) -> ModelGrid:
    """
    Convert the repository's GridSpec into an FWD3D ModelGrid.

    The repository density model uses:

        model[z, y, x]

    GridSpec values are interpreted as cell-center coordinates. This matches
    the current repository convention, where a 64-cell X grid runs from
    x_min=0 to x_max=630 with dx=10.

    Parameters
    ----------
    grid
        Repository grid specification.

    Returns
    -------
    ModelGrid
        Equivalent model grid for the translated MATLAB FWD3D solver.
    """

    if grid.nx < 1:
        raise ValueError("grid.nx must be at least one.")

    if grid.ny < 1:
        raise ValueError("grid.ny must be at least one.")

    if grid.nz < 1:
        raise ValueError("grid.nz must be at least one.")

    if grid.dx <= 0.0:
        raise ValueError("grid.dx must be greater than zero.")

    if grid.dy <= 0.0:
        raise ValueError("grid.dy must be greater than zero.")

    if grid.dz <= 0.0:
        raise ValueError("grid.dz must be greater than zero.")

    x_centers = np.linspace(
        grid.x_min,
        grid.x_max,
        grid.nx,
        dtype=np.float64,
    )

    y_centers = np.linspace(
        grid.y_min,
        grid.y_max,
        grid.ny,
        dtype=np.float64,
    )

    z_centers = np.linspace(
        grid.z_min,
        grid.z_max,
        grid.nz,
        dtype=np.float64,
    )

    vertical_thicknesses = np.full(
        grid.nz,
        grid.dz,
        dtype=np.float64,
    )

    model_grid = ModelGrid(
        x_centers=x_centers,
        y_centers=y_centers,
        z_centers=z_centers,
        dx=grid.dx,
        dy=grid.dy,
        dz=vertical_thicknesses,
    )

    expected_shape = (
        grid.nz,
        grid.ny,
        grid.nx,
    )

    if model_grid.model_shape != expected_shape:
        raise RuntimeError(
            "Converted model-grid shape does not match GridSpec. "
            f"Expected {expected_shape}, "
            f"received {model_grid.model_shape}."
        )

    return model_grid