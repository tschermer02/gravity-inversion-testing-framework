from __future__ import annotations

import numpy as np
import numpy.typing as npt

from forward_modeling.matlab_fwd3d.grid import ModelGrid
from forward_modeling.matlab_fwd3d.receivers import ReceiverGrid


FloatArray = npt.NDArray[np.float64]


MATLAB_EXAMPLE_CHANNELS = (
    4,
    5,
    6,
    7,
    8,
    9,
    10,
)


def build_matlab_example_grid() -> ModelGrid:
    """
    Build the exact gravity-model grid from the supplied ``inpt3.m``.

    Returns
    -------
    ModelGrid
        A 20 x 20 x 5 model grid.
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


def build_matlab_example_density(
    grid: ModelGrid,
) -> FloatArray:
    """
    Build the uniform 1 g/cm3 MATLAB example density model.
    """

    return np.ones(
        grid.model_shape,
        dtype=np.float64,
    )


def build_matlab_example_receivers() -> ReceiverGrid:
    """
    Build the exact 11 x 11 receiver plane from ``inpt3.m``.

    Receivers are located 30 meters above the surface, represented as
    Z = -30 m under the positive-downward convention.
    """

    return ReceiverGrid.from_ranges(
        x_min=-250.0,
        x_max=250.0,
        x_step=50.0,
        y_min=-250.0,
        y_max=250.0,
        y_step=50.0,
        z_values=(-30.0,),
    )


def build_example_3d_receivers() -> ReceiverGrid:
    """
    Build an initial multi-elevation receiver volume.

    This is not part of the original MATLAB output. It is the first
    extension used after the one-plane MATLAB validation passes.
    """

    return ReceiverGrid.from_ranges(
        x_min=-250.0,
        x_max=250.0,
        x_step=50.0,
        y_min=-250.0,
        y_max=250.0,
        y_step=50.0,
        z_values=(
            -10.0,
            -20.0,
            -30.0,
            -40.0,
            -50.0,
        ),
    )