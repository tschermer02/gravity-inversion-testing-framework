from __future__ import annotations

import numpy as np
import numpy.typing as npt

from forward_modeling.matlab_fwd3d.gravity import (
    calculate_gravity_volume,
)
from forward_modeling.matlab_fwd3d.grid import ModelGrid
from forward_modeling.matlab_fwd3d.receivers import ReceiverGrid


FloatArray = npt.NDArray[np.float64]


class FWD3DGravityForwardModel:
    """
    Repository-facing interface to the translated MATLAB FWD3D solver.

    The input density model uses array order:

        model[z, y, x]

    The returned gravity volume uses array order:

        gravity[z_receiver, y_receiver, x_receiver]

    Parameters
    ----------
    model_grid
        Grid describing the three-dimensional density model.
    receiver_grid
        Grid describing the three-dimensional receiver locations.
    channel
        Gravity channel code. Channel 4 is Gz in mGal.
    receiver_chunk_size
        Number of receiver locations processed together.
    """

    def __init__(
        self,
        *,
        model_grid: ModelGrid,
        receiver_grid: ReceiverGrid,
        channel: int = 4,
        receiver_chunk_size: int = 128,
    ) -> None:
        if channel < 1 or channel > 11:
            raise ValueError(
                "channel must be a valid FWD3D gravity channel "
                "between 1 and 11."
            )

        if receiver_chunk_size < 1:
            raise ValueError(
                "receiver_chunk_size must be at least one."
            )

        self.model_grid = model_grid
        self.receiver_grid = receiver_grid
        self.channel = int(channel)
        self.receiver_chunk_size = int(
            receiver_chunk_size
        )

    @property
    def input_shape(
        self,
    ) -> tuple[int, int, int]:
        """
        Return the required density-model shape.

        The order is ``(z, y, x)``.
        """

        return self.model_grid.model_shape

    @property
    def output_shape(
        self,
    ) -> tuple[int, int, int]:
        """
        Return the gravity-volume shape.

        The order is ``(receiver_z, receiver_y, receiver_x)``.
        """

        return self.receiver_grid.volume_shape

    def calculate(
        self,
        model: npt.ArrayLike,
    ) -> FloatArray:
        """
        Calculate one gravity component on the receiver volume.

        Parameters
        ----------
        model
            Density model in g/cm3 with shape ``(nz, ny, nx)``.

        Returns
        -------
        numpy.ndarray
            Gravity volume with shape
            ``(receiver_nz, receiver_ny, receiver_nx)``.
        """

        model_array = np.asarray(
            model,
            dtype=np.float64,
        )

        if model_array.shape != self.input_shape:
            raise ValueError(
                f"Expected density model shape {self.input_shape}, "
                f"but received {model_array.shape}."
            )

        if not np.all(
            np.isfinite(model_array)
        ):
            raise ValueError(
                "Density model contains NaN or infinite values."
            )

        gravity_volume = calculate_gravity_volume(
            model=model_array,
            model_grid=self.model_grid,
            receiver_grid=self.receiver_grid,
            channel=self.channel,
            receiver_chunk_size=(
                self.receiver_chunk_size
            ),
        )

        if gravity_volume.shape != self.output_shape:
            raise RuntimeError(
                f"Expected gravity output shape {self.output_shape}, "
                f"but received {gravity_volume.shape}."
            )

        if not np.all(
            np.isfinite(gravity_volume)
        ):
            raise RuntimeError(
                "Calculated gravity volume contains "
                "NaN or infinite values."
            )

        return gravity_volume