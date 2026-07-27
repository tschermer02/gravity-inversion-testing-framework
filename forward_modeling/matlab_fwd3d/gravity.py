from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt

from forward_modeling.matlab_fwd3d.grid import ModelGrid
from forward_modeling.matlab_fwd3d.receivers import ReceiverGrid


FloatArray = npt.NDArray[np.float64]


GRAVITATIONAL_CONSTANT = 6.67e-11


GRAVITY_CHANNELS: dict[int, str] = {
    1: "Gt",
    2: "Gx",
    3: "Gy",
    4: "Gz",
    5: "Gxx",
    6: "Gyy",
    7: "Gzz",
    8: "Gxy",
    9: "Gzx",
    10: "Gzy",
    11: "Gd",
}


GRAVITY_CHANNEL_UNITS: dict[int, str] = {
    1: "mGal",
    2: "mGal",
    3: "mGal",
    4: "mGal",
    5: "Eotvos",
    6: "Eotvos",
    7: "Eotvos",
    8: "Eotvos",
    9: "Eotvos",
    10: "Eotvos",
    11: "Eotvos",
}


@dataclass(frozen=True)
class GravityForwardResult:
    """
    Gravity values calculated at arbitrary receiver points.

    Parameters
    ----------
    receiver_points
        Receiver coordinates with columns X, Y, and Z.
    channels
        Gravity channel codes.
    values
        Calculated gravity values with shape
        ``(number_of_receivers, number_of_channels)``.
    """

    receiver_points: FloatArray
    channels: tuple[int, ...]
    values: FloatArray

    def channel_values(
        self,
        channel: int,
    ) -> FloatArray:
        """
        Return all receiver values for one gravity channel.
        """

        try:
            channel_index = self.channels.index(channel)
        except ValueError as exc:
            raise KeyError(
                f"Channel {channel} was not calculated."
            ) from exc

        return self.values[:, channel_index]


def _validate_channels(
    channels: Iterable[int],
) -> tuple[int, ...]:
    """
    Validate gravity-channel codes.
    """

    normalized_channels = tuple(
        int(channel)
        for channel in channels
    )

    if not normalized_channels:
        raise ValueError(
            "At least one gravity channel must be requested."
        )

    if len(set(normalized_channels)) != len(
        normalized_channels
    ):
        raise ValueError(
            "Gravity channels must not contain duplicates."
        )

    invalid_channels = [
        channel
        for channel in normalized_channels
        if channel not in GRAVITY_CHANNELS
    ]

    if invalid_channels:
        raise ValueError(
            "Unsupported gravity channel codes: "
            f"{invalid_channels}."
        )

    return normalized_channels


def _validate_receiver_points(
    receiver_points: npt.ArrayLike,
) -> FloatArray:
    """
    Validate an arbitrary receiver-coordinate array.
    """

    receiver_array = np.asarray(
        receiver_points,
        dtype=np.float64,
    )

    if (
        receiver_array.ndim != 2
        or receiver_array.shape[1] != 3
    ):
        raise ValueError(
            "receiver_points must have shape "
            "(number_of_receivers, 3)."
        )

    if receiver_array.shape[0] == 0:
        raise ValueError(
            "At least one receiver point is required."
        )

    if not np.all(np.isfinite(receiver_array)):
        raise ValueError(
            "receiver_points contains NaN or infinite values."
        )

    return receiver_array


def calculate_gravity(
    *,
    model: npt.ArrayLike,
    model_grid: ModelGrid,
    receiver_points: npt.ArrayLike,
    channels: Iterable[int] = (4,),
    receiver_chunk_size: int = 128,
) -> GravityForwardResult:
    """
    Calculate gravity at arbitrary three-dimensional receiver locations.

    This reproduces the cell-centered pulse-basis calculation in the
    MATLAB ``getPredGR.m`` implementation.

    Parameters
    ----------
    model
        Density model with shape ``(nz, ny, nx)`` in g/cm3.
    model_grid
        Three-dimensional model grid.
    receiver_points
        Receiver coordinates with shape ``(N, 3)`` and columns
        ``[X, Y, Z]``.
    channels
        Gravity channels to calculate.

        Channel 4 is Gz in mGal. Channels 5 through 11 are gravity
        gradients in Eotvos.
    receiver_chunk_size
        Number of receiver locations processed simultaneously.

    Returns
    -------
    GravityForwardResult
        Receiver coordinates, channel codes, and calculated values.
    """

    if receiver_chunk_size < 1:
        raise ValueError(
            "receiver_chunk_size must be at least one."
        )

    requested_channels = _validate_channels(
        channels
    )

    receivers = _validate_receiver_points(
        receiver_points
    )

    density = model_grid.flatten_model(
        model
    )

    cell_x, cell_y, cell_z = (
        model_grid.flattened_cell_centers()
    )

    cell_volumes = (
        model_grid.flattened_cell_volumes()
    )

    # Synthetic models normally have zero background density contrast.
    # Cells with exactly zero contrast contribute nothing to the gravity
    # response, so removing them greatly reduces the computation.
    active_cells = density != 0.0

    if not np.any(active_cells):
        zero_values = np.zeros(
            (
                receivers.shape[0],
                len(requested_channels),
            ),
            dtype=np.float64,
        )

        return GravityForwardResult(
            receiver_points=receivers,
            channels=requested_channels,
            values=zero_values,
        )

    density = density[active_cells]
    cell_x = cell_x[active_cells]
    cell_y = cell_y[active_cells]
    cell_z = cell_z[active_cells]
    cell_volumes = cell_volumes[active_cells]

    gamma = (
        1.0e8
        * GRAVITATIONAL_CONSTANT
        * cell_volumes
    )

    number_of_receivers = receivers.shape[0]
    number_of_channels = len(requested_channels)

    predicted = np.empty(
        (
            number_of_receivers,
            number_of_channels,
        ),
        dtype=np.float64,
    )

    for start in range(
        0,
        number_of_receivers,
        receiver_chunk_size,
    ):
        stop = min(
            start + receiver_chunk_size,
            number_of_receivers,
        )

        receiver_chunk = receivers[start:stop]

        delta_x = (
            cell_x[np.newaxis, :]
            - receiver_chunk[:, 0, np.newaxis]
        )
        delta_y = (
            cell_y[np.newaxis, :]
            - receiver_chunk[:, 1, np.newaxis]
        )
        delta_z = (
            cell_z[np.newaxis, :]
            - receiver_chunk[:, 2, np.newaxis]
        )

        radius_squared = (
            delta_x**2
            + delta_y**2
            + delta_z**2
        )

        if np.any(radius_squared == 0.0):
            raise ValueError(
                "A receiver is located exactly at a model-cell center. "
                "The point-cell gravity approximation is singular there."
            )

        radius = np.sqrt(
            radius_squared
        )

        weighted_density = (
            gamma
            * density
        )[np.newaxis, :]

        for channel_index, channel in enumerate(
            requested_channels
        ):
            kernel = _calculate_channel_kernel(
                channel=channel,
                delta_x=delta_x,
                delta_y=delta_y,
                delta_z=delta_z,
                radius=radius,
                radius_squared=radius_squared,
            )

            predicted[
                start:stop,
                channel_index,
            ] = np.sum(
                weighted_density
                * kernel,
                axis=1,
            )

    if not np.all(np.isfinite(predicted)):
        raise RuntimeError(
            "The calculated gravity contains NaN or infinite values."
        )

    return GravityForwardResult(
        receiver_points=receivers,
        channels=requested_channels,
        values=predicted,
    )


def _calculate_channel_kernel(
    *,
    channel: int,
    delta_x: FloatArray,
    delta_y: FloatArray,
    delta_z: FloatArray,
    radius: FloatArray,
    radius_squared: FloatArray,
) -> FloatArray:
    """
    Return the MATLAB-equivalent kernel for one gravity channel.
    """

    radius_cubed = (
        radius
        * radius_squared
    )

    radius_fifth = (
        radius
        * radius_squared
        * radius_squared
    )

    if channel == 1:
        return 1.0 / radius_squared

    if channel == 2:
        return (
            delta_x
            / radius_cubed
        )

    if channel == 3:
        return (
            delta_y
            / radius_cubed
        )

    if channel == 4:
        return (
            delta_z
            / radius_cubed
        )

    if channel == 5:
        return (
            1.0e4
            * (
                2.0 * delta_x**2
                - delta_y**2
                - delta_z**2
            )
            / radius_fifth
        )

    if channel == 6:
        return (
            1.0e4
            * (
                2.0 * delta_y**2
                - delta_x**2
                - delta_z**2
            )
            / radius_fifth
        )

    if channel == 7:
        return (
            1.0e4
            * (
                2.0 * delta_z**2
                - delta_x**2
                - delta_y**2
            )
            / radius_fifth
        )

    if channel == 8:
        return (
            3.0e4
            * delta_x
            * delta_y
            / radius_fifth
        )

    if channel == 9:
        return (
            3.0e4
            * delta_x
            * delta_z
            / radius_fifth
        )

    if channel == 10:
        return (
            3.0e4
            * delta_z
            * delta_y
            / radius_fifth
        )

    if channel == 11:
        return (
            1.5e4
            * (
                delta_x**2
                - delta_y**2
            )
            / radius_fifth
        )

    raise ValueError(
        f"Unsupported gravity channel: {channel}."
    )


def calculate_gravity_table(
    *,
    model: npt.ArrayLike,
    model_grid: ModelGrid,
    receiver_points: npt.ArrayLike,
    channels: Iterable[int] = (4,),
    receiver_chunk_size: int = 128,
) -> FloatArray:
    """
    Calculate gravity and return a MATLAB-style output table.

    The five output columns are:

    ``X, Y, Z, channel_code, calculated_value``

    Rows are receiver-major with channel changing fastest, matching
    MATLAB ``getBulk.m`` and ``getPredGR.m``.

    Returns
    -------
    numpy.ndarray
        Forward-model table with five columns.
    """

    result = calculate_gravity(
        model=model,
        model_grid=model_grid,
        receiver_points=receiver_points,
        channels=channels,
        receiver_chunk_size=receiver_chunk_size,
    )

    number_of_receivers = (
        result.receiver_points.shape[0]
    )
    number_of_channels = len(
        result.channels
    )

    repeated_receivers = np.repeat(
        result.receiver_points,
        repeats=number_of_channels,
        axis=0,
    )

    tiled_channels = np.tile(
        np.asarray(
            result.channels,
            dtype=np.float64,
        ),
        reps=number_of_receivers,
    )

    flattened_values = result.values.reshape(
        -1,
        order="C",
    )

    return np.column_stack(
        (
            repeated_receivers,
            tiled_channels,
            flattened_values,
        )
    )


def calculate_gravity_volume(
    *,
    model: npt.ArrayLike,
    model_grid: ModelGrid,
    receiver_grid: ReceiverGrid,
    channel: int = 4,
    receiver_chunk_size: int = 128,
) -> FloatArray:
    """
    Calculate one gravity channel on a three-dimensional receiver grid.

    Parameters
    ----------
    model
        Density model with shape ``(nz, ny, nx)``.
    model_grid
        Subsurface density-model grid.
    receiver_grid
        Three-dimensional receiver grid.
    channel
        Gravity channel code. Channel 4 is Gz.
    receiver_chunk_size
        Number of receivers processed simultaneously.

    Returns
    -------
    numpy.ndarray
        Gravity volume with shape
        ``(receiver_nz, receiver_ny, receiver_nx)``.
    """

    result = calculate_gravity(
        model=model,
        model_grid=model_grid,
        receiver_points=receiver_grid.points(),
        channels=(channel,),
        receiver_chunk_size=receiver_chunk_size,
    )

    return receiver_grid.reshape_values(
        result.channel_values(
            channel
        )
    )