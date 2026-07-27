from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ReceiverGrid:
    """
    Regular three-dimensional receiver grid.

    The resulting gravity volume has shape:

    ``(number_of_z_levels, number_of_y_locations, number_of_x_locations)``

    Parameters
    ----------
    x
        Receiver X coordinates, in meters.
    y
        Receiver Y coordinates, in meters.
    z
        Receiver Z coordinates, in meters.

        FWD3D uses positive-downward Z. Therefore, receivers above the
        surface normally have negative Z values.
    """

    x: FloatArray
    y: FloatArray
    z: FloatArray

    def __post_init__(self) -> None:
        """Validate and normalize receiver coordinates."""

        x = np.asarray(
            self.x,
            dtype=np.float64,
        )
        y = np.asarray(
            self.y,
            dtype=np.float64,
        )
        z = np.asarray(
            self.z,
            dtype=np.float64,
        )

        for name, values in {
            "x": x,
            "y": y,
            "z": z,
        }.items():
            if values.ndim != 1:
                raise ValueError(
                    f"Receiver {name} coordinates must be one-dimensional."
                )

            if values.size == 0:
                raise ValueError(
                    f"Receiver {name} coordinates must not be empty."
                )

            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Receiver {name} coordinates contain "
                    "NaN or infinite values."
                )

            if np.unique(values).size != values.size:
                raise ValueError(
                    f"Receiver {name} coordinates contain duplicates."
                )

        object.__setattr__(
            self,
            "x",
            x,
        )
        object.__setattr__(
            self,
            "y",
            y,
        )
        object.__setattr__(
            self,
            "z",
            z,
        )

    @classmethod
    def from_ranges(
        cls,
        *,
        x_min: float,
        x_max: float,
        x_step: float,
        y_min: float,
        y_max: float,
        y_step: float,
        z_values: npt.ArrayLike,
    ) -> "ReceiverGrid":
        """
        Construct a receiver grid from coordinate ranges.

        Endpoints are included when they fall on the requested spacing.

        Parameters
        ----------
        x_min, x_max
            Minimum and maximum receiver X coordinates.
        x_step
            Receiver spacing in X.
        y_min, y_max
            Minimum and maximum receiver Y coordinates.
        y_step
            Receiver spacing in Y.
        z_values
            One or more receiver elevations or depths.

        Returns
        -------
        ReceiverGrid
            Constructed receiver grid.
        """

        if x_step <= 0.0:
            raise ValueError(
                "x_step must be greater than zero."
            )

        if y_step <= 0.0:
            raise ValueError(
                "y_step must be greater than zero."
            )

        if x_max < x_min:
            raise ValueError(
                "x_max must be greater than or equal to x_min."
            )

        if y_max < y_min:
            raise ValueError(
                "y_max must be greater than or equal to y_min."
            )

        x = np.arange(
            x_min,
            x_max + x_step / 2.0,
            x_step,
            dtype=np.float64,
        )

        y = np.arange(
            y_min,
            y_max + y_step / 2.0,
            y_step,
            dtype=np.float64,
        )

        z = np.asarray(
            z_values,
            dtype=np.float64,
        )

        if z.ndim == 0:
            z = z.reshape(1)

        return cls(
            x=x,
            y=y,
            z=z,
        )

    @property
    def nx(self) -> int:
        """Return the number of receiver X coordinates."""

        return int(self.x.size)

    @property
    def ny(self) -> int:
        """Return the number of receiver Y coordinates."""

        return int(self.y.size)

    @property
    def nz(self) -> int:
        """Return the number of receiver Z levels."""

        return int(self.z.size)

    @property
    def number_of_receivers(self) -> int:
        """Return the total number of receiver locations."""

        return self.nx * self.ny * self.nz

    @property
    def volume_shape(self) -> tuple[int, int, int]:
        """
        Return the gravity-volume shape.

        The order is ``(z_receiver, y_receiver, x_receiver)``.
        """

        return (
            self.nz,
            self.ny,
            self.nx,
        )

    def points(self) -> FloatArray:
        """
        Return all receiver coordinates.

        Receiver order is:

        1. Z level changes slowest.
        2. Y changes next.
        3. X changes fastest.

        This allows a returned receiver vector to be reshaped directly
        into ``(z, y, x)``.

        Returns
        -------
        numpy.ndarray
            Receiver coordinate array with shape ``(N, 3)``.
        """

        z_grid, y_grid, x_grid = np.meshgrid(
            self.z,
            self.y,
            self.x,
            indexing="ij",
        )

        return np.column_stack(
            (
                x_grid.ravel(order="C"),
                y_grid.ravel(order="C"),
                z_grid.ravel(order="C"),
            )
        )

    def reshape_values(
        self,
        values: npt.ArrayLike,
    ) -> FloatArray:
        """
        Reshape receiver values into a three-dimensional gravity volume.

        Parameters
        ----------
        values
            One value for every receiver.

        Returns
        -------
        numpy.ndarray
            Gravity volume with order ``(z, y, x)``.
        """

        values_array = np.asarray(
            values,
            dtype=np.float64,
        )

        expected_size = self.number_of_receivers

        if values_array.size != expected_size:
            raise ValueError(
                f"Expected {expected_size} receiver values, "
                f"but received {values_array.size}."
            )

        return values_array.reshape(
            self.volume_shape,
            order="C",
        )