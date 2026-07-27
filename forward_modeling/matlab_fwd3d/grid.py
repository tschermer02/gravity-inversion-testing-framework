from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ModelGrid:
    """
    Three-dimensional cell-centered model grid.

    The coordinate system follows the MATLAB FWD3D convention:

    - X is positive east.
    - Y is positive north.
    - Z is positive downward.
    - Bounds describe cell edges.
    - Cell locations are stored at cell centers.

    Density models supplied to this grid use array order ``(z, y, x)``.

    Parameters
    ----------
    x_centers
        X coordinates of model-cell centers, in meters.
    y_centers
        Y coordinates of model-cell centers, in meters.
    z_centers
        Z coordinates of model-cell centers, in meters.
    dx
        Uniform X cell width, in meters.
    dy
        Uniform Y cell width, in meters.
    dz
        Vertical cell thickness for each Z layer, in meters.
    """

    x_centers: FloatArray
    y_centers: FloatArray
    z_centers: FloatArray

    dx: float
    dy: float
    dz: FloatArray

    def __post_init__(self) -> None:
        """Validate and normalize grid parameters."""

        x_centers = np.asarray(
            self.x_centers,
            dtype=np.float64,
        )
        y_centers = np.asarray(
            self.y_centers,
            dtype=np.float64,
        )
        z_centers = np.asarray(
            self.z_centers,
            dtype=np.float64,
        )
        dz = np.asarray(
            self.dz,
            dtype=np.float64,
        )

        if x_centers.ndim != 1:
            raise ValueError("x_centers must be one-dimensional.")

        if y_centers.ndim != 1:
            raise ValueError("y_centers must be one-dimensional.")

        if z_centers.ndim != 1:
            raise ValueError("z_centers must be one-dimensional.")

        if dz.ndim != 1:
            raise ValueError("dz must be one-dimensional.")

        if x_centers.size == 0:
            raise ValueError("The model grid must contain X cells.")

        if y_centers.size == 0:
            raise ValueError("The model grid must contain Y cells.")

        if z_centers.size == 0:
            raise ValueError("The model grid must contain Z cells.")

        if z_centers.size != dz.size:
            raise ValueError(
                "The number of Z centers must equal the number of "
                "vertical cell thicknesses."
            )

        if self.dx <= 0.0:
            raise ValueError("dx must be greater than zero.")

        if self.dy <= 0.0:
            raise ValueError("dy must be greater than zero.")

        if np.any(dz <= 0.0):
            raise ValueError(
                "Every vertical cell thickness must be greater than zero."
            )

        for name, values in {
            "x_centers": x_centers,
            "y_centers": y_centers,
            "z_centers": z_centers,
            "dz": dz,
        }.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{name} contains NaN or infinite values."
                )

        if np.any(np.diff(x_centers) <= 0.0):
            raise ValueError(
                "x_centers must be strictly increasing."
            )

        if np.any(np.diff(y_centers) <= 0.0):
            raise ValueError(
                "y_centers must be strictly increasing."
            )

        if np.any(np.diff(z_centers) <= 0.0):
            raise ValueError(
                "z_centers must be strictly increasing."
            )

        object.__setattr__(
            self,
            "x_centers",
            x_centers,
        )
        object.__setattr__(
            self,
            "y_centers",
            y_centers,
        )
        object.__setattr__(
            self,
            "z_centers",
            z_centers,
        )
        object.__setattr__(
            self,
            "dz",
            dz,
        )

    @classmethod
    def from_bounds(
        cls,
        *,
        bounds: Sequence[float],
        dx: float,
        dy: float,
        dz: float | Sequence[float],
    ) -> "ModelGrid":
        """
        Build a grid from MATLAB-style model bounds.

        Parameters
        ----------
        bounds
            Model bounds in the order
            ``[xmin, xmax, ymin, ymax, zmin, zmax]``.
        dx
            Uniform X cell width, in meters.
        dy
            Uniform Y cell width, in meters.
        dz
            Either one uniform vertical cell thickness or one thickness
            per vertical layer.

        Returns
        -------
        ModelGrid
            Constructed cell-centered model grid.
        """

        bounds_array = np.asarray(
            bounds,
            dtype=np.float64,
        )

        if bounds_array.shape != (6,):
            raise ValueError(
                "bounds must contain exactly six values: "
                "[xmin, xmax, ymin, ymax, zmin, zmax]."
            )

        if not np.all(np.isfinite(bounds_array)):
            raise ValueError(
                "bounds contains NaN or infinite values."
            )

        xmin, xmax, ymin, ymax, zmin, zmax = (
            bounds_array
        )

        if xmax <= xmin:
            raise ValueError("xmax must be greater than xmin.")

        if ymax <= ymin:
            raise ValueError("ymax must be greater than ymin.")

        if zmax <= zmin:
            raise ValueError("zmax must be greater than zmin.")

        if dx <= 0.0:
            raise ValueError("dx must be greater than zero.")

        if dy <= 0.0:
            raise ValueError("dy must be greater than zero.")

        x_centers = np.arange(
            xmin + dx / 2.0,
            xmax,
            dx,
            dtype=np.float64,
        )

        y_centers = np.arange(
            ymin + dy / 2.0,
            ymax,
            dy,
            dtype=np.float64,
        )

        dz_array = cls._normalize_vertical_thicknesses(
            zmin=zmin,
            zmax=zmax,
            dz=dz,
        )

        z_centers = cls._calculate_vertical_centers(
            zmin=zmin,
            dz=dz_array,
        )

        valid = z_centers <= zmax

        z_centers = z_centers[valid]
        dz_array = dz_array[valid]

        return cls(
            x_centers=x_centers,
            y_centers=y_centers,
            z_centers=z_centers,
            dx=float(dx),
            dy=float(dy),
            dz=dz_array,
        )

    @staticmethod
    def _normalize_vertical_thicknesses(
        *,
        zmin: float,
        zmax: float,
        dz: float | Sequence[float],
    ) -> FloatArray:
        """
        Convert scalar or variable vertical spacing into an array.

        This follows the behavior of MATLAB ``getGpars.m``. For a scalar
        thickness, the number of layers is calculated with ``floor``.
        """

        dz_array = np.asarray(
            dz,
            dtype=np.float64,
        )

        if dz_array.ndim == 0:
            scalar_dz = float(dz_array)

            if scalar_dz <= 0.0:
                raise ValueError(
                    "dz must be greater than zero."
                )

            number_of_layers = int(
                np.floor(
                    (zmax - zmin) / scalar_dz
                )
            )

            if number_of_layers < 1:
                raise ValueError(
                    "The supplied dz does not produce any vertical cells."
                )

            return np.full(
                number_of_layers,
                scalar_dz,
                dtype=np.float64,
            )

        if dz_array.ndim != 1:
            raise ValueError(
                "dz must be a scalar or one-dimensional sequence."
            )

        if dz_array.size == 0:
            raise ValueError(
                "dz must contain at least one value."
            )

        if np.any(dz_array <= 0.0):
            raise ValueError(
                "Every dz value must be greater than zero."
            )

        return dz_array

    @staticmethod
    def _calculate_vertical_centers(
        *,
        zmin: float,
        dz: FloatArray,
    ) -> FloatArray:
        """
        Calculate Z cell centers for variable vertical spacing.
        """

        z_centers = np.empty(
            dz.size,
            dtype=np.float64,
        )

        z_centers[0] = (
            zmin
            + dz[0] / 2.0
        )

        for index in range(1, dz.size):
            z_centers[index] = (
                z_centers[index - 1]
                + (
                    dz[index - 1]
                    + dz[index]
                )
                / 2.0
            )

        return z_centers

    @property
    def nx(self) -> int:
        """Return the number of X cells."""

        return int(self.x_centers.size)

    @property
    def ny(self) -> int:
        """Return the number of Y cells."""

        return int(self.y_centers.size)

    @property
    def nz(self) -> int:
        """Return the number of Z cells."""

        return int(self.z_centers.size)

    @property
    def number_of_cells(self) -> int:
        """Return the total number of model cells."""

        return self.nx * self.ny * self.nz

    @property
    def model_shape(self) -> tuple[int, int, int]:
        """
        Return the expected density-model shape.

        The order is ``(z, y, x)``.
        """

        return (
            self.nz,
            self.ny,
            self.nx,
        )

    def flattened_cell_centers(
        self,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Return flattened model-cell coordinates.

        The returned ordering matches MATLAB ``ndgrid`` followed by
        ``(:)``:

        1. X changes fastest.
        2. Y changes next.
        3. Z changes slowest.

        Returns
        -------
        tuple of numpy.ndarray
            Flattened X, Y, and Z coordinates.
        """

        x_grid, y_grid, z_grid = np.meshgrid(
            self.x_centers,
            self.y_centers,
            self.z_centers,
            indexing="ij",
        )

        return (
            x_grid.ravel(order="F"),
            y_grid.ravel(order="F"),
            z_grid.ravel(order="F"),
        )

    def flattened_cell_volumes(self) -> FloatArray:
        """
        Return one volume for every flattened model cell.

        Returns
        -------
        numpy.ndarray
            Cell volumes in cubic meters using MATLAB model ordering.
        """

        dz_grid = np.broadcast_to(
            self.dz[np.newaxis, np.newaxis, :],
            (
                self.nx,
                self.ny,
                self.nz,
            ),
        )

        volumes = (
            self.dx
            * self.dy
            * dz_grid
        )

        return np.asarray(
            volumes,
            dtype=np.float64,
        ).ravel(order="F")

    def flatten_model(
        self,
        model: npt.ArrayLike,
    ) -> FloatArray:
        """
        Flatten a ``model[z, y, x]`` density array.

        C-order flattening of ``(z, y, x)`` makes X change fastest,
        matching the MATLAB model-vector ordering.

        Parameters
        ----------
        model
            Three-dimensional density model in g/cm3.

        Returns
        -------
        numpy.ndarray
            One-dimensional model vector.
        """

        model_array = np.asarray(
            model,
            dtype=np.float64,
        )

        if model_array.shape != self.model_shape:
            raise ValueError(
                f"Expected model shape {self.model_shape}, "
                f"but received {model_array.shape}."
            )

        if not np.all(np.isfinite(model_array)):
            raise ValueError(
                "The density model contains NaN or infinite values."
            )

        return np.ascontiguousarray(
            model_array,
            dtype=np.float64,
        ).ravel(order="C")

def model_grid_from_matlab_bounds(
    *,
    bounds: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
    ],
    dx: float,
    dy: float,
    dz: float,
) -> ModelGrid:
    """
    Build an FWD3D ModelGrid from MATLAB-style cell-edge bounds.

    Parameters
    ----------
    bounds
        Bounds in MATLAB order:

        ``(xmin, xmax, ymin, ymax, zmin, zmax)``

        These values describe cell edges, not cell centers.
    dx, dy, dz
        Uniform cell dimensions in meters.

    Returns
    -------
    ModelGrid
        Cell-centered FWD3D model grid.
    """

    (
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
    ) = bounds

    if dx <= 0.0:
        raise ValueError(
            "dx must be greater than zero."
        )

    if dy <= 0.0:
        raise ValueError(
            "dy must be greater than zero."
        )

    if dz <= 0.0:
        raise ValueError(
            "dz must be greater than zero."
        )

    x_extent = x_max - x_min
    y_extent = y_max - y_min
    z_extent = z_max - z_min

    nx_float = x_extent / dx
    ny_float = y_extent / dy
    nz_float = z_extent / dz

    if not np.isclose(
        nx_float,
        round(nx_float),
    ):
        raise ValueError(
            "The X extent must be evenly divisible by dx."
        )

    if not np.isclose(
        ny_float,
        round(ny_float),
    ):
        raise ValueError(
            "The Y extent must be evenly divisible by dy."
        )

    if not np.isclose(
        nz_float,
        round(nz_float),
    ):
        raise ValueError(
            "The Z extent must be evenly divisible by dz."
        )

    nx = int(round(nx_float))
    ny = int(round(ny_float))
    nz = int(round(nz_float))

    x_centers = (
        x_min
        + dx / 2.0
        + np.arange(
            nx,
            dtype=np.float64,
        )
        * dx
    )

    y_centers = (
        y_min
        + dy / 2.0
        + np.arange(
            ny,
            dtype=np.float64,
        )
        * dy
    )

    z_centers = (
        z_min
        + dz / 2.0
        + np.arange(
            nz,
            dtype=np.float64,
        )
        * dz
    )

    return ModelGrid(
        x_centers=x_centers,
        y_centers=y_centers,
        z_centers=z_centers,
        dx=dx,
        dy=dy,
        dz=np.full(
            nz,
            dz,
            dtype=np.float64,
        ),
    )