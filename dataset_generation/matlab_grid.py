from __future__ import annotations

from dataclasses import dataclass

from synthetic_models.common.grid import GridSpec


@dataclass(frozen=True)
class MatlabCompatibleGridSpec(GridSpec):
    """
    Grid specification for MATLAB-compatible FWD3D datasets.

    This class remains compatible with the repository's existing
    ``GridSpec`` interface, where the minimum and maximum coordinates
    represent cell-center locations.

    The translated MATLAB FWD3D solver instead defines model bounds using
    physical cell edges. This adapter therefore places cell centers half a
    cell inside the desired physical model bounds.

    Horizontal geometry
    -------------------
    The existing repository geometry is retained:

    - 64 X cell centers from 0 to 630 m
    - 64 Y cell centers from 0 to 630 m
    - dx = 10 m
    - dy = 10 m

    The corresponding horizontal physical edges are:

    - X: -5 to 635 m
    - Y: -5 to 635 m

    Vertical geometry
    -----------------
    The vertical grid contains 24 cells with:

    - Z cell centers from 5 to 235 m
    - dz = 10 m
    - Physical Z edges from 0 to 240 m

    Therefore, a Python slice ``density[z_start:z_end]`` corresponds to:

    - top physical depth: ``z_start * 10 m``
    - bottom physical depth: ``z_end * 10 m``

    For example, ``density[5:10]`` represents a body extending from
    50 to 100 m depth.

    Coordinate convention
    ---------------------
    - X positive east
    - Y positive north
    - Z positive downward
    - Density array order: ``density[z, y, x]``
    """

    z_min: float = 5.0
    z_max: float = 235.0

    @property
    def physical_z_min(self) -> float:
        """
        Return the physical top edge of the model.

        Returns
        -------
        float
            Physical top edge in meters.
        """

        return self.z_min - self.dz / 2.0

    @property
    def physical_z_max(self) -> float:
        """
        Return the physical bottom edge of the model.

        Returns
        -------
        float
            Physical bottom edge in meters.
        """

        return self.z_max + self.dz / 2.0

    def z_center_from_index(
        self,
        cell_index: int,
    ) -> float:
        """
        Convert a vertical cell index to its physical center depth.

        Parameters
        ----------
        cell_index
            Vertical cell index in the range ``[0, nz - 1]``.

        Returns
        -------
        float
            Cell-center depth in meters.

        Raises
        ------
        ValueError
            If the index lies outside the vertical grid.
        """

        if not 0 <= cell_index < self.nz:
            raise ValueError(
                "cell_index must lie within "
                f"[0, {self.nz - 1}], received {cell_index}."
            )

        return self.z_min + cell_index * self.dz

    def z_edge_from_index(
        self,
        edge_index: int,
    ) -> float:
        """
        Convert a vertical voxel-edge index to physical depth.

        Python slicing uses exclusive end indices. Therefore, a body with
        ``z_start=5`` and ``z_end=10`` has physical edges obtained from
        ``z_edge_from_index(5)`` and ``z_edge_from_index(10)``.

        Parameters
        ----------
        edge_index
            Vertical edge index in the range ``[0, nz]``.

        Returns
        -------
        float
            Physical edge depth in meters.

        Raises
        ------
        ValueError
            If the index lies outside the model edges.
        """

        if not 0 <= edge_index <= self.nz:
            raise ValueError(
                "edge_index must lie within "
                f"[0, {self.nz}], received {edge_index}."
            )

        return self.physical_z_min + edge_index * self.dz