from __future__ import annotations

from dataclasses import dataclass

from synthetic_models.common.grid import GridSpec


@dataclass(frozen=True)
class MatlabCompatibleGridSpec(GridSpec):
    """
    Grid used only for the MATLAB-compatible FWD3D dataset.

    This class inherits from the repository's existing ``GridSpec``, so
    it can be passed directly to existing body-generation functions such
    as ``build_density_model``.

    The horizontal grid remains unchanged:

    - 64 X locations from 0 to 630 m
    - 64 Y locations from 0 to 630 m
    - 10 m spacing in X and Y

    The vertical extent is changed specifically for this dataset:

    - 24 Z locations from 0 to 230 m
    - 10 m spacing in Z

    Coordinate convention:

    - X positive east
    - Y positive north
    - Z positive downward
    - Density array order: density[z, y, x]
    """

    z_min: float = 0.0
    z_max: float = 230.0