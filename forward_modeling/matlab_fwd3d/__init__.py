"""
Python recreation of the MATLAB FWD3D gravity forward model.

The package reproduces the gravity-related behavior of:

- getGpars.m
- getQuadPoints.m
- getBulk.m
- getRpars.m
- getPredGR.m

The model coordinate convention is:

- X positive east
- Y positive north
- Z positive downward
- Density model array order: model[z, y, x]
- Gravity volume order: gravity[z_receiver, y_receiver, x_receiver]
"""

from forward_modeling.matlab_fwd3d.gravity import (
    GRAVITY_CHANNELS,
    GravityForwardResult,
    calculate_gravity,
    calculate_gravity_table,
    calculate_gravity_volume,
)
from forward_modeling.matlab_fwd3d.grid import ModelGrid
from forward_modeling.matlab_fwd3d.receivers import ReceiverGrid

__all__ = [
    "GRAVITY_CHANNELS",
    "GravityForwardResult",
    "ModelGrid",
    "ReceiverGrid",
    "calculate_gravity",
    "calculate_gravity_table",
    "calculate_gravity_volume",
]