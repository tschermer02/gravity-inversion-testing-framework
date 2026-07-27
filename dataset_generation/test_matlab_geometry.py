from __future__ import annotations

import numpy as np

from dataset_generation.config import (
    DatasetGenerationConfig,
)
from forward_modeling.matlab_fwd3d.grid_adapter import (
    model_grid_from_grid_spec,
)

from dataset_generation.matlab_grid import MatlabCompatibleGridSpec




def main() -> None:
    """
    Confirm that dataset geometry follows MATLAB FWD3D conventions.
    """

    grid = MatlabCompatibleGridSpec()

    config = DatasetGenerationConfig()

    receiver_grid = (
        config.receivers.build_receiver_grid(
            grid
        )
    )

    model_grid = model_grid_from_grid_spec(
        grid
    )

    expected_spacing = 10.0

    if not np.isclose(
        grid.dx,
        expected_spacing,
    ):
        raise AssertionError(
            f"Expected dx=10 m, received {grid.dx}."
        )

    if not np.isclose(
        grid.dy,
        expected_spacing,
    ):
        raise AssertionError(
            f"Expected dy=10 m, received {grid.dy}."
        )

    if not np.isclose(
        grid.dz,
        expected_spacing,
    ):
        raise AssertionError(
            f"Expected dz=10 m, received {grid.dz}."
        )

    expected_receiver_z = np.array(
        [
            -10.0,
            -20.0,
            -30.0,
            -40.0,
            -50.0,
            -60.0,
            -70.0,
            -80.0,
        ],
        dtype=np.float64,
    )

    if not np.array_equal(
        receiver_grid.z,
        expected_receiver_z,
    ):
        raise AssertionError(
            "Receiver Z coordinates do not match "
            "the expected MATLAB-compatible geometry."
        )

    if -30.0 not in receiver_grid.z:
        raise AssertionError(
            "The receiver volume must include the original "
            "MATLAB benchmark plane at Z=-30 m."
        )

    if model_grid.model_shape != (
        24,
        64,
        64,
    ):
        raise AssertionError(
            f"Unexpected model shape: "
            f"{model_grid.model_shape}"
        )

    if not np.allclose(
        model_grid.dz,
        10.0,
    ):
        raise AssertionError(
            "FWD3D model-grid cells do not have "
            "10 m vertical thickness."
        )

    print()
    print("MATLAB-compatible dataset geometry")
    print("=" * 34)
    print(
        f"Density-model shape: "
        f"{model_grid.model_shape}"
    )
    print(
        f"Cell spacing: "
        f"{grid.dx:g} x "
        f"{grid.dy:g} x "
        f"{grid.dz:g} m"
    )
    print(
        f"Physical model extent: "
        f"{grid.x_min:g} to {grid.x_max:g} m X, "
        f"{grid.y_min:g} to {grid.y_max:g} m Y, "
        f"{grid.z_min:g} to {grid.z_max:g} m Z"
    )
    print(
        f"Receiver Z levels: "
        f"{receiver_grid.z}"
    )
    print()
    print("Isotropic 10 m cells: PASSED")
    print("Positive-downward Z convention: PASSED")
    print("MATLAB Z=-30 m receiver plane included: PASSED")
    print("Eight-plane gravity-volume extension: PASSED")


if __name__ == "__main__":
    main()