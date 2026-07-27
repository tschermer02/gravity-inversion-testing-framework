from __future__ import annotations

import numpy as np

from forward_modeling.matlab_fwd3d.example_config import (
    build_matlab_example_density,
    build_matlab_example_grid,
)
from forward_modeling.matlab_fwd3d.forward_model import (
    FWD3DGravityForwardModel,
)
from forward_modeling.matlab_fwd3d.receivers import (
    ReceiverGrid,
)


def build_receiver_grid() -> ReceiverGrid:
    """
    Build the validated multi-elevation receiver grid.
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
            -60.0,
            -70.0,
            -80.0,
        ),
    )


def main() -> None:
    """
    Test the repository-facing FWD3D forward-model class.
    """

    model_grid = build_matlab_example_grid()

    receiver_grid = build_receiver_grid()

    density_model = build_matlab_example_density(
        model_grid
    )

    forward_model = FWD3DGravityForwardModel(
        model_grid=model_grid,
        receiver_grid=receiver_grid,
        channel=4,
    )

    gravity_volume = forward_model.calculate(
        density_model
    )

    expected_input_shape = (
        5,
        20,
        20,
    )

    expected_output_shape = (
        8,
        11,
        11,
    )

    if forward_model.input_shape != expected_input_shape:
        raise AssertionError(
            f"Unexpected input shape: "
            f"{forward_model.input_shape}"
        )

    if gravity_volume.shape != expected_output_shape:
        raise AssertionError(
            f"Unexpected gravity shape: "
            f"{gravity_volume.shape}"
        )

    if not np.all(
        np.isfinite(gravity_volume)
    ):
        raise AssertionError(
            "Gravity volume contains invalid values."
        )

    matlab_reference_value = 0.030789074

    z_minus_30_index = int(
        np.where(
            receiver_grid.z == -30.0
        )[0][0]
    )

    corner_value = gravity_volume[
        z_minus_30_index,
        0,
        0,
    ]

    reference_error = abs(
        corner_value
        - matlab_reference_value
    )

    if reference_error > 1.0e-6:
        raise AssertionError(
            "The adapter did not preserve the validated "
            "MATLAB reference value. "
            f"Expected approximately {matlab_reference_value}, "
            f"received {corner_value}."
        )

    print()
    print("FWD3D forward-model adapter test")
    print("=" * 34)
    print(
        f"Input density shape: "
        f"{density_model.shape}"
    )
    print(
        f"Output gravity shape: "
        f"{gravity_volume.shape}"
    )
    print(
        f"Gz at X=-250, Y=-250, Z=-30: "
        f"{corner_value:.12f} mGal"
    )
    print(
        f"Reference error: "
        f"{reference_error:.12e} mGal"
    )
    print(
        "Finite-value check: PASSED"
    )
    print(
        "Input-shape check: PASSED"
    )
    print(
        "Output-shape check: PASSED"
    )
    print(
        "MATLAB reference check: PASSED"
    )


if __name__ == "__main__":
    main()