from __future__ import annotations

import numpy as np

from forward_modeling.matlab_fwd3d.forward_model import (
    FWD3DGravityForwardModel,
)
from forward_modeling.matlab_fwd3d.grid_adapter import (
    model_grid_from_grid_spec,
)
from forward_modeling.matlab_fwd3d.receivers import (
    ReceiverGrid,
)
from synthetic_models.common.bodies import (
    RectangularBodySpec,
    build_density_model,
)
from synthetic_models.common.grid import (
    GridSpec,
)


def build_repository_receiver_grid(
    grid: GridSpec,
) -> ReceiverGrid:
    """
    Build a multi-level receiver grid aligned with the repository X-Y grid.

    The first receiver is one vertical grid spacing above the surface.
    Under the positive-downward convention, above-surface coordinates
    are negative.
    """

    receiver_z = -grid.dz * np.arange(
        1,
        9,
        dtype=np.float64,
    )

    return ReceiverGrid(
        x=np.linspace(
            grid.x_min,
            grid.x_max,
            grid.nx,
            dtype=np.float64,
        ),
        y=np.linspace(
            grid.y_min,
            grid.y_max,
            grid.ny,
            dtype=np.float64,
        ),
        z=receiver_z,
    )


def build_test_body(
    grid: GridSpec,
) -> RectangularBodySpec:
    """
    Build one centered rectangular body using repository indices.
    """

    body_width_x = 12
    body_width_y = 12
    body_thickness_z = 5

    x_start = (
        grid.nx - body_width_x
    ) // 2

    y_start = (
        grid.ny - body_width_y
    ) // 2

    z_start = 6

    return RectangularBodySpec(
        name="fwd3d_repository_grid_test",
        x_start=x_start,
        x_end=x_start + body_width_x,
        y_start=y_start,
        y_end=y_start + body_width_y,
        z_start=z_start,
        z_end=z_start + body_thickness_z,
        density_contrast=1.0,
    )


def main() -> None:
    """
    Test FWD3D using the repository's normal density-grid dimensions.
    """

    repository_grid = GridSpec()

    fwd3d_grid = model_grid_from_grid_spec(
        repository_grid
    )

    receiver_grid = build_repository_receiver_grid(
        repository_grid
    )

    body = build_test_body(
        repository_grid
    )

    density_model = build_density_model(
        grid=repository_grid,
        case=body,
    )

    expected_density_shape = (
        repository_grid.nz,
        repository_grid.ny,
        repository_grid.nx,
    )

    if density_model.shape != expected_density_shape:
        raise AssertionError(
            "Generated density shape is incorrect. "
            f"Expected {expected_density_shape}, "
            f"received {density_model.shape}."
        )

    forward_model = FWD3DGravityForwardModel(
        model_grid=fwd3d_grid,
        receiver_grid=receiver_grid,
        channel=4,
        receiver_chunk_size=32,
    )

    gravity_volume = forward_model.calculate(
        density_model
    )

    expected_gravity_shape = (
        8,
        repository_grid.ny,
        repository_grid.nx,
    )

    if gravity_volume.shape != expected_gravity_shape:
        raise AssertionError(
            "Gravity-volume shape is incorrect. "
            f"Expected {expected_gravity_shape}, "
            f"received {gravity_volume.shape}."
        )

    if not np.all(np.isfinite(gravity_volume)):
        raise AssertionError(
            "Gravity volume contains NaN or infinite values."
        )

    if np.all(gravity_volume == 0.0):
        raise AssertionError(
            "Gravity volume contains only zeros."
        )

    layer_maximums = np.max(
        gravity_volume,
        axis=(1, 2),
    )

    if not np.all(
        np.diff(layer_maximums) < 0.0
    ):
        raise AssertionError(
            "Gravity peak does not decrease as receivers "
            "move farther above the model."
        )

    print()
    print("Repository-grid FWD3D test")
    print("=" * 29)
    print(
        f"Repository density shape: "
        f"{density_model.shape}"
    )
    print(
        f"FWD3D model shape: "
        f"{fwd3d_grid.model_shape}"
    )
    print(
        f"Gravity-volume shape: "
        f"{gravity_volume.shape}"
    )
    print(
        f"Receiver Z values: "
        f"{receiver_grid.z}"
    )
    print(
        f"Density nonzero cells: "
        f"{np.count_nonzero(density_model)}"
    )
    print()
    print("Layer maximum Gz values:")

    for receiver_z, maximum in zip(
        receiver_grid.z,
        layer_maximums,
        strict=True,
    ):
        print(
            f"Z = {receiver_z:7.2f} | "
            f"maximum Gz = {maximum:.12e} mGal"
        )

    print()
    print("Repository GridSpec conversion: PASSED")
    print("Density-model generation: PASSED")
    print("Gravity-volume generation: PASSED")
    print("Receiver-distance behavior: PASSED")


if __name__ == "__main__":
    main()