from __future__ import annotations

from time import perf_counter

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


def build_receiver_grid(
    grid: GridSpec,
) -> ReceiverGrid:
    """Build the repository-sized multi-level receiver grid."""

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
        z=-grid.dz
        * np.arange(
            1,
            9,
            dtype=np.float64,
        ),
    )


def build_test_model(
    grid: GridSpec,
) -> np.ndarray:
    """Build one centered rectangular density body."""

    body = RectangularBodySpec(
        name="sparse_optimization_test",
        x_start=26,
        x_end=38,
        y_start=26,
        y_end=38,
        z_start=6,
        z_end=11,
        density_contrast=1.0,
    )

    return build_density_model(
        grid=grid,
        case=body,
    )


def main() -> None:
    """Run and time the optimized forward model."""

    repository_grid = GridSpec()

    model_grid = model_grid_from_grid_spec(
        repository_grid
    )

    receiver_grid = build_receiver_grid(
        repository_grid
    )

    density_model = build_test_model(
        repository_grid
    )

    total_cells = density_model.size

    active_cells = int(
        np.count_nonzero(
            density_model
        )
    )

    forward_model = FWD3DGravityForwardModel(
        model_grid=model_grid,
        receiver_grid=receiver_grid,
        channel=4,
        receiver_chunk_size=128,
    )

    start_time = perf_counter()

    gravity_volume = forward_model.calculate(
        density_model
    )

    elapsed_seconds = (
        perf_counter()
        - start_time
    )

    expected_shape = (
        8,
        64,
        64,
    )

    if gravity_volume.shape != expected_shape:
        raise AssertionError(
            f"Expected gravity shape {expected_shape}, "
            f"received {gravity_volume.shape}."
        )

    if not np.all(
        np.isfinite(gravity_volume)
    ):
        raise AssertionError(
            "Gravity volume contains invalid values."
        )

    reference_value = (
        1.862204817338e-01
    )

    calculated_peak = float(
        np.max(
            gravity_volume[0]
        )
    )

    reference_error = abs(
        calculated_peak
        - reference_value
    )

    if reference_error > 1.0e-10:
        raise AssertionError(
            "The nonzero-cell optimization changed the result. "
            f"Expected peak {reference_value:.12e}, "
            f"received {calculated_peak:.12e}."
        )

    zero_model = np.zeros(
        repository_grid.nz
        * repository_grid.ny
        * repository_grid.nx,
        dtype=np.float64,
    ).reshape(
        (
            repository_grid.nz,
            repository_grid.ny,
            repository_grid.nx,
        )
    )

    zero_gravity = forward_model.calculate(
        zero_model
    )

    if not np.all(
        zero_gravity == 0.0
    ):
        raise AssertionError(
            "A zero-density model did not produce zero gravity."
        )

    reduction_factor = (
        total_cells
        / active_cells
    )

    print()
    print("Nonzero-cell gravity optimization test")
    print("=" * 38)
    print(
        f"Total model cells: "
        f"{total_cells:,}"
    )
    print(
        f"Active density cells: "
        f"{active_cells:,}"
    )
    print(
        f"Cell reduction factor: "
        f"{reduction_factor:.2f}x"
    )
    print(
        f"Receiver points: "
        f"{receiver_grid.number_of_receivers:,}"
    )
    print(
        f"Gravity shape: "
        f"{gravity_volume.shape}"
    )
    print(
        f"Elapsed time: "
        f"{elapsed_seconds:.3f} seconds"
    )
    print(
        f"First-layer peak Gz: "
        f"{calculated_peak:.12e} mGal"
    )
    print(
        f"Reference error: "
        f"{reference_error:.12e} mGal"
    )
    print()
    print("Numerical-equivalence check: PASSED")
    print("Zero-density model check: PASSED")
    print("Finite-value check: PASSED")
    print("Shape check: PASSED")


if __name__ == "__main__":
    main()