from __future__ import annotations

import numpy as np

from dataset_generation.config import (
    DatasetGenerationConfig,
)
from forward_modeling.matlab_fwd3d.grid_adapter import (
    model_grid_from_grid_spec,
)

from dataset_generation.matlab_grid import MatlabCompatibleGridSpec

from forward_modeling.matlab_fwd3d.grid import ModelGrid
from dataset_generation.case_sampler import RectangularBodySampler
from dataset_generation.config import BodySamplingConfig


def test_matlab_example_bounds_create_expected_centers() -> None:
    """
    Verify that MATLAB edge bounds create the expected cell centers.

    The MATLAB example uses physical edge bounds:

    - X: -100 to 100 m
    - Y: -100 to 100 m
    - Z: 50 to 100 m

    with 10 m cells. The resulting grid must contain 20 x 20 x 5
    cells centered half a cell inside each boundary.
    """

    grid = ModelGrid.from_bounds(
        bounds=(
            -100.0,
            100.0,
            -100.0,
            100.0,
            50.0,
            100.0,
        ),
        dx=10.0,
        dy=10.0,
        dz=10.0,
    )

    expected_x_centers = np.arange(
        -95.0,
        100.0,
        10.0,
        dtype=np.float64,
    )

    expected_y_centers = np.arange(
        -95.0,
        100.0,
        10.0,
        dtype=np.float64,
    )

    expected_z_centers = np.arange(
        55.0,
        100.0,
        10.0,
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        grid.x_centers,
        expected_x_centers,
        rtol=0.0,
        atol=0.0,
    )

    np.testing.assert_allclose(
        grid.y_centers,
        expected_y_centers,
        rtol=0.0,
        atol=0.0,
    )

    np.testing.assert_allclose(
        grid.z_centers,
        expected_z_centers,
        rtol=0.0,
        atol=0.0,
    )

    assert grid.model_shape == (
        5,
        20,
        20,
    )

    assert grid.number_of_cells == 2_000


def test_matlab_example_cell_volumes_are_correct() -> None:
    """
    Verify the pulse-basis cell volumes from the MATLAB example.

    Every cell is 10 x 10 x 10 m, so every volume must be 1,000 m3.
    """

    grid = ModelGrid.from_bounds(
        bounds=(
            -100.0,
            100.0,
            -100.0,
            100.0,
            50.0,
            100.0,
        ),
        dx=10.0,
        dy=10.0,
        dz=10.0,
    )

    volumes = grid.flattened_cell_volumes()

    assert volumes.shape == (2_000,)

    np.testing.assert_allclose(
        volumes,
        np.full(
            2_000,
            1_000.0,
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_density_flattening_matches_matlab_cell_order() -> None:
    """
    Verify that density[z, y, x] flattens with X changing fastest.

    MATLAB uses ``ndgrid(x, y, z)`` followed by column-major
    vectorization. A NumPy ``(z, y, x)`` array flattened in C order
    must produce the same model-vector ordering.
    """

    grid = ModelGrid.from_bounds(
        bounds=(
            0.0,
            30.0,
            0.0,
            20.0,
            0.0,
            20.0,
        ),
        dx=10.0,
        dy=10.0,
        dz=10.0,
    )

    density = np.arange(
        grid.number_of_cells,
        dtype=np.float64,
    ).reshape(
        grid.model_shape
    )

    flattened_density = grid.flatten_model(
        density
    )

    expected_density = np.arange(
        grid.number_of_cells,
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        flattened_density,
        expected_density,
    )


def test_matlab_z_bounds_represent_physical_edges() -> None:
    """
    Verify that MATLAB Z bounds are edges rather than cell centers.

    Bounds of 50 to 100 m with dz=10 m must produce centers from
    55 to 95 m. Neither physical edge should appear as a center.
    """

    grid = ModelGrid.from_bounds(
        bounds=(
            -100.0,
            100.0,
            -100.0,
            100.0,
            50.0,
            100.0,
        ),
        dx=10.0,
        dy=10.0,
        dz=10.0,
    )

    assert grid.z_centers[0] == 55.0
    assert grid.z_centers[-1] == 95.0

    assert 50.0 not in grid.z_centers
    assert 100.0 not in grid.z_centers

from dataset_generation.matlab_grid import MatlabCompatibleGridSpec


def test_matlab_compatible_repository_grid_spacing() -> None:
    """
    Verify the repository-sized MATLAB-compatible grid spacing.
    """

    grid = MatlabCompatibleGridSpec()

    assert grid.nx == 64
    assert grid.ny == 64
    assert grid.nz == 24

    assert grid.dx == 10.0
    assert grid.dy == 10.0
    assert grid.dz == 10.0


def test_matlab_compatible_vertical_centers() -> None:
    """
    Verify that vertical locations are MATLAB-style cell centers.
    """

    grid = MatlabCompatibleGridSpec()

    assert grid.z_min == 5.0
    assert grid.z_max == 235.0

    assert grid.z_center_from_index(0) == 5.0
    assert grid.z_center_from_index(1) == 15.0
    assert grid.z_center_from_index(23) == 235.0


def test_matlab_compatible_vertical_edges() -> None:
    """
    Verify that the vertical physical model spans 0 to 240 m.
    """

    grid = MatlabCompatibleGridSpec()

    assert grid.physical_z_min == 0.0
    assert grid.physical_z_max == 240.0

    assert grid.z_edge_from_index(0) == 0.0
    assert grid.z_edge_from_index(24) == 240.0


def test_sampled_body_metadata_uses_correct_slice_semantics() -> None:
    """
    Verify occupied indices, center indices, and physical depths.

    A body with z_start=5 and thickness=5 occupies indices 5 through 9
    and has physical edge depths of 50 and 100 m.
    """

    grid = MatlabCompatibleGridSpec()

    config = BodySamplingConfig(
        minimum_width_x=4,
        maximum_width_x=4,
        minimum_width_y=6,
        maximum_width_y=6,
        minimum_thickness_z=5,
        maximum_thickness_z=5,
        minimum_top_depth_index=5,
        maximum_top_depth_index=5,
        minimum_density_contrast=0.5,
        maximum_density_contrast=0.5,
        horizontal_margin_cells=2,
    )

    sampler = RectangularBodySampler(
        grid=grid,
        config=config,
        random_generator=np.random.default_rng(123),
    )

    sampled = sampler.sample(
        sample_index=0
    )

    body = sampled.specification

    assert body.z_start == 5
    assert body.z_end == 10

    assert sampled.top_depth_index == 5
    assert sampled.bottom_depth_index == 9
    assert sampled.center_z_index == 7.0

    assert sampled.top_depth_m == 50.0
    assert sampled.bottom_depth_m == 100.0
    assert sampled.center_depth_m == 75.0

    assert sampled.width_x_m == 40.0
    assert sampled.width_y_m == 60.0
    assert sampled.thickness_z_m == 50.0

    assert sampled.width_x == 4
    assert sampled.width_y == 6
    assert sampled.thickness_z == 5


def test_matlab_body_slice_matches_physical_bounds() -> None:
    """
    Verify that slice indices 5:10 represent depths 50 to 100 m.
    """

    grid = MatlabCompatibleGridSpec()

    z_start = 5
    z_end = 10

    assert grid.z_edge_from_index(z_start) == 50.0
    assert grid.z_edge_from_index(z_end) == 100.0

    occupied_centers = [
        grid.z_center_from_index(index)
        for index in range(z_start, z_end)
    ]

    assert occupied_centers == [
        55.0,
        65.0,
        75.0,
        85.0,
        95.0,
    ]


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