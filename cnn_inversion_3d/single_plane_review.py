from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from dataset_generation.matlab_grid import (
    MatlabCompatibleGridSpec,
)
from forward_modeling.matlab_fwd3d.forward_model import (
    FWD3DGravityForwardModel,
)
from forward_modeling.matlab_fwd3d.grid_adapter import (
    model_grid_from_grid_spec,
)
from forward_modeling.matlab_fwd3d.receivers import (
    ReceiverGrid,
)


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SinglePlaneReviewConfig:
    """Physical configuration for the single-plane review scenario."""

    nx: int = 64
    ny: int = 64
    nz: int = 24
    dx_m: float = 10.0
    dy_m: float = 10.0
    dz_m: float = 10.0
    density_x_min_center_m: float = 0.0
    density_y_min_center_m: float = 0.0
    density_z_min_center_m: float = 5.0
    observation_x_min_m: float = -85.0
    observation_x_max_m: float = 715.0
    observation_y_min_m: float = -85.0
    observation_y_max_m: float = 715.0
    observation_spacing_m: float = 10.0
    observation_z_m: float = 0.0
    observation_marker_stride: int = 8
    gravity_display_margin_m: float | None = 0.0
    minimum_top_depth_m: float = 20.0
    maximum_top_depth_m: float = 80.0
    minimum_thickness_m: float = 20.0
    maximum_thickness_m: float = 80.0
    minimum_width_x_m: float = 40.0
    maximum_width_x_m: float = 160.0
    minimum_width_y_m: float = 40.0
    maximum_width_y_m: float = 160.0
    minimum_density_contrast_g_cm3: float = 0.2
    maximum_density_contrast_g_cm3: float = 1.0
    maximum_bottom_depth_m: float = 160.0
    density_unit: str = "g/cm3"
    gravity_unit: str = "mGal"
    gravity_component: str = "Gz"
    gravity_channel: int = 4
    receiver_chunk_size: int = 128

    @property
    def density_shape(self) -> tuple[int, int, int]:
        """Return density shape in ``(z, y, x)`` order."""

        return (
            self.nz,
            self.ny,
            self.nx,
        )

    @property
    def density_x_max_center_m(self) -> float:
        """Return the last density X-cell center."""

        return (
            self.density_x_min_center_m
            + (self.nx - 1) * self.dx_m
        )

    @property
    def density_y_max_center_m(self) -> float:
        """Return the last density Y-cell center."""

        return (
            self.density_y_min_center_m
            + (self.ny - 1) * self.dy_m
        )

    @property
    def density_z_max_center_m(self) -> float:
        """Return the last density Z-cell center."""

        return (
            self.density_z_min_center_m
            + (self.nz - 1) * self.dz_m
        )

    @property
    def density_x_edges_m(self) -> tuple[float, float]:
        """Return physical X edges of the density domain."""

        return (
            self.density_x_min_center_m
            - self.dx_m / 2.0,
            self.density_x_max_center_m
            + self.dx_m / 2.0,
        )

    @property
    def density_y_edges_m(self) -> tuple[float, float]:
        """Return physical Y edges of the density domain."""

        return (
            self.density_y_min_center_m
            - self.dy_m / 2.0,
            self.density_y_max_center_m
            + self.dy_m / 2.0,
        )

    @property
    def density_z_edges_m(self) -> tuple[float, float]:
        """Return physical depth edges of the density domain."""

        return (
            self.density_z_min_center_m
            - self.dz_m / 2.0,
            self.density_z_max_center_m
            + self.dz_m / 2.0,
        )

    @property
    def observation_x_m(self) -> FloatArray:
        """Return observation X coordinates."""

        return np.arange(
            self.observation_x_min_m,
            self.observation_x_max_m
            + self.observation_spacing_m / 2.0,
            self.observation_spacing_m,
            dtype=np.float64,
        )

    @property
    def observation_y_m(self) -> FloatArray:
        """Return observation Y coordinates."""

        return np.arange(
            self.observation_y_min_m,
            self.observation_y_max_m
            + self.observation_spacing_m / 2.0,
            self.observation_spacing_m,
            dtype=np.float64,
        )

    @property
    def gravity_display_xlim(self) -> tuple[float, float]:
        """Return display-only gravity X limits in physical meters."""

        if self.gravity_display_margin_m is None:
            return (
                float(self.observation_x_m[0]),
                float(self.observation_x_m[-1]),
            )
        return (
            max(
                float(self.observation_x_m[0]),
                self.density_x_edges_m[0]
                - self.gravity_display_margin_m,
            ),
            min(
                float(self.observation_x_m[-1]),
                self.density_x_edges_m[1]
                + self.gravity_display_margin_m,
            ),
        )

    @property
    def gravity_display_ylim(self) -> tuple[float, float]:
        """Return display-only gravity Y limits in physical meters."""

        if self.gravity_display_margin_m is None:
            return (
                float(self.observation_y_m[0]),
                float(self.observation_y_m[-1]),
            )
        return (
            max(
                float(self.observation_y_m[0]),
                self.density_y_edges_m[0]
                - self.gravity_display_margin_m,
            ),
            min(
                float(self.observation_y_m[-1]),
                self.density_y_edges_m[1]
                + self.gravity_display_margin_m,
            ),
        )

    def grid_spec(self) -> MatlabCompatibleGridSpec:
        """Return the existing MATLAB-compatible density grid."""

        return MatlabCompatibleGridSpec(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            x_min=self.density_x_min_center_m,
            x_max=self.density_x_max_center_m,
            y_min=self.density_y_min_center_m,
            y_max=self.density_y_max_center_m,
            z_min=self.density_z_min_center_m,
            z_max=self.density_z_max_center_m,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return complete JSON-compatible physical metadata."""

        metadata: dict[str, object] = asdict(
            self
        )
        metadata.update(
            {
                "density_shape_zyx": list(
                    self.density_shape
                ),
                "density_x_center_range_m": [
                    self.density_x_min_center_m,
                    self.density_x_max_center_m,
                ],
                "density_y_center_range_m": [
                    self.density_y_min_center_m,
                    self.density_y_max_center_m,
                ],
                "density_z_center_range_m": [
                    self.density_z_min_center_m,
                    self.density_z_max_center_m,
                ],
                "density_x_edge_range_m": list(
                    self.density_x_edges_m
                ),
                "density_y_edge_range_m": list(
                    self.density_y_edges_m
                ),
                "density_z_edge_range_m": list(
                    self.density_z_edges_m
                ),
                "observation_x_coordinates_m": (
                    self.observation_x_m.tolist()
                ),
                "observation_y_coordinates_m": (
                    self.observation_y_m.tolist()
                ),
                "observation_shape_yx": [
                    self.observation_y_m.size,
                    self.observation_x_m.size,
                ],
                "density_array_order": "density[z, y, x]",
                "gravity_array_order": "gravity[y, x]",
                "coordinate_convention": {
                    "x": "positive east",
                    "y": "positive north",
                    "z": "positive downward",
                    "density_coordinates": "cell centers",
                    "body_bounds": "physical cell edges",
                    "receiver_coordinates": "physical points",
                },
            }
        )
        return metadata


@dataclass(frozen=True)
class SinglePlaneBody:
    """Manually specified rectangular body in physical units."""

    name: str
    top_depth_m: float
    width_x_m: float
    width_y_m: float
    thickness_m: float
    density_contrast_g_cm3: float
    center_x_m: float
    center_y_m: float

    @property
    def bottom_depth_m(self) -> float:
        """Return body bottom depth."""

        return (
            self.top_depth_m
            + self.thickness_m
        )

    @property
    def x_bounds_m(self) -> tuple[float, float]:
        """Return physical body X edges."""

        return (
            self.center_x_m
            - self.width_x_m / 2.0,
            self.center_x_m
            + self.width_x_m / 2.0,
        )

    @property
    def y_bounds_m(self) -> tuple[float, float]:
        """Return physical body Y edges."""

        return (
            self.center_y_m
            - self.width_y_m / 2.0,
            self.center_y_m
            + self.width_y_m / 2.0,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return body parameters and bounds in physical units."""

        return {
            **asdict(self),
            "bottom_depth_m": (
                self.bottom_depth_m
            ),
            "x_bounds_m": list(
                self.x_bounds_m
            ),
            "y_bounds_m": list(
                self.y_bounds_m
            ),
            "z_bounds_m": [
                self.top_depth_m,
                self.bottom_depth_m,
            ],
        }


def single_plane_review_examples() -> tuple[SinglePlaneBody, ...]:
    """
    Return five deterministic review bodies.

    Bodies are horizontally centered as closely as cell-edge alignment
    permits. Even-cell widths use center 315 m; odd-cell widths use
    center 320 m.
    """

    return (
        SinglePlaneBody(
            name="example_01_shallow_small",
            top_depth_m=20.0,
            width_x_m=40.0,
            width_y_m=40.0,
            thickness_m=20.0,
            density_contrast_g_cm3=0.4,
            center_x_m=315.0,
            center_y_m=315.0,
        ),
        SinglePlaneBody(
            name="example_02_shallow_broad",
            top_depth_m=30.0,
            width_x_m=120.0,
            width_y_m=100.0,
            thickness_m=40.0,
            density_contrast_g_cm3=0.6,
            center_x_m=315.0,
            center_y_m=315.0,
        ),
        SinglePlaneBody(
            name="example_03_intermediate",
            top_depth_m=50.0,
            width_x_m=80.0,
            width_y_m=100.0,
            thickness_m=50.0,
            density_contrast_g_cm3=0.7,
            center_x_m=315.0,
            center_y_m=315.0,
        ),
        SinglePlaneBody(
            name="example_04_deeper_compact",
            top_depth_m=70.0,
            width_x_m=50.0,
            width_y_m=60.0,
            thickness_m=50.0,
            density_contrast_g_cm3=0.8,
            center_x_m=320.0,
            center_y_m=315.0,
        ),
        SinglePlaneBody(
            name="example_05_deepest_largest",
            top_depth_m=80.0,
            width_x_m=160.0,
            width_y_m=160.0,
            thickness_m=80.0,
            density_contrast_g_cm3=1.0,
            center_x_m=315.0,
            center_y_m=315.0,
        ),
    )


def controlled_single_plane_review_examples(
    config: SinglePlaneReviewConfig,
) -> tuple[SinglePlaneBody, ...]:
    """Return the deterministic one-variable-at-a-time review set."""

    center_x = sum(config.density_x_edges_m) / 2.0
    center_y = sum(config.density_y_edges_m) / 2.0
    shift_x = 140.0
    common = {
        "top_depth_m": 20.0,
        "width_x_m": 60.0,
        "width_y_m": 60.0,
        "thickness_m": 30.0,
        "density_contrast_g_cm3": 0.5,
        "center_x_m": center_x,
        "center_y_m": center_y,
    }

    def body(
        name: str,
        **changes: float,
    ) -> SinglePlaneBody:
        return SinglePlaneBody(
            name=name,
            **{
                **common,
                **changes,
            },
        )

    return (
        body("controlled_example_a_baseline"),
        body(
            "controlled_example_b_increased_depth",
            top_depth_m=70.0,
        ),
        body(
            "controlled_example_c_increased_horizontal_size",
            width_x_m=140.0,
            width_y_m=140.0,
        ),
        body(
            "controlled_example_d_increased_density_contrast",
            density_contrast_g_cm3=1.0,
        ),
        body(
            "controlled_example_e_increased_thickness",
            thickness_m=80.0,
        ),
        body(
            "controlled_example_f_maximum_combined_case",
            top_depth_m=80.0,
            width_x_m=160.0,
            width_y_m=160.0,
            thickness_m=80.0,
            density_contrast_g_cm3=1.0,
        ),
        body(
            "controlled_example_g_left_shifted_position",
            center_x_m=center_x - shift_x,
        ),
        body(
            "controlled_example_h_right_shifted_position",
            center_x_m=center_x + shift_x,
        ),
    )


def controlled_example_change(
    body: SinglePlaneBody,
) -> dict[str, object]:
    """Return the controlled parameter changed relative to baseline."""

    changes = {
        "controlled_example_a_baseline": (
            "none",
            "baseline",
        ),
        "controlled_example_b_increased_depth": (
            "top_depth_m",
            body.top_depth_m,
        ),
        "controlled_example_c_increased_horizontal_size": (
            "horizontal_size",
            {
                "width_x_m": body.width_x_m,
                "width_y_m": body.width_y_m,
            },
        ),
        "controlled_example_d_increased_density_contrast": (
            "density_contrast_g_cm3",
            body.density_contrast_g_cm3,
        ),
        "controlled_example_e_increased_thickness": (
            "thickness_m",
            body.thickness_m,
        ),
        "controlled_example_f_maximum_combined_case": (
            "multiple_parameters",
            "maximum combined case",
        ),
        "controlled_example_g_left_shifted_position": (
            "horizontal_position",
            {
                "center_x_m": body.center_x_m,
                "center_y_m": body.center_y_m,
            },
        ),
        "controlled_example_h_right_shifted_position": (
            "horizontal_position",
            {
                "center_x_m": body.center_x_m,
                "center_y_m": body.center_y_m,
            },
        ),
    }
    parameter, value = changes[body.name]
    return {
        "changed_parameter_relative_to_baseline": parameter,
        "changed_value": value,
    }


def format_single_plane_example_title(
    body: SinglePlaneBody,
) -> str:
    """Return a concise presentation title for a review body."""

    prefix = "controlled_example_"
    if body.name.startswith(prefix):
        remainder = body.name[len(prefix):]
        letter, description = remainder.split("_", maxsplit=1)
        return (
            f"Controlled Example {letter.upper()}: "
            f"{description.replace('_', ' ').title()}"
        )
    return body.name.replace("_", " ").title()


def _is_spacing_multiple(
    value: float,
    spacing: float,
) -> bool:
    """Return whether a physical length is grid aligned."""

    return bool(
        np.isclose(
            value / spacing,
            round(
                value / spacing
            ),
        )
    )


def validate_single_plane_review_geometry(
    config: SinglePlaneReviewConfig,
    examples: Sequence[SinglePlaneBody],
) -> None:
    """
    Validate review geometry in physical units.

    Raises
    ------
    ValueError
        If grid, observation plane, body, or five-times criteria fail.
    """

    if not examples:
        raise ValueError(
            "At least one review body is required."
        )

    if (
        config.density_shape
        != (24, 64, 64)
    ):
        raise ValueError(
            "Review density shape must be (24, 64, 64)."
        )

    observation_x = config.observation_x_m
    observation_y = config.observation_y_m

    if (
        not np.allclose(
            np.diff(observation_x),
            config.observation_spacing_m,
        )
        or not np.allclose(
            np.diff(observation_y),
            config.observation_spacing_m,
        )
    ):
        raise ValueError(
            "Observation coordinates must have fixed spacing in meters."
        )

    x_edge_min, x_edge_max = (
        config.density_x_edges_m
    )
    y_edge_min, y_edge_max = (
        config.density_y_edges_m
    )

    for body in examples:
        for label, value, spacing in (
            (
                "width_x_m",
                body.width_x_m,
                config.dx_m,
            ),
            (
                "width_y_m",
                body.width_y_m,
                config.dy_m,
            ),
            (
                "top_depth_m",
                body.top_depth_m,
                config.dz_m,
            ),
            (
                "thickness_m",
                body.thickness_m,
                config.dz_m,
            ),
        ):
            if not _is_spacing_multiple(
                value,
                spacing,
            ):
                raise ValueError(
                    f"{body.name} {label}={value:g} m is not an "
                    f"integer multiple of {spacing:g} m."
                )

        if not (
            config.minimum_top_depth_m
            <= body.top_depth_m
            <= config.maximum_top_depth_m
        ):
            raise ValueError(
                f"{body.name} top depth {body.top_depth_m:g} m is "
                "outside the configured range."
            )

        if not (
            config.minimum_thickness_m
            <= body.thickness_m
            <= config.maximum_thickness_m
        ):
            raise ValueError(
                f"{body.name} thickness {body.thickness_m:g} m is "
                "outside the configured range."
            )

        if not (
            config.minimum_width_x_m
            <= body.width_x_m
            <= config.maximum_width_x_m
        ):
            raise ValueError(
                f"{body.name} X width {body.width_x_m:g} m is "
                "outside the configured range."
            )

        if not (
            config.minimum_width_y_m
            <= body.width_y_m
            <= config.maximum_width_y_m
        ):
            raise ValueError(
                f"{body.name} Y width {body.width_y_m:g} m is "
                "outside the configured range."
            )

        if not (
            config.minimum_density_contrast_g_cm3
            <= body.density_contrast_g_cm3
            <= config.maximum_density_contrast_g_cm3
        ):
            raise ValueError(
                f"{body.name} density contrast "
                f"{body.density_contrast_g_cm3:g} {config.density_unit} "
                "is outside the configured range."
            )

        if body.bottom_depth_m > (
            config.maximum_bottom_depth_m
        ):
            raise ValueError(
                f"{body.name} bottom depth {body.bottom_depth_m:g} m "
                f"exceeds {config.maximum_bottom_depth_m:g} m."
            )

        if not (
            x_edge_min
            <= body.x_bounds_m[0]
            < body.x_bounds_m[1]
            <= x_edge_max
        ):
            raise ValueError(
                f"{body.name} X bounds {body.x_bounds_m} m lie outside "
                f"the density domain {(x_edge_min, x_edge_max)} m."
            )

        if not (
            y_edge_min
            <= body.y_bounds_m[0]
            < body.y_bounds_m[1]
            <= y_edge_max
        ):
            raise ValueError(
                f"{body.name} Y bounds {body.y_bounds_m} m lie outside "
                f"the density domain {(y_edge_min, y_edge_max)} m."
            )

        for bound in (
            *body.x_bounds_m,
        ):
            if not _is_spacing_multiple(
                bound - x_edge_min,
                config.dx_m,
            ):
                raise ValueError(
                    f"{body.name} X bound {bound:g} m is not aligned "
                    "with a density-cell edge."
                )

        for bound in (
            *body.y_bounds_m,
        ):
            if not _is_spacing_multiple(
                bound - y_edge_min,
                config.dy_m,
            ):
                raise ValueError(
                    f"{body.name} Y bound {bound:g} m is not aligned "
                    "with a density-cell edge."
                )

    observation_width = float(
        observation_x[-1]
        - observation_x[0]
    )
    observation_length = float(
        observation_y[-1]
        - observation_y[0]
    )
    required_extent = 5.0 * max(
        max(
            body.width_x_m
            for body in examples
        ),
        max(
            body.width_y_m
            for body in examples
        ),
        max(
            body.bottom_depth_m
            for body in examples
        ),
    )

    if (
        observation_width < required_extent
        or observation_length < required_extent
    ):
        raise ValueError(
            "Observation extent fails the five-times criterion: "
            f"{observation_width:g} x {observation_length:g} m is "
            f"smaller than {required_extent:g} m."
        )


def build_single_plane_density(
    config: SinglePlaneReviewConfig,
    body: SinglePlaneBody,
) -> FloatArray:
    """
    Build one edge-aligned rectangular density model.

    Returns
    -------
    numpy.ndarray
        Density in ``(z, y, x)`` order.
    """

    validate_single_plane_review_geometry(
        config,
        [body],
    )
    x_edge_min, _ = config.density_x_edges_m
    y_edge_min, _ = config.density_y_edges_m
    z_edge_min, _ = config.density_z_edges_m
    x_start = int(
        round(
            (
                body.x_bounds_m[0]
                - x_edge_min
            )
            / config.dx_m
        )
    )
    x_end = int(
        round(
            (
                body.x_bounds_m[1]
                - x_edge_min
            )
            / config.dx_m
        )
    )
    y_start = int(
        round(
            (
                body.y_bounds_m[0]
                - y_edge_min
            )
            / config.dy_m
        )
    )
    y_end = int(
        round(
            (
                body.y_bounds_m[1]
                - y_edge_min
            )
            / config.dy_m
        )
    )
    z_start = int(
        round(
            (
                body.top_depth_m
                - z_edge_min
            )
            / config.dz_m
        )
    )
    z_end = int(
        round(
            (
                body.bottom_depth_m
                - z_edge_min
            )
            / config.dz_m
        )
    )
    density = np.zeros(
        config.density_shape,
        dtype=np.float64,
    )
    density[
        z_start:z_end,
        y_start:y_end,
        x_start:x_end,
    ] = body.density_contrast_g_cm3
    return density


def forward_model_single_plane(
    density: npt.ArrayLike,
    *,
    config: SinglePlaneReviewConfig,
) -> FloatArray:
    """
    Compute Gz on one horizontal observation plane.

    Returns
    -------
    numpy.ndarray
        Two-dimensional gravity map in ``(y_observation, x_observation)``
        order and mGal units.
    """

    density_array = np.asarray(
        density,
        dtype=np.float64,
    )

    if density_array.shape != config.density_shape:
        raise ValueError(
            f"density must have shape {config.density_shape}, "
            f"received {density_array.shape}."
        )

    if not np.all(
        np.isfinite(density_array)
    ):
        raise ValueError(
            "density contains NaN or infinite values."
        )

    receiver_grid = ReceiverGrid(
        x=config.observation_x_m,
        y=config.observation_y_m,
        z=np.asarray(
            [
                config.observation_z_m,
            ],
            dtype=np.float64,
        ),
    )
    forward_model = FWD3DGravityForwardModel(
        model_grid=model_grid_from_grid_spec(
            config.grid_spec()
        ),
        receiver_grid=receiver_grid,
        channel=config.gravity_channel,
        receiver_chunk_size=(
            config.receiver_chunk_size
        ),
    )
    gravity_volume = forward_model.calculate(
        density_array
    )

    if gravity_volume.shape != (
        1,
        config.observation_y_m.size,
        config.observation_x_m.size,
    ):
        raise RuntimeError(
            "Single-plane solver returned unexpected shape "
            f"{gravity_volume.shape}."
        )

    return np.asarray(
        gravity_volume[0],
        dtype=np.float64,
    )


def _density_extents(
    config: SinglePlaneReviewConfig,
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    """Return plan, XZ, and YZ physical image extents."""

    x_min, x_max = config.density_x_edges_m
    y_min, y_max = config.density_y_edges_m
    z_min, z_max = config.density_z_edges_m
    return (
        (
            x_min,
            x_max,
            y_min,
            y_max,
        ),
        (
            x_min,
            x_max,
            z_max,
            z_min,
        ),
        (
            y_min,
            y_max,
            z_max,
            z_min,
        ),
    )


def compute_common_gravity_limits(
    gravity_maps: Sequence[npt.ArrayLike],
) -> tuple[float, float]:
    """
    Return common plotting limits containing every gravity value.

    Parameters
    ----------
    gravity_maps
        Nonempty collection of finite gravity maps.

    Returns
    -------
    tuple
        Global minimum and maximum in the maps' gravity units.
    """

    if not gravity_maps:
        raise ValueError(
            "At least one gravity map is required."
        )

    arrays = [
        np.asarray(
            gravity,
            dtype=np.float64,
        )
        for gravity in gravity_maps
    ]

    if any(
        array.ndim != 2
        or not np.all(
            np.isfinite(array)
        )
        for array in arrays
    ):
        raise ValueError(
            "Gravity maps must be finite two-dimensional arrays."
        )

    return (
        min(
            float(
                np.min(array)
            )
            for array in arrays
        ),
        max(
            float(
                np.max(array)
            )
            for array in arrays
        ),
    )


def compute_individual_gravity_limits(
    gravity: npt.ArrayLike,
) -> tuple[float, float]:
    """Return the finite data range for one gravity map."""

    array = np.asarray(gravity, dtype=np.float64)
    if (
        array.ndim != 2
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(
            "Gravity must be a finite two-dimensional array."
        )
    return float(np.min(array)), float(np.max(array))


def gravity_range_statistics(
    gravity: npt.ArrayLike,
) -> dict[str, float]:
    """Return exact minimum, maximum, and peak absolute Gz values."""

    minimum, maximum = compute_individual_gravity_limits(gravity)
    return {
        "minimum": minimum,
        "maximum": maximum,
        "peak_amplitude": max(abs(minimum), abs(maximum)),
    }


def _nondegenerate_plot_limits(
    limits: tuple[float, float],
) -> tuple[float, float]:
    """Expand a constant range only enough for stable linear plotting."""

    minimum, maximum = limits
    if minimum != maximum:
        return minimum, maximum
    padding = max(abs(minimum) * 1.0e-6, 1.0e-12)
    return minimum - padding, maximum + padding


def format_gravity_range_text(
    gravity: npt.ArrayLike,
    *,
    unit: str,
) -> str:
    """Format array-derived numerical Gz range information."""

    statistics = gravity_range_statistics(gravity)
    return (
        f"Peak Gz: {statistics['peak_amplitude']:.3g} {unit}"
    )


def observation_marker_coordinates(
    config: SinglePlaneReviewConfig,
    *,
    stride: int,
) -> tuple[FloatArray, FloatArray]:
    """
    Return configured observation points subsampled for display.

    Parameters
    ----------
    config
        Active single-plane physical configuration.
    stride
        Positive index stride applied in X and Y.

    Returns
    -------
    tuple
        Flattened X and Y marker coordinates in meters.
    """

    if stride < 1:
        raise ValueError(
            "Observation marker stride must be at least one."
        )

    marker_x, marker_y = np.meshgrid(
        config.observation_x_m[
            ::stride
        ],
        config.observation_y_m[
            ::stride
        ],
        indexing="xy",
    )
    return (
        marker_x.ravel(),
        marker_y.ravel(),
    )


def format_body_parameter_text(
    body_metadata: dict[str, object],
    *,
    density_unit: str,
) -> str:
    """
    Format a compact parameter box from saved body metadata.

    Parameters
    ----------
    body_metadata
        Mapping returned by ``SinglePlaneBody.to_metadata``.
    density_unit
        Density unit label.

    Returns
    -------
    str
        Presentation-ready parameter text.
    """

    return "\n".join(
        (
            "Body parameters",
            f"Center X: {float(body_metadata['center_x_m']):g} m",
            f"Center Y: {float(body_metadata['center_y_m']):g} m",
            f"Top depth: {float(body_metadata['top_depth_m']):g} m",
            f"Bottom depth: {float(body_metadata['bottom_depth_m']):g} m",
            f"Width X: {float(body_metadata['width_x_m']):g} m",
            f"Width Y: {float(body_metadata['width_y_m']):g} m",
            f"Thickness: {float(body_metadata['thickness_m']):g} m",
            "Density contrast: "
            f"{float(body_metadata['density_contrast_g_cm3']):g} "
            f"{density_unit}",
        )
    )


def gravity_panel_title(
    config: SinglePlaneReviewConfig,
) -> str:
    """Return the precise common-scale gravity-panel title."""

    return (
        f"Common-scale vertical gravity anomaly "
        f"{config.gravity_component}\n"
        f"Observation plane: z = {config.observation_z_m:g} m"
    )


def individual_gravity_panel_title(
    config: SinglePlaneReviewConfig,
) -> str:
    """Return the per-example linear gravity-panel title."""

    return (
        f"Individually scaled vertical gravity anomaly "
        f"{config.gravity_component}\n"
        "Shape/extent view; colors are not comparable between examples"
    )


def plot_single_plane_example(
    density: npt.ArrayLike,
    gravity: npt.ArrayLike,
    body: SinglePlaneBody,
    config: SinglePlaneReviewConfig,
    output_directory: Path,
    *,
    density_maximum: float,
    gravity_limits: tuple[float, float],
    observation_marker_stride: int,
) -> None:
    """Create all publication figures for one review example."""

    density_array = np.asarray(
        density,
        dtype=np.float64,
    )
    gravity_array = np.asarray(
        gravity,
        dtype=np.float64,
    )
    plan_extent, xz_extent, yz_extent = (
        _density_extents(
            config
        )
    )
    gravity_extent = (
        config.observation_x_m[0],
        config.observation_x_m[-1],
        config.observation_y_m[0],
        config.observation_y_m[-1],
    )
    marker_x, marker_y = (
        observation_marker_coordinates(
            config,
            stride=observation_marker_stride,
        )
    )
    individual_gravity_limits = (
        compute_individual_gravity_limits(
            gravity_array
        )
    )
    individual_plot_limits = (
        _nondegenerate_plot_limits(
            individual_gravity_limits
        )
    )
    common_plot_limits = _nondegenerate_plot_limits(
        gravity_limits
    )
    gravity_range_text = format_gravity_range_text(
        gravity_array,
        unit=config.gravity_unit,
    )
    x_centers = np.linspace(
        config.density_x_min_center_m,
        config.density_x_max_center_m,
        config.nx,
    )
    y_centers = np.linspace(
        config.density_y_min_center_m,
        config.density_y_max_center_m,
        config.ny,
    )
    x_index = int(
        np.argmin(
            np.abs(
                x_centers
                - body.center_x_m
            )
        )
    )
    y_index = int(
        np.argmin(
            np.abs(
                y_centers
                - body.center_y_m
            )
        )
    )
    plan = np.max(
        density_array,
        axis=0,
    )
    xz = density_array[
        :,
        y_index,
        :,
    ]
    yz = density_array[
        :,
        :,
        x_index,
    ]

    def save_single(
        values: FloatArray,
        *,
        extent: tuple[float, float, float, float],
        title: str,
        x_label: str,
        y_label: str,
        output_name: str,
        cmap: str,
        vmin: float,
        vmax: float,
        colorbar_label: str,
        origin: str,
        show_observation_points: bool = False,
        annotation: str | None = None,
    ) -> None:
        figure, axis = plt.subplots(
            figsize=(7.0, 6.0)
        )
        image = axis.imshow(
            values,
            origin=origin,
            extent=extent,
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        if show_observation_points:
            axis.scatter(
                marker_x,
                marker_y,
                s=7.0,
                marker="o",
                facecolors="none",
                edgecolors="white",
                linewidths=0.35,
                alpha=0.55,
                label=(
                    "Observation locations "
                    f"(every {observation_marker_stride}th point)"
                ),
            )
            axis.legend(
                loc="lower right",
                fontsize=7,
                framealpha=0.75,
            )
            axis.set_xlim(config.gravity_display_xlim)
            axis.set_ylim(config.gravity_display_ylim)
        if annotation is not None:
            axis.text(
                0.02,
                0.98,
                annotation,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "black",
                    "edgecolor": "white",
                    "alpha": 0.65,
                },
            )
        figure.colorbar(
            image,
            ax=axis,
            label=colorbar_label,
        )
        figure.tight_layout()
        figure.savefig(
            output_directory
            / output_name,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)

    save_single(
        plan,
        extent=plan_extent,
        title=(
            "Horizontal density projection\n"
            "Maximum density over depth"
        ),
        x_label="X (m)",
        y_label="Y (m)",
        output_name="density_plan_view.png",
        cmap="viridis",
        vmin=0.0,
        vmax=density_maximum,
        colorbar_label=(
            f"Density contrast ({config.density_unit})"
        ),
        origin="lower",
    )
    save_single(
        xz,
        extent=xz_extent,
        title=(
            f"X-Z section at y = {body.center_y_m:g} m"
        ),
        x_label="X (m)",
        y_label="Depth (m)",
        output_name="density_xz_section.png",
        cmap="viridis",
        vmin=0.0,
        vmax=density_maximum,
        colorbar_label=(
            f"Density contrast ({config.density_unit})"
        ),
        origin="upper",
    )
    save_single(
        yz,
        extent=yz_extent,
        title=(
            f"Y-Z section at x = {body.center_x_m:g} m"
        ),
        x_label="Y (m)",
        y_label="Depth (m)",
        output_name="density_yz_section.png",
        cmap="viridis",
        vmin=0.0,
        vmax=density_maximum,
        colorbar_label=(
            f"Density contrast ({config.density_unit})"
        ),
        origin="upper",
    )
    save_single(
        gravity_array,
        extent=gravity_extent,
        title=gravity_panel_title(
            config
        ),
        x_label="X (m)",
        y_label="Y (m)",
        output_name="gravity_map.png",
        cmap="magma",
        vmin=common_plot_limits[0],
        vmax=common_plot_limits[1],
        colorbar_label=(
            "Vertical gravity component "
            f"{config.gravity_component} ({config.gravity_unit})"
        ),
        origin="lower",
        show_observation_points=True,
        annotation=gravity_range_text,
    )
    save_single(
        gravity_array,
        extent=gravity_extent,
        title=individual_gravity_panel_title(
            config
        ),
        x_label="X (m)",
        y_label="Y (m)",
        output_name="gravity_map_individual_scale.png",
        cmap="magma",
        vmin=individual_plot_limits[0],
        vmax=individual_plot_limits[1],
        colorbar_label=(
            "Vertical gravity component "
            f"{config.gravity_component} ({config.gravity_unit})"
        ),
        origin="lower",
        show_observation_points=True,
        annotation=gravity_range_text,
    )

    figure = plt.figure(
        figsize=(16.0, 10.0),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        2,
        4,
        height_ratios=(0.85, 1.15),
        width_ratios=(1.0, 0.72, 1.0, 1.0),
    )
    parameter_axis = figure.add_subplot(
        grid[0, 1]
    )
    density_axes = [
        figure.add_subplot(grid[0, index])
        for index in (0, 2, 3)
    ]
    gravity_axes = [
        figure.add_subplot(grid[1, 0:2]),
        figure.add_subplot(grid[1, 2:4]),
    ]

    parameter_axis.axis("off")
    parameter_axis.text(
        0.5,
        0.5,
        format_body_parameter_text(
            body.to_metadata(),
            density_unit=config.density_unit,
        ),
        ha="center",
        va="center",
        fontsize=8.8,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#f7f7f7",
            "edgecolor": "0.35",
            "alpha": 1.0,
        },
    )

    density_panels = (
        (
            plan,
            plan_extent,
            "Horizontal density projection\n"
            "Maximum density over depth",
            "X (m)",
            "Y (m)",
            "lower",
        ),
        (
            xz,
            xz_extent,
            "X-Z section",
            "X (m)",
            "Depth (m)",
            "upper",
        ),
        (
            yz,
            yz_extent,
            "Y-Z section",
            "Y (m)",
            "Depth (m)",
            "upper",
        ),
    )
    for axis, panel in zip(
        density_axes,
        density_panels,
        strict=True,
    ):
        values, extent, title, x_label, y_label, origin = panel
        image = axis.imshow(
            values,
            origin=origin,
            extent=extent,
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=density_maximum,
        )
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        figure.colorbar(
            image,
            ax=axis,
            shrink=0.82,
            label=config.density_unit,
        )

    gravity_panels = (
        (
            gravity_axes[0],
            gravity_panel_title(config),
            common_plot_limits,
        ),
        (
            gravity_axes[1],
            individual_gravity_panel_title(config),
            individual_plot_limits,
        ),
    )
    for axis, title, limits in gravity_panels:
        image = axis.imshow(
            gravity_array,
            origin="lower",
            extent=gravity_extent,
            aspect="equal",
            cmap="magma",
            vmin=limits[0],
            vmax=limits[1],
        )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.scatter(
            marker_x,
            marker_y,
            s=5.0,
            marker="o",
            facecolors="none",
            edgecolors="white",
            linewidths=0.3,
            alpha=0.5,
        )
        axis.set_xlim(config.gravity_display_xlim)
        axis.set_ylim(config.gravity_display_ylim)
        axis.text(
            0.02,
            0.98,
            gravity_range_text,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="white",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "black",
                "edgecolor": "white",
                "alpha": 0.65,
            },
        )
        figure.colorbar(
            image,
            ax=axis,
            shrink=0.88,
            label=config.gravity_unit,
        )

    figure.suptitle(
        format_single_plane_example_title(body),
        fontsize=16,
    )
    figure.savefig(
        output_directory
        / "model_gravity_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def observation_coordinate_metadata(
    config: SinglePlaneReviewConfig,
) -> dict[str, object]:
    """Return exact configured observation-coordinate information."""

    x = config.observation_x_m
    y = config.observation_y_m
    return {
        "observation_plane_z_m": config.observation_z_m,
        "x_coordinates_m": x.tolist(),
        "y_coordinates_m": y.tolist(),
        "x_min_m": float(x[0]),
        "x_max_m": float(x[-1]),
        "y_min_m": float(y[0]),
        "y_max_m": float(y[-1]),
        "x_spacing_m": float(x[1] - x[0]),
        "y_spacing_m": float(y[1] - y[0]),
        "number_of_x_points": int(x.size),
        "number_of_y_points": int(y.size),
        "total_observation_points": int(x.size * y.size),
        "total_x_extent_m": float(x[-1] - x[0]),
        "total_y_extent_m": float(y[-1] - y[0]),
    }


def plot_observation_coordinates_summary(
    config: SinglePlaneReviewConfig,
    output_path: Path,
) -> None:
    """Create a presentation summary of observation coordinates."""

    metadata = observation_coordinate_metadata(config)
    figure, axis = plt.subplots(figsize=(12.0, 6.75))
    axis.set_axis_off()
    lines = (
        f"Observation plane: z = {metadata['observation_plane_z_m']:g} m",
        (
            f"x = {metadata['x_min_m']:g}, "
            f"{metadata['x_min_m']:g} + "
            f"{metadata['x_spacing_m']:g}, …, "
            f"{metadata['x_max_m']:g} m"
        ),
        (
            f"y = {metadata['y_min_m']:g}, "
            f"{metadata['y_min_m']:g} + "
            f"{metadata['y_spacing_m']:g}, …, "
            f"{metadata['y_max_m']:g} m"
        ),
        f"Spacing: Δx = {metadata['x_spacing_m']:g} m; "
        f"Δy = {metadata['y_spacing_m']:g} m",
        f"Grid: {metadata['number_of_x_points']} × "
        f"{metadata['number_of_y_points']} points",
        f"Total observation points: "
        f"{metadata['total_observation_points']:,}",
        f"Coordinate extents: "
        f"{metadata['total_x_extent_m']:g} m × "
        f"{metadata['total_y_extent_m']:g} m",
    )
    axis.text(
        0.5,
        0.5,
        "\n\n".join(lines),
        ha="center",
        va="center",
        fontsize=16,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=1.0",
            "facecolor": "#f5f8fb",
            "edgecolor": "#376d99",
            "linewidth": 1.5,
        },
    )
    figure.suptitle(
        "Single-Plane Observation Coordinates",
        fontsize=21,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_comparison_row(
    figure: plt.Figure,
    axes: Sequence[plt.Axes],
    density: FloatArray,
    gravity: FloatArray,
    body: SinglePlaneBody,
    config: SinglePlaneReviewConfig,
    common_limits: tuple[float, float],
    label: str,
) -> None:
    """Plot one density/common-Gz/individual-Gz comparison row."""

    plan_extent = _density_extents(config)[0]
    gravity_extent = (
        config.observation_x_m[0],
        config.observation_x_m[-1],
        config.observation_y_m[0],
        config.observation_y_m[-1],
    )
    marker_x, marker_y = observation_marker_coordinates(
        config,
        stride=config.observation_marker_stride,
    )
    individual = _nondegenerate_plot_limits(
        compute_individual_gravity_limits(gravity)
    )
    density_image = axes[0].imshow(
        np.max(density, axis=0),
        origin="lower",
        extent=plan_extent,
        aspect="equal",
        cmap="viridis",
        vmin=0.0,
        vmax=config.maximum_density_contrast_g_cm3,
    )
    axes[0].set_title(f"{label}\nDensity plan")
    axes[0].text(
        0.02,
        0.98,
        f"Center x = {body.center_x_m:g} m",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
        color="white",
    )
    figure.colorbar(density_image, ax=axes[0], shrink=0.75)
    for axis, limits, title in (
        (axes[1], common_limits, "Common-scale Gz"),
        (axes[2], individual, "Individual-scale Gz"),
    ):
        image = axis.imshow(
            gravity,
            origin="lower",
            extent=gravity_extent,
            aspect="equal",
            cmap="magma",
            vmin=limits[0],
            vmax=limits[1],
        )
        axis.scatter(
            marker_x,
            marker_y,
            s=3,
            facecolors="none",
            edgecolors="white",
            linewidths=0.25,
            alpha=0.45,
        )
        axis.set_xlim(config.gravity_display_xlim)
        axis.set_ylim(config.gravity_display_ylim)
        axis.set_title(
            f"{title}\n"
            f"{format_gravity_range_text(gravity, unit=config.gravity_unit)}"
        )
        figure.colorbar(image, ax=axis, shrink=0.75)
    for axis in axes:
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")


def plot_left_right_position_comparison(
    config: SinglePlaneReviewConfig,
    bodies: Sequence[SinglePlaneBody],
    densities: dict[str, FloatArray],
    gravity_maps: dict[str, FloatArray],
    common_limits: tuple[float, float],
    output_path: Path,
) -> None:
    """Compare symmetric left- and right-shifted controlled bodies."""

    figure, axes = plt.subplots(2, 3, figsize=(13.5, 9.0))
    for row, body in enumerate(bodies):
        _plot_comparison_row(
            figure,
            axes[row],
            densities[body.name],
            gravity_maps[body.name],
            body,
            config,
            common_limits,
            "Left shifted" if row == 0 else "Right shifted",
        )
    figure.suptitle(
        "Horizontal-Position Control: Only Center X Changes",
        fontsize=17,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_controlled_examples_comparison(
    config: SinglePlaneReviewConfig,
    bodies: Sequence[SinglePlaneBody],
    densities: dict[str, FloatArray],
    gravity_maps: dict[str, FloatArray],
    common_limits: tuple[float, float],
    output_path: Path,
) -> None:
    """Create a compact comparison of controlled examples A through F."""

    figure, axes = plt.subplots(
        len(bodies),
        4,
        figsize=(14.5, 3.0 * len(bodies)),
    )
    baseline = bodies[0]
    for row, body in enumerate(bodies):
        _plot_comparison_row(
            figure,
            axes[row, :3],
            densities[body.name],
            gravity_maps[body.name],
            body,
            config,
            common_limits,
            body.name.replace("controlled_example_", "").replace("_", " ").title(),
        )
        axes[row, 3].axis("off")
        change = controlled_example_change(body)
        parameter = change[
            "changed_parameter_relative_to_baseline"
        ]
        if parameter == "none":
            change_text = "Reference baseline"
        elif parameter == "top_depth_m":
            change_text = (
                f"Top depth\n{baseline.top_depth_m:g} → "
                f"{body.top_depth_m:g} m"
            )
        elif parameter == "horizontal_size":
            change_text = (
                "Width X and Y\n"
                f"{baseline.width_x_m:g} → {body.width_x_m:g} m"
            )
        elif parameter == "density_contrast_g_cm3":
            change_text = (
                "Density contrast\n"
                f"{baseline.density_contrast_g_cm3:g} → "
                f"{body.density_contrast_g_cm3:g} "
                f"{config.density_unit}"
            )
        elif parameter == "thickness_m":
            change_text = (
                f"Thickness\n{baseline.thickness_m:g} → "
                f"{body.thickness_m:g} m"
            )
        else:
            change_text = (
                "Maximum combined\n"
                "depth, size, thickness, density"
            )
        axes[row, 3].text(
            0.05,
            0.5,
            "Changed relative to baseline\n\n"
            f"{change_text}",
            va="center",
            fontsize=10,
        )
    figure.suptitle(
        "Controlled Single-Plane Physical Effects",
        fontsize=18,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_coordinate_system_geometry(
    config: SinglePlaneReviewConfig,
    body: SinglePlaneBody,
    output_path: Path,
) -> None:
    """
    Plot the coordinate system, domains, observation plane, and body.

    Parameters
    ----------
    config
        Active physical configuration.
    body
        Representative buried body shown in the schematic.
    output_path
        Presentation figure destination.
    """

    figure, (
        plan_axis,
        section_axis,
    ) = plt.subplots(
        1,
        2,
        figsize=(14.0, 6.0),
    )
    observation_x_min = float(
        config.observation_x_m[0]
    )
    observation_x_max = float(
        config.observation_x_m[-1]
    )
    observation_y_min = float(
        config.observation_y_m[0]
    )
    observation_y_max = float(
        config.observation_y_m[-1]
    )
    density_x_min, density_x_max = (
        config.density_x_edges_m
    )
    density_y_min, density_y_max = (
        config.density_y_edges_m
    )
    _, density_z_max = (
        config.density_z_edges_m
    )

    plan_axis.add_patch(
        plt.Rectangle(
            (
                observation_x_min,
                observation_y_min,
            ),
            observation_x_max
            - observation_x_min,
            observation_y_max
            - observation_y_min,
            facecolor="#d9ecff",
            edgecolor="#2468a2",
            linewidth=2.0,
            label="Observation area (800 m × 800 m)",
        )
    )
    marker_x, marker_y = observation_marker_coordinates(
        config,
        stride=config.observation_marker_stride,
    )
    plan_axis.scatter(
        marker_x,
        marker_y,
        s=5,
        facecolors="none",
        edgecolors="#2468a2",
        linewidths=0.4,
        alpha=0.6,
        label="Observation points (display subsample)",
    )
    plan_axis.add_patch(
        plt.Rectangle(
            (
                density_x_min,
                density_y_min,
            ),
            density_x_max
            - density_x_min,
            density_y_max
            - density_y_min,
            facecolor="#e8e8e8",
            edgecolor="#333333",
            linewidth=2.0,
            label="Density model (640 m × 640 m)",
        )
    )
    plan_axis.add_patch(
        plt.Rectangle(
            (
                body.x_bounds_m[0],
                body.y_bounds_m[0],
            ),
            body.width_x_m,
            body.width_y_m,
            facecolor="#d95f02",
            edgecolor="#8c2d04",
            alpha=0.85,
            label="Example buried body footprint",
        )
    )
    plan_axis.arrow(
        density_x_min,
        density_y_min,
        110.0,
        0.0,
        width=4.0,
        head_width=20.0,
        head_length=25.0,
        color="black",
        length_includes_head=True,
    )
    plan_axis.text(
        density_x_min + 125.0,
        density_y_min,
        "+X east",
        va="center",
    )
    plan_axis.arrow(
        density_x_min,
        density_y_min,
        0.0,
        110.0,
        width=4.0,
        head_width=20.0,
        head_length=25.0,
        color="black",
        length_includes_head=True,
    )
    plan_axis.text(
        density_x_min,
        density_y_min + 130.0,
        "+Y north",
        ha="center",
    )
    plan_axis.set_xlim(
        observation_x_min - 40.0,
        observation_x_max + 40.0,
    )
    plan_axis.set_ylim(
        observation_y_min - 40.0,
        observation_y_max + 40.0,
    )
    plan_axis.set_aspect(
        "equal"
    )
    plan_axis.set_xlabel(
        "X (m)"
    )
    plan_axis.set_ylabel(
        "Y (m)"
    )
    plan_axis.set_title(
        "Plan view: centered observation and density domains"
    )
    plan_axis.legend(
        loc="upper right",
        fontsize=8,
    )
    plan_axis.grid(
        alpha=0.2
    )

    section_axis.add_patch(
        plt.Rectangle(
            (
                density_x_min,
                0.0,
            ),
            density_x_max
            - density_x_min,
            density_z_max,
            facecolor="#e8e8e8",
            edgecolor="#333333",
            linewidth=2.0,
            label="Density model: 640 m × 240 m",
        )
    )
    section_axis.add_patch(
        plt.Rectangle(
            (
                body.x_bounds_m[0],
                body.top_depth_m,
            ),
            body.width_x_m,
            body.thickness_m,
            facecolor="#d95f02",
            edgecolor="#8c2d04",
            alpha=0.85,
            label="Example buried body",
        )
    )
    section_axis.plot(
        [
            observation_x_min,
            observation_x_max,
        ],
        [
            config.observation_z_m,
            config.observation_z_m,
        ],
        color="#2468a2",
        linewidth=3.0,
        marker="o",
        markevery=8,
        markersize=3.0,
        label=(
            "Horizontal observation plane: "
            f"z = {config.observation_z_m:g} m"
        ),
    )
    section_axis.annotate(
        "+Z / positive depth",
        xy=(
            density_x_min + 35.0,
            150.0,
        ),
        xytext=(
            density_x_min + 35.0,
            45.0,
        ),
        arrowprops={
            "arrowstyle": "-|>",
            "linewidth": 2.0,
            "color": "black",
        },
        ha="center",
    )
    section_axis.annotate(
        f"Top depth = {body.top_depth_m:g} m",
        xy=(body.center_x_m, body.top_depth_m),
        xytext=(body.center_x_m + 105.0, body.top_depth_m - 5.0),
        arrowprops={"arrowstyle": "->", "color": "#8c2d04"},
        fontsize=8,
    )
    section_axis.text(
        body.center_x_m,
        body.top_depth_m + body.thickness_m / 2.0,
        f"width = {body.width_x_m:g} m\n"
        f"thickness = {body.thickness_m:g} m",
        ha="center",
        va="center",
        fontsize=8,
    )
    section_axis.text(
        (observation_x_min + observation_x_max) / 2.0,
        -18.0,
        f"Observation-plane width = "
        f"{observation_x_max - observation_x_min:g} m; "
        f"elevation z = {config.observation_z_m:g} m",
        ha="center",
        fontsize=8,
    )
    section_axis.set_xlim(
        observation_x_min - 40.0,
        observation_x_max + 40.0,
    )
    section_axis.set_ylim(
        density_z_max + 30.0,
        -30.0,
    )
    section_axis.set_aspect(
        "equal"
    )
    section_axis.set_xlabel(
        "X (m)"
    )
    section_axis.set_ylabel(
        "Depth Z (m; positive downward)"
    )
    section_axis.set_title(
        "Vertical section through the model center"
    )
    section_axis.legend(
        loc="lower right",
        fontsize=8,
    )
    section_axis.grid(
        alpha=0.2
    )

    figure.suptitle(
        "Single-Plane Gravity Review: Coordinate System and Geometry",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Depth coordinate is positive downward.",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def plot_model_geometry_summary(
    config: SinglePlaneReviewConfig,
    examples: Sequence[SinglePlaneBody],
    output_path: Path,
) -> None:
    """
    Create a presentation-ready physical-geometry summary.

    All displayed values are derived from the active configuration and
    deterministic example collection.
    """

    observation_extent_x = float(
        config.observation_x_m[-1]
        - config.observation_x_m[0]
    )
    observation_extent_y = float(
        config.observation_y_m[-1]
        - config.observation_y_m[0]
    )
    density_extent_x = (
        config.density_x_edges_m[1]
        - config.density_x_edges_m[0]
    )
    density_extent_y = (
        config.density_y_edges_m[1]
        - config.density_y_edges_m[0]
    )
    density_extent_z = (
        config.density_z_edges_m[1]
        - config.density_z_edges_m[0]
    )
    maximum_width = max(
        max(
            body.width_x_m
            for body in examples
        ),
        max(
            body.width_y_m
            for body in examples
        ),
    )
    maximum_bottom = max(
        body.bottom_depth_m
        for body in examples
    )
    sections = (
        (
            "DENSITY MODEL",
            "\n".join(
                (
                    f"Grid shape: {config.nz} × {config.ny} × {config.nx} (z, y, x)",
                    f"Cell dimensions: {config.dz_m:g} × {config.dy_m:g} × {config.dx_m:g} m",
                    f"Physical extent: {density_extent_x:g} × {density_extent_y:g} × {density_extent_z:g} m",
                    "Background density contrast: 0.0 "
                    f"{config.density_unit}",
                    "Bodies: one positive rectangular parallelepiped",
                )
            ),
        ),
        (
            "OBSERVATION PLANE",
            "\n".join(
                (
                    f"X range: {config.observation_x_m[0]:g} to {config.observation_x_m[-1]:g} m",
                    f"Y range: {config.observation_y_m[0]:g} to {config.observation_y_m[-1]:g} m",
                    f"Z coordinate: {config.observation_z_m:g} m (physical surface)",
                    f"Grid shape: {config.observation_y_m.size} × {config.observation_x_m.size}",
                    f"Point spacing: {config.observation_spacing_m:g} m",
                    f"Coordinate extent: {observation_extent_x:g} × {observation_extent_y:g} m",
                )
            ),
        ),
        (
            "BODY PARAMETER RANGES",
            "\n".join(
                (
                    f"Top depth: {config.minimum_top_depth_m:g}–{config.maximum_top_depth_m:g} m",
                    f"Bottom depth: ≤ {config.maximum_bottom_depth_m:g} m",
                    f"Width X: {config.minimum_width_x_m:g}–{config.maximum_width_x_m:g} m",
                    f"Width Y: {config.minimum_width_y_m:g}–{config.maximum_width_y_m:g} m",
                    f"Thickness: {config.minimum_thickness_m:g}–{config.maximum_thickness_m:g} m",
                    "Density contrast: "
                    f"{config.minimum_density_contrast_g_cm3:g}–"
                    f"{config.maximum_density_contrast_g_cm3:g} "
                    f"{config.density_unit}",
                    "Horizontal position: centered as closely as "
                    "cell-edge alignment permits",
                )
            ),
        ),
        (
            "SPATIAL-SIZE VALIDATION",
            "\n".join(
                (
                    f"Observation extent = {observation_extent_x:g} m",
                    "Maximum body width = "
                    f"{maximum_width:g} m → 5 × width = "
                    f"{5.0 * maximum_width:g} m",
                    "Maximum body bottom depth = "
                    f"{maximum_bottom:g} m → 5 × depth = "
                    f"{5.0 * maximum_bottom:g} m",
                    f"{observation_extent_x:g} ≥ "
                    f"{5.0 * maximum_width:g}: PASS",
                    f"{observation_extent_x:g} ≥ "
                    f"{5.0 * maximum_bottom:g}: PASS",
                )
            ),
        ),
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 8.0),
    )

    for axis, (
        heading,
        text,
    ) in zip(
        axes.ravel(),
        sections,
        strict=True,
    ):
        axis.set_axis_off()
        axis.text(
            0.03,
            0.94,
            heading,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="#174a75",
        )
        axis.text(
            0.03,
            0.82,
            text,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            linespacing=1.55,
        )
        axis.add_patch(
            plt.Rectangle(
                (
                    0.0,
                    0.0,
                ),
                1.0,
                1.0,
                transform=axis.transAxes,
                fill=False,
                edgecolor="#8aa9c2",
                linewidth=1.4,
            )
        )

    figure.suptitle(
        "Proposed Single-Plane Gravity Modeling Geometry",
        fontsize=18,
        fontweight="bold",
    )
    figure.tight_layout(
        rect=(
            0.02,
            0.02,
            0.98,
            0.94,
        )
    )
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def write_geometry_summary(
    output_path: Path,
    config: SinglePlaneReviewConfig,
    examples: Sequence[SinglePlaneBody],
    *,
    common_gravity_limits: tuple[float, float],
) -> None:
    """Write a human-readable Markdown geometry summary."""

    observation_width = (
        config.observation_x_m[-1]
        - config.observation_x_m[0]
    )
    maximum_dimension = max(
        max(
            body.width_x_m
            for body in examples
        ),
        max(
            body.width_y_m
            for body in examples
        ),
        max(
            body.bottom_depth_m
            for body in examples
        ),
    )
    lines = [
        "# Single-Plane Gravity Review Geometry",
        "",
        "## Coordinate and unit conventions",
        "",
        "- X: positive east",
        "- Y: positive north",
        "- Z/depth: positive downward",
        "- Density array: `density[z, y, x]`",
        "- Gravity array: `gravity[y, x]`",
        f"- Density contrast: {config.density_unit}",
        "- Gravity: vertical gravity anomaly "
        f"{config.gravity_component} in {config.gravity_unit}",
        "",
        "## Density domain",
        "",
        f"- Grid shape: {config.nz} × {config.ny} × {config.nx}",
        f"- Cell spacing: {config.dx_m:g} × {config.dy_m:g} × {config.dz_m:g} m",
        f"- X physical edges: {config.density_x_edges_m[0]:g} to {config.density_x_edges_m[1]:g} m",
        f"- Y physical edges: {config.density_y_edges_m[0]:g} to {config.density_y_edges_m[1]:g} m",
        f"- Depth physical edges: {config.density_z_edges_m[0]:g} to {config.density_z_edges_m[1]:g} m",
        "",
        "## Observation plane",
        "",
        f"- X coordinates: {config.observation_x_m[0]:g} to {config.observation_x_m[-1]:g} m",
        f"- Y coordinates: {config.observation_y_m[0]:g} to {config.observation_y_m[-1]:g} m",
        f"- Z coordinate: {config.observation_z_m:g} m",
        f"- Shape: {config.observation_y_m.size} × {config.observation_x_m.size}",
        f"- Number of points: {config.observation_x_m.size * config.observation_y_m.size:,}",
        f"- Spacing: {config.observation_spacing_m:g} m",
        f"- Coordinate span: {observation_width:g} m",
        "- Figure markers: actual observation locations, displayed "
        f"every {config.observation_marker_stride}th point in X and Y",
        "- Gravity figure display limits (plotting only): "
        f"X {config.gravity_display_xlim[0]:g} to "
        f"{config.gravity_display_xlim[1]:g} m; "
        f"Y {config.gravity_display_ylim[0]:g} to "
        f"{config.gravity_display_ylim[1]:g} m",
        "- Saved gravity arrays, receiver coordinates, and forward-model "
        "domain retain the complete observation extent.",
        "",
        "## Common plotting scales",
        "",
        "- Density scale for every plan and section: "
        f"0.0 to {config.maximum_density_contrast_g_cm3:g} "
        f"{config.density_unit}",
        "- Common-scale vertical gravity anomaly Gz for every example: "
        f"{common_gravity_limits[0]:.12e} to "
        f"{common_gravity_limits[1]:.12e} "
        f"{config.gravity_unit}",
        "- Each example also includes a linearly, individually scaled "
        "Gz shape/extent panel. Its colors are not amplitude-comparable "
        "between examples.",
        "",
        "The horizontal density projection is the maximum density "
        "contrast at each X-Y location over the full depth dimension.",
        "",
        "## Five-times criterion",
        "",
        f"- Maximum width/length/bottom depth: {maximum_dimension:g} m",
        f"- Required extent: 5 × {maximum_dimension:g} = {5.0 * maximum_dimension:g} m",
        f"- Actual extent: {observation_width:g} m",
        "- Result: PASS",
        "",
        "## Deterministic examples",
        "",
        "| Example | Top (m) | Bottom (m) | Width X (m) | Width Y (m) | Thickness (m) | Density (g/cm³) | Center X (m) | Center Y (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for body in examples:
        lines.append(
            f"| {body.name} | {body.top_depth_m:g} | "
            f"{body.bottom_depth_m:g} | {body.width_x_m:g} | "
            f"{body.width_y_m:g} | {body.thickness_m:g} | "
            f"{body.density_contrast_g_cm3:g} | "
            f"{body.center_x_m:g} | {body.center_y_m:g} |"
        )

    output_path.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )


def save_json(
    output_path: Path,
    values: dict[str, object],
) -> None:
    """Save indented JSON metadata."""

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            values,
            output_file,
            indent=2,
        )
