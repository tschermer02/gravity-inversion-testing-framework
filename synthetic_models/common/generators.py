from __future__ import annotations

from collections.abc import Iterable
from random import Random
import numpy as np

from synthetic_models.common.bodies import RectangularBodySpec, MultiBodyCaseSpec, DippingBodySpec, SaltDomeSpec, BasementReliefSpec
from synthetic_models.common.grid import GridSpec


DEFAULT_X_WIDTH = 10
DEFAULT_Y_WIDTH = 10
DEFAULT_Z_THICKNESS = 5
DEFAULT_DENSITY_CONTRAST = 0.5


def generate_single_body(
    *,
    name: str,
    x_start: int,
    y_start: int,
    z_start: int,
    x_width: int = DEFAULT_X_WIDTH,
    y_width: int = DEFAULT_Y_WIDTH,
    z_thickness: int = DEFAULT_Z_THICKNESS,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    grid: GridSpec | None = None,
) -> RectangularBodySpec:
    """
    Create one rectangular constant-density body.

    The body is defined by its starting cell indices and its dimensions
    in grid cells.

    Parameters
    ----------
    name
        Unique case name.
    x_start, y_start, z_start
        Starting cell indices.
    x_width, y_width
        Horizontal body dimensions in cells.
    z_thickness
        Vertical body thickness in cells.
    density_contrast
        Constant density contrast assigned to the body.
    grid
        Optional grid used to validate that the body is within bounds.

    Returns
    -------
    RectangularBodySpec
        Rectangular body specification.
    """

    if not name.strip():
        raise ValueError("Body name must not be empty.")

    if x_width <= 0:
        raise ValueError("x_width must be greater than zero.")

    if y_width <= 0:
        raise ValueError("y_width must be greater than zero.")

    if z_thickness <= 0:
        raise ValueError("z_thickness must be greater than zero.")

    if density_contrast == 0.0:
        raise ValueError("density_contrast must not be zero.")

    body = RectangularBodySpec(
        name=name,
        x_start=x_start,
        x_end=x_start + x_width,
        y_start=y_start,
        y_end=y_start + y_width,
        z_start=z_start,
        z_end=z_start + z_thickness,
        density_contrast=density_contrast,
    )

    if grid is not None:
        body.validate(grid)

    return body

def generate_random_body(
    *,
    name: str,
    grid: GridSpec,
    rng: Random | None = None,
    x_width: int = DEFAULT_X_WIDTH,
    y_width: int = DEFAULT_Y_WIDTH,
    z_thickness: int = DEFAULT_Z_THICKNESS,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    edge_margin_x: int = 0,
    edge_margin_y: int = 0,
    edge_margin_z: int = 0,
) -> RectangularBodySpec:
    """
    Generate one random rectangular body that fits within the grid.

    Parameters
    ----------
    name
        Unique case name.
    grid
        Grid in which the body must fit.
    rng
        Optional ``random.Random`` instance for reproducibility.
    x_width, y_width, z_thickness
        Body dimensions in cells.
    density_contrast
        Constant density contrast.
    edge_margin_x, edge_margin_y, edge_margin_z
        Minimum empty-cell margins between the body and each pair of
        grid boundaries.

    Returns
    -------
    RectangularBodySpec
        Random valid body specification.
    """

    if rng is None:
        rng = Random()

    if edge_margin_x < 0:
        raise ValueError("edge_margin_x must be nonnegative.")

    if edge_margin_y < 0:
        raise ValueError("edge_margin_y must be nonnegative.")

    if edge_margin_z < 0:
        raise ValueError("edge_margin_z must be nonnegative.")

    x_start_min = edge_margin_x
    y_start_min = edge_margin_y
    z_start_min = edge_margin_z

    x_start_max = grid.nx - edge_margin_x - x_width
    y_start_max = grid.ny - edge_margin_y - y_width
    z_start_max = grid.nz - edge_margin_z - z_thickness

    if x_start_max < x_start_min:
        raise ValueError(
            "The requested x width and margin do not fit within the grid."
        )

    if y_start_max < y_start_min:
        raise ValueError(
            "The requested y width and margin do not fit within the grid."
        )

    if z_start_max < z_start_min:
        raise ValueError(
            "The requested z thickness and margin do not fit within the grid."
        )

    x_start = rng.randint(x_start_min, x_start_max)
    y_start = rng.randint(y_start_min, y_start_max)
    z_start = rng.randint(z_start_min, z_start_max)

    return generate_single_body(
        name=name,
        x_start=x_start,
        y_start=y_start,
        z_start=z_start,
        x_width=x_width,
        y_width=y_width,
        z_thickness=z_thickness,
        density_contrast=density_contrast,
        grid=grid,
    )

def generate_random_cases(
    *,
    number_of_cases: int,
    grid: GridSpec,
    seed: int | None = None,
    name_prefix: str = "random_body",
    x_width: int = DEFAULT_X_WIDTH,
    y_width: int = DEFAULT_Y_WIDTH,
    z_thickness: int = DEFAULT_Z_THICKNESS,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    edge_margin_x: int = 0,
    edge_margin_y: int = 0,
    edge_margin_z: int = 0,
) -> list[RectangularBodySpec]:
    """
    Generate multiple reproducible random rectangular-body cases.
    """

    if number_of_cases <= 0:
        raise ValueError("number_of_cases must be greater than zero.")

    rng = Random(seed)

    return [
        generate_random_body(
            name=f"{name_prefix}_{index:03d}",
            grid=grid,
            rng=rng,
            x_width=x_width,
            y_width=y_width,
            z_thickness=z_thickness,
            density_contrast=density_contrast,
            edge_margin_x=edge_margin_x,
            edge_margin_y=edge_margin_y,
            edge_margin_z=edge_margin_z,
        )
        for index in range(1, number_of_cases + 1)
    ]

def generate_multi_body_case(
    *,
    name: str,
    bodies: list[RectangularBodySpec],
    grid: GridSpec,
    allow_overlap: bool = False,
) -> MultiBodyCaseSpec:
    """Create and validate one case containing multiple bodies."""

    case = MultiBodyCaseSpec(
        name=name,
        bodies=tuple(bodies),
        allow_overlap=allow_overlap,
    )

    case.validate(grid)

    return case

def _rectangular_bodies_overlap(
    first: RectangularBodySpec,
    second: RectangularBodySpec,
) -> bool:
    """Return True when two rectangular bodies share at least one cell."""

    x_overlap = (
        first.x_start < second.x_end
        and second.x_start < first.x_end
    )

    y_overlap = (
        first.y_start < second.y_end
        and second.y_start < first.y_end
    )

    z_overlap = (
        first.z_start < second.z_end
        and second.z_start < first.z_end
    )

    return x_overlap and y_overlap and z_overlap

def generate_random_multi_body_case(
    *,
    name: str,
    number_of_bodies: int,
    grid: GridSpec,
    seed: int | None = None,
    minimum_x_width: int = 4,
    maximum_x_width: int = 12,
    minimum_y_width: int = 4,
    maximum_y_width: int = 12,
    minimum_z_thickness: int = 3,
    maximum_z_thickness: int = 7,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    edge_margin_x: int = 0,
    edge_margin_y: int = 0,
    edge_margin_z: int = 0,
    maximum_attempts_per_body: int = 1000,
) -> MultiBodyCaseSpec:
    """
    Generate one case containing randomly sized, nonoverlapping bodies.

    Each body receives a random size and position while remaining
    completely inside the grid. Bodies are not allowed to overlap.
    """

    if number_of_bodies <= 0:
        raise ValueError(
            "number_of_bodies must be greater than zero."
        )

    if minimum_x_width <= 0 or minimum_x_width > maximum_x_width:
        raise ValueError(
            "Invalid x-width limits."
        )

    if minimum_y_width <= 0 or minimum_y_width > maximum_y_width:
        raise ValueError(
            "Invalid y-width limits."
        )

    if (
        minimum_z_thickness <= 0
        or minimum_z_thickness > maximum_z_thickness
    ):
        raise ValueError(
            "Invalid z-thickness limits."
        )

    if maximum_attempts_per_body <= 0:
        raise ValueError(
            "maximum_attempts_per_body must be greater than zero."
        )

    for margin_name, margin in {
        "edge_margin_x": edge_margin_x,
        "edge_margin_y": edge_margin_y,
        "edge_margin_z": edge_margin_z,
    }.items():
        if margin < 0:
            raise ValueError(
                f"{margin_name} must be nonnegative."
            )

    rng = Random(seed)
    generated_bodies: list[RectangularBodySpec] = []

    for body_index in range(1, number_of_bodies + 1):
        body_created = False

        for _ in range(maximum_attempts_per_body):
            x_width = rng.randint(
                minimum_x_width,
                maximum_x_width,
            )

            y_width = rng.randint(
                minimum_y_width,
                maximum_y_width,
            )

            z_thickness = rng.randint(
                minimum_z_thickness,
                maximum_z_thickness,
            )

            x_start_min = edge_margin_x
            y_start_min = edge_margin_y
            z_start_min = edge_margin_z

            x_start_max = (
                grid.nx
                - edge_margin_x
                - x_width
            )

            y_start_max = (
                grid.ny
                - edge_margin_y
                - y_width
            )

            z_start_max = (
                grid.nz
                - edge_margin_z
                - z_thickness
            )

            if (
                x_start_max < x_start_min
                or y_start_max < y_start_min
                or z_start_max < z_start_min
            ):
                raise ValueError(
                    "The requested random-body size and margins "
                    "cannot fit inside the grid."
                )

            candidate = generate_single_body(
                name=f"{name}_body_{body_index:03d}",
                x_start=rng.randint(
                    x_start_min,
                    x_start_max,
                ),
                y_start=rng.randint(
                    y_start_min,
                    y_start_max,
                ),
                z_start=rng.randint(
                    z_start_min,
                    z_start_max,
                ),
                x_width=x_width,
                y_width=y_width,
                z_thickness=z_thickness,
                density_contrast=density_contrast,
                grid=grid,
            )

            overlaps_existing_body = any(
                _rectangular_bodies_overlap(
                    candidate,
                    existing_body,
                )
                for existing_body in generated_bodies
            )

            if overlaps_existing_body:
                continue

            generated_bodies.append(candidate)
            body_created = True
            break

        if not body_created:
            raise RuntimeError(
                f"{name}: could not place body {body_index} "
                f"without overlap after "
                f"{maximum_attempts_per_body} attempts. "
                "Reduce the number or maximum size of the bodies."
            )

    return generate_multi_body_case(
        name=name,
        bodies=generated_bodies,
        grid=grid,
        allow_overlap=False,
    )

def generate_dipping_body(
    *,
    name: str,
    center_x: float,
    center_y: float,
    center_z: float,
    strike_length: float,
    dip_length: float,
    thickness: float,
    strike_degrees: float,
    dip_degrees: float,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    grid: GridSpec | None = None,
) -> DippingBodySpec:
    """
    Create one inclined or elongated constant-density prism.

    All locations and dimensions are expressed in grid-cell coordinates.
    For example, a center coordinate of ``10.5`` corresponds to the center
    of cell index 10.

    Parameters
    ----------
    name
        Unique case name.
    center_x, center_y, center_z
        Continuous grid coordinates of the body's geometric center.
    strike_length
        Full body length along strike, in cells.
    dip_length
        Full body length down dip, in cells.
    thickness
        Full body thickness normal to the inclined plane, in cells.
    strike_degrees
        Strike measured clockwise from positive y.
    dip_degrees
        Downward dip angle from horizontal, between 0 and 90 degrees.
    density_contrast
        Constant density contrast inside the body.
    grid
        Optional grid used to validate that the full body fits.

    Returns
    -------
    DippingBodySpec
        Validated dipping-body specification.
    """
    body = DippingBodySpec(
        name=name,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        strike_length=strike_length,
        dip_length=dip_length,
        thickness=thickness,
        strike_degrees=strike_degrees,
        dip_degrees=dip_degrees,
        density_contrast=density_contrast,
    )

    if grid is not None:
        body.validate(grid)

    return body

def generate_dipping_body_from_shallow_center(
    *,
    name: str,
    shallow_center_x: float,
    shallow_center_y: float,
    shallow_center_z: float,
    strike_length: float,
    dip_length: float,
    thickness: float,
    strike_degrees: float,
    dip_degrees: float,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    grid: GridSpec | None = None,
) -> DippingBodySpec:
    """
    Create an inclined prism from the center of its shallow down-dip face.

    This constructor is convenient when the top depth of a geological body
    is easier to specify than its geometric center.
    """
    if dip_length <= 0.0:
        raise ValueError("dip_length must be greater than zero.")

    if not 0.0 <= dip_degrees <= 90.0:
        raise ValueError("dip_degrees must be between 0 and 90.")

    strike_radians = np.deg2rad(strike_degrees % 360.0)
    dip_radians = np.deg2rad(dip_degrees)
    dip_azimuth_radians = strike_radians + np.pi / 2.0

    down_dip_axis = np.array(
        [
            np.sin(dip_azimuth_radians) * np.cos(dip_radians),
            np.cos(dip_azimuth_radians) * np.cos(dip_radians),
            np.sin(dip_radians),
        ],
        dtype=np.float64,
    )

    center_offset = 0.5 * dip_length * down_dip_axis

    return generate_dipping_body(
        name=name,
        center_x=shallow_center_x + float(center_offset[0]),
        center_y=shallow_center_y + float(center_offset[1]),
        center_z=shallow_center_z + float(center_offset[2]),
        strike_length=strike_length,
        dip_length=dip_length,
        thickness=thickness,
        strike_degrees=strike_degrees,
        dip_degrees=dip_degrees,
        density_contrast=density_contrast,
        grid=grid,
    )

def generate_random_dipping_body(
    *,
    name: str,
    grid: GridSpec,
    rng: Random | None = None,
    minimum_strike_length: float = 10.0,
    maximum_strike_length: float = 30.0,
    minimum_dip_length: float = 8.0,
    maximum_dip_length: float = 24.0,
    minimum_thickness: float = 2.0,
    maximum_thickness: float = 6.0,
    minimum_strike_degrees: float = 0.0,
    maximum_strike_degrees: float = 180.0,
    minimum_dip_degrees: float = 10.0,
    maximum_dip_degrees: float = 80.0,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    edge_margin_x: float = 0.0,
    edge_margin_y: float = 0.0,
    edge_margin_z: float = 0.0,
    maximum_attempts: int = 1000,
) -> DippingBodySpec:
    """
    Generate one random inclined prism that fits completely inside the grid.

    The body's dimensions, strike, dip, and center are sampled randomly. Before
    selecting the center, the function calculates the axis-aligned extents of
    the rotated prism. This guarantees that every accepted body, including its
    corners, remains within the requested grid margins.

    Parameters
    ----------
    name
        Unique case name.
    grid
        Grid in which the complete dipping body must fit.
    rng
        Optional ``random.Random`` instance. Supply a seeded instance for
        reproducible sequences of bodies.
    minimum_strike_length, maximum_strike_length
        Inclusive range for body length along strike, in grid cells.
    minimum_dip_length, maximum_dip_length
        Inclusive range for body length down dip, in grid cells.
    minimum_thickness, maximum_thickness
        Inclusive range for thickness normal to the body plane, in grid cells.
    minimum_strike_degrees, maximum_strike_degrees
        Range from which strike is sampled, measured clockwise from positive y.
    minimum_dip_degrees, maximum_dip_degrees
        Range from which downward dip is sampled. Values must lie in [0, 90].
    density_contrast
        Constant density contrast assigned to the body.
    edge_margin_x, edge_margin_y, edge_margin_z
        Minimum clear distance, in cells, between the rotated body and each
        pair of grid boundaries.
    maximum_attempts
        Maximum number of random geometries to try before reporting that the
        requested size and angle ranges cannot fit within the grid.

    Returns
    -------
    DippingBodySpec
        Random, validated dipping-body specification.

    Raises
    ------
    ValueError
        If an input range is invalid.
    RuntimeError
        If no valid body can be generated within ``maximum_attempts``.
    """
    _validate_random_dipping_body_parameters(
        name=name,
        minimum_strike_length=minimum_strike_length,
        maximum_strike_length=maximum_strike_length,
        minimum_dip_length=minimum_dip_length,
        maximum_dip_length=maximum_dip_length,
        minimum_thickness=minimum_thickness,
        maximum_thickness=maximum_thickness,
        minimum_strike_degrees=minimum_strike_degrees,
        maximum_strike_degrees=maximum_strike_degrees,
        minimum_dip_degrees=minimum_dip_degrees,
        maximum_dip_degrees=maximum_dip_degrees,
        density_contrast=density_contrast,
        edge_margin_x=edge_margin_x,
        edge_margin_y=edge_margin_y,
        edge_margin_z=edge_margin_z,
        maximum_attempts=maximum_attempts,
    )

    if rng is None:
        rng = Random()

    for _ in range(maximum_attempts):
        strike_length = rng.uniform(
            minimum_strike_length,
            maximum_strike_length,
        )
        dip_length = rng.uniform(
            minimum_dip_length,
            maximum_dip_length,
        )
        thickness = rng.uniform(
            minimum_thickness,
            maximum_thickness,
        )
        strike_degrees = rng.uniform(
            minimum_strike_degrees,
            maximum_strike_degrees,
        )
        dip_degrees = rng.uniform(
            minimum_dip_degrees,
            maximum_dip_degrees,
        )

        half_extent_x, half_extent_y, half_extent_z = (
            _calculate_dipping_body_half_extents(
                strike_length=strike_length,
                dip_length=dip_length,
                thickness=thickness,
                strike_degrees=strike_degrees,
                dip_degrees=dip_degrees,
            )
        )

        center_x_limits = (
            edge_margin_x + half_extent_x,
            grid.nx - edge_margin_x - half_extent_x,
        )
        center_y_limits = (
            edge_margin_y + half_extent_y,
            grid.ny - edge_margin_y - half_extent_y,
        )
        center_z_limits = (
            edge_margin_z + half_extent_z,
            grid.nz - edge_margin_z - half_extent_z,
        )

        if not all(
            lower <= upper
            for lower, upper in (
                center_x_limits,
                center_y_limits,
                center_z_limits,
            )
        ):
            continue

        body = generate_dipping_body(
            name=name,
            center_x=rng.uniform(*center_x_limits),
            center_y=rng.uniform(*center_y_limits),
            center_z=rng.uniform(*center_z_limits),
            strike_length=strike_length,
            dip_length=dip_length,
            thickness=thickness,
            strike_degrees=strike_degrees,
            dip_degrees=dip_degrees,
            density_contrast=density_contrast,
            grid=grid,
        )

        return body

    raise RuntimeError(
        f"{name}: could not generate a dipping body that fits inside the "
        f"grid after {maximum_attempts} attempts. Reduce the body dimensions, "
        "narrow the angle ranges, or reduce the edge margins."
    )

def generate_random_dipping_cases(
    *,
    number_of_cases: int,
    grid: GridSpec,
    seed: int | None = None,
    name_prefix: str = "random_dipping_body",
    minimum_strike_length: float = 10.0,
    maximum_strike_length: float = 30.0,
    minimum_dip_length: float = 8.0,
    maximum_dip_length: float = 24.0,
    minimum_thickness: float = 2.0,
    maximum_thickness: float = 6.0,
    minimum_strike_degrees: float = 0.0,
    maximum_strike_degrees: float = 180.0,
    minimum_dip_degrees: float = 10.0,
    maximum_dip_degrees: float = 80.0,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    edge_margin_x: float = 0.0,
    edge_margin_y: float = 0.0,
    edge_margin_z: float = 0.0,
    maximum_attempts: int = 1000,
) -> list[DippingBodySpec]:
    """Generate multiple reproducible random dipping-body cases.

    A single seeded random-number generator is shared across all cases, giving
    a deterministic but different geometry for each generated body.
    """
    if number_of_cases <= 0:
        raise ValueError("number_of_cases must be greater than zero.")

    if not name_prefix.strip():
        raise ValueError("name_prefix must not be empty.")

    rng = Random(seed)

    return [
        generate_random_dipping_body(
            name=f"{name_prefix}_{case_index:03d}",
            grid=grid,
            rng=rng,
            minimum_strike_length=minimum_strike_length,
            maximum_strike_length=maximum_strike_length,
            minimum_dip_length=minimum_dip_length,
            maximum_dip_length=maximum_dip_length,
            minimum_thickness=minimum_thickness,
            maximum_thickness=maximum_thickness,
            minimum_strike_degrees=minimum_strike_degrees,
            maximum_strike_degrees=maximum_strike_degrees,
            minimum_dip_degrees=minimum_dip_degrees,
            maximum_dip_degrees=maximum_dip_degrees,
            density_contrast=density_contrast,
            edge_margin_x=edge_margin_x,
            edge_margin_y=edge_margin_y,
            edge_margin_z=edge_margin_z,
            maximum_attempts=maximum_attempts,
        )
        for case_index in range(1, number_of_cases + 1)
    ]

def _calculate_dipping_body_half_extents(
    *,
    strike_length: float,
    dip_length: float,
    thickness: float,
    strike_degrees: float,
    dip_degrees: float,
) -> tuple[float, float, float]:
    """Return the rotated prism's half-extents along global x, y, and z."""
    strike_radians = np.deg2rad(strike_degrees % 360.0)
    dip_radians = np.deg2rad(dip_degrees)
    dip_azimuth_radians = strike_radians + np.pi / 2.0

    strike_axis = np.array(
        [
            np.sin(strike_radians),
            np.cos(strike_radians),
            0.0,
        ],
        dtype=np.float64,
    )
    dip_axis = np.array(
        [
            np.sin(dip_azimuth_radians) * np.cos(dip_radians),
            np.cos(dip_azimuth_radians) * np.cos(dip_radians),
            np.sin(dip_radians),
        ],
        dtype=np.float64,
    )
    normal_axis = np.cross(strike_axis, dip_axis)
    normal_axis /= np.linalg.norm(normal_axis)

    half_extents = (
        0.5 * strike_length * np.abs(strike_axis)
        + 0.5 * dip_length * np.abs(dip_axis)
        + 0.5 * thickness * np.abs(normal_axis)
    )

    return (
        float(half_extents[0]),
        float(half_extents[1]),
        float(half_extents[2]),
    )

def _validate_random_dipping_body_parameters(
    *,
    name: str,
    minimum_strike_length: float,
    maximum_strike_length: float,
    minimum_dip_length: float,
    maximum_dip_length: float,
    minimum_thickness: float,
    maximum_thickness: float,
    minimum_strike_degrees: float,
    maximum_strike_degrees: float,
    minimum_dip_degrees: float,
    maximum_dip_degrees: float,
    density_contrast: float,
    edge_margin_x: float,
    edge_margin_y: float,
    edge_margin_z: float,
    maximum_attempts: int,
) -> None:
    """Validate parameter ranges used by the random dipping-body generator."""
    if not name.strip():
        raise ValueError("Body name must not be empty.")

    for range_name, minimum, maximum in (
        (
            "strike-length",
            minimum_strike_length,
            maximum_strike_length,
        ),
        (
            "dip-length",
            minimum_dip_length,
            maximum_dip_length,
        ),
        (
            "thickness",
            minimum_thickness,
            maximum_thickness,
        ),
    ):
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError(
                f"Invalid {range_name} range: minimum must be positive and "
                "must not exceed maximum."
            )

    if maximum_strike_degrees < minimum_strike_degrees:
        raise ValueError(
            "minimum_strike_degrees must not exceed maximum_strike_degrees."
        )

    if not (
        0.0 <= minimum_dip_degrees <= maximum_dip_degrees <= 90.0
    ):
        raise ValueError(
            "Dip limits must satisfy 0 <= minimum <= maximum <= 90."
        )

    if density_contrast == 0.0:
        raise ValueError("density_contrast must not be zero.")

    for margin_name, margin in (
        ("edge_margin_x", edge_margin_x),
        ("edge_margin_y", edge_margin_y),
        ("edge_margin_z", edge_margin_z),
    ):
        if margin < 0.0:
            raise ValueError(f"{margin_name} must be nonnegative.")

    if maximum_attempts <= 0:
        raise ValueError("maximum_attempts must be greater than zero.")

def generate_salt_dome(
    *,
    name: str,
    center_x: float,
    center_y: float,
    top_depth: float,
    bottom_depth: float,
    stem_radius_x: float,
    stem_radius_y: float,
    bulb_additional_radius_x: float = 0.0,
    bulb_additional_radius_y: float = 0.0,
    bulb_center_depth: float | None = None,
    bulb_vertical_scale: float | None = None,
    taper_fraction: float = 0.0,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    grid: GridSpec | None = None,
) -> SaltDomeSpec:
    """
    Create one parameterized vertical salt-dome specification.

    The dome has an elliptical horizontal cross-section whose radii vary
    continuously with depth. Its radius profile consists of a linearly
    tapered stem plus an optional Gaussian bulb contribution.

    ``stem_radius_x`` and ``stem_radius_y`` define the untapered baseline
    stem radii. ``bulb_additional_radius_x`` and
    ``bulb_additional_radius_y`` define extra radius added by the Gaussian
    bulb; they are not total bulb radii.

    When ``bulb_center_depth`` is omitted, it defaults to the midpoint of the
    dome's vertical interval.

    When ``bulb_vertical_scale`` is omitted, it defaults to one quarter of
    the dome's vertical height. This parameter must remain positive even when
    the bulb additions are zero because the general salt-dome profile always
    has a defined Gaussian term.
    """
    dome_height = bottom_depth - top_depth

    if bulb_center_depth is None:
        bulb_center_depth = (
            top_depth
            + 0.5 * dome_height
        )

    if bulb_vertical_scale is None:
        bulb_vertical_scale = (
            0.25 * dome_height
        )

    body = SaltDomeSpec(
        name=name,
        center_x=center_x,
        center_y=center_y,
        top_depth=top_depth,
        bottom_depth=bottom_depth,
        stem_radius_x=stem_radius_x,
        stem_radius_y=stem_radius_y,
        bulb_additional_radius_x=(
            bulb_additional_radius_x
        ),
        bulb_additional_radius_y=(
            bulb_additional_radius_y
        ),
        bulb_center_depth=bulb_center_depth,
        bulb_vertical_scale=bulb_vertical_scale,
        taper_fraction=taper_fraction,
        density_contrast=density_contrast,
    )

    if grid is not None:
        body.validate(grid)

    return body

def generate_cylindrical_salt_plug(
    *,
    name: str,
    center_x: float,
    center_y: float,
    top_depth: float,
    bottom_depth: float,
    radius_x: float,
    radius_y: float | None = None,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    grid: GridSpec | None = None,
) -> SaltDomeSpec:
    """
    Create a vertical cylindrical or elliptical salt plug.
    """
    if radius_y is None:
        radius_y = radius_x

    return generate_salt_dome(
        name=name,
        center_x=center_x,
        center_y=center_y,
        top_depth=top_depth,
        bottom_depth=bottom_depth,
        stem_radius_x=radius_x,
        stem_radius_y=radius_y,
        bulb_additional_radius_x=0.0,
        bulb_additional_radius_y=0.0,
        taper_fraction=0.0,
        density_contrast=density_contrast,
        grid=grid,
    )

def generate_tapered_salt_dome(
    *,
    name: str,
    center_x: float,
    center_y: float,
    top_depth: float,
    bottom_depth: float,
    top_radius_x: float,
    top_radius_y: float | None = None,
    taper_fraction: float = 0.35,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    grid: GridSpec | None = None,
) -> SaltDomeSpec:
    """
    Create a salt dome with a linearly narrowing stem and no bulb.
    """
    if top_radius_y is None:
        top_radius_y = top_radius_x

    return generate_salt_dome(
        name=name,
        center_x=center_x,
        center_y=center_y,
        top_depth=top_depth,
        bottom_depth=bottom_depth,
        stem_radius_x=top_radius_x,
        stem_radius_y=top_radius_y,
        bulb_additional_radius_x=0.0,
        bulb_additional_radius_y=0.0,
        taper_fraction=taper_fraction,
        density_contrast=density_contrast,
        grid=grid,
    )

def generate_bulbous_salt_dome(
    *,
    name: str,
    center_x: float,
    center_y: float,
    top_depth: float,
    bottom_depth: float,
    stem_radius_x: float,
    stem_radius_y: float | None = None,
    bulb_additional_radius_x: float = 3.0,
    bulb_additional_radius_y: float | None = None,
    bulb_center_depth: float | None = None,
    bulb_vertical_scale: float | None = None,
    taper_fraction: float = 0.15,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    grid: GridSpec | None = None,
) -> SaltDomeSpec:
    """
    Create a salt dome with a moderate Gaussian widening.
    """
    if stem_radius_y is None:
        stem_radius_y = stem_radius_x

    if bulb_additional_radius_y is None:
        bulb_additional_radius_y = (
            bulb_additional_radius_x
        )

    return generate_salt_dome(
        name=name,
        center_x=center_x,
        center_y=center_y,
        top_depth=top_depth,
        bottom_depth=bottom_depth,
        stem_radius_x=stem_radius_x,
        stem_radius_y=stem_radius_y,
        bulb_additional_radius_x=(
            bulb_additional_radius_x
        ),
        bulb_additional_radius_y=(
            bulb_additional_radius_y
        ),
        bulb_center_depth=bulb_center_depth,
        bulb_vertical_scale=bulb_vertical_scale,
        taper_fraction=taper_fraction,
        density_contrast=density_contrast,
        grid=grid,
    )

def generate_mushroom_salt_dome(
    *,
    name: str,
    center_x: float,
    center_y: float,
    top_depth: float,
    bottom_depth: float,
    stem_radius_x: float,
    stem_radius_y: float | None = None,
    cap_additional_radius_x: float = 6.0,
    cap_additional_radius_y: float | None = None,
    cap_center_depth: float | None = None,
    cap_vertical_scale: float = 2.0,
    taper_fraction: float = 0.1,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    grid: GridSpec | None = None,
) -> SaltDomeSpec:
    """
    Create a narrow-stem salt dome with a broad shallow cap.
    """
    if stem_radius_y is None:
        stem_radius_y = stem_radius_x

    if cap_additional_radius_y is None:
        cap_additional_radius_y = (
            cap_additional_radius_x
        )

    if cap_center_depth is None:
        cap_center_depth = (
            top_depth
            + 0.25 * (
                bottom_depth
                - top_depth
            )
        )

    return generate_salt_dome(
        name=name,
        center_x=center_x,
        center_y=center_y,
        top_depth=top_depth,
        bottom_depth=bottom_depth,
        stem_radius_x=stem_radius_x,
        stem_radius_y=stem_radius_y,
        bulb_additional_radius_x=(
            cap_additional_radius_x
        ),
        bulb_additional_radius_y=(
            cap_additional_radius_y
        ),
        bulb_center_depth=cap_center_depth,
        bulb_vertical_scale=cap_vertical_scale,
        taper_fraction=taper_fraction,
        density_contrast=density_contrast,
        grid=grid,
    )

def generate_basement_relief(
    *,
    name: str,
    base_depth: float,
    reference_x: float,
    reference_y: float,
    slope_x: float = 0.0,
    slope_y: float = 0.0,
    gaussian_amplitude: float = 0.0,
    gaussian_center_x: float = 0.0,
    gaussian_center_y: float = 0.0,
    gaussian_scale_x: float = 1.0,
    gaussian_scale_y: float = 1.0,
    sinusoid_amplitude: float = 0.0,
    sinusoid_wavelength: float = 1.0,
    sinusoid_azimuth_degrees: float = 0.0,
    sinusoid_phase_degrees: float = 0.0,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    grid: GridSpec | None = None,
) -> BasementReliefSpec:
    """
    Create one parameterized basement-relief specification.
    """
    body = BasementReliefSpec(
        name=name,
        base_depth=base_depth,
        reference_x=reference_x,
        reference_y=reference_y,
        slope_x=slope_x,
        slope_y=slope_y,
        gaussian_amplitude=gaussian_amplitude,
        gaussian_center_x=gaussian_center_x,
        gaussian_center_y=gaussian_center_y,
        gaussian_scale_x=gaussian_scale_x,
        gaussian_scale_y=gaussian_scale_y,
        sinusoid_amplitude=sinusoid_amplitude,
        sinusoid_wavelength=sinusoid_wavelength,
        sinusoid_azimuth_degrees=sinusoid_azimuth_degrees,
        sinusoid_phase_degrees=sinusoid_phase_degrees,
        density_contrast=density_contrast,
    )

    if grid is not None:
        body.validate(grid)

    return body

def generate_flat_basement(
    *,
    name: str,
    depth: float,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    grid: GridSpec,
) -> BasementReliefSpec:
    """Create a flat horizontal basement interface."""
    return generate_basement_relief(
        name=name,
        base_depth=depth,
        reference_x=grid.nx / 2.0,
        reference_y=grid.ny / 2.0,
        gaussian_center_x=grid.nx / 2.0,
        gaussian_center_y=grid.ny / 2.0,
        density_contrast=density_contrast,
        grid=grid,
    )

def generate_tilted_basement(
    *,
    name: str,
    base_depth: float,
    slope_x: float = 0.0,
    slope_y: float = 0.0,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    grid: GridSpec,
) -> BasementReliefSpec:
    """Create a planar tilted basement interface."""
    return generate_basement_relief(
        name=name,
        base_depth=base_depth,
        reference_x=grid.nx / 2.0,
        reference_y=grid.ny / 2.0,
        slope_x=slope_x,
        slope_y=slope_y,
        gaussian_center_x=grid.nx / 2.0,
        gaussian_center_y=grid.ny / 2.0,
        density_contrast=density_contrast,
        grid=grid,
    )

def generate_basement_uplift(
    *,
    name: str,
    base_depth: float,
    uplift_height: float,
    center_x: float,
    center_y: float,
    scale_x: float,
    scale_y: float,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    grid: GridSpec,
) -> BasementReliefSpec:
    """
    Create a smooth Gaussian basement uplift.

    ``uplift_height`` is supplied as a positive magnitude and converted to a
    negative interface-depth perturbation.
    """
    if uplift_height <= 0.0:
        raise ValueError(
            "uplift_height must be positive."
        )

    return generate_basement_relief(
        name=name,
        base_depth=base_depth,
        reference_x=grid.nx / 2.0,
        reference_y=grid.ny / 2.0,
        gaussian_amplitude=-uplift_height,
        gaussian_center_x=center_x,
        gaussian_center_y=center_y,
        gaussian_scale_x=scale_x,
        gaussian_scale_y=scale_y,
        density_contrast=density_contrast,
        grid=grid,
    )

def generate_basement_basin(
    *,
    name: str,
    base_depth: float,
    basin_depth: float,
    center_x: float,
    center_y: float,
    scale_x: float,
    scale_y: float,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    grid: GridSpec,
) -> BasementReliefSpec:
    """
    Create a smooth Gaussian basement depression.
    """
    if basin_depth <= 0.0:
        raise ValueError(
            "basin_depth must be positive."
        )

    return generate_basement_relief(
        name=name,
        base_depth=base_depth,
        reference_x=grid.nx / 2.0,
        reference_y=grid.ny / 2.0,
        gaussian_amplitude=basin_depth,
        gaussian_center_x=center_x,
        gaussian_center_y=center_y,
        gaussian_scale_x=scale_x,
        gaussian_scale_y=scale_y,
        density_contrast=density_contrast,
        grid=grid,
    )

def generate_sinusoidal_basement(
    *,
    name: str,
    base_depth: float,
    amplitude: float,
    wavelength: float,
    azimuth_degrees: float = 0.0,
    phase_degrees: float = 0.0,
    density_contrast: float = DEFAULT_DENSITY_CONTRAST,
    grid: GridSpec,
) -> BasementReliefSpec:
    """Create a smoothly undulating sinusoidal basement interface."""
    return generate_basement_relief(
        name=name,
        base_depth=base_depth,
        reference_x=grid.nx / 2.0,
        reference_y=grid.ny / 2.0,
        gaussian_center_x=grid.nx / 2.0,
        gaussian_center_y=grid.ny / 2.0,
        sinusoid_amplitude=amplitude,
        sinusoid_wavelength=wavelength,
        sinusoid_azimuth_degrees=azimuth_degrees,
        sinusoid_phase_degrees=phase_degrees,
        density_contrast=density_contrast,
        grid=grid,
    )


