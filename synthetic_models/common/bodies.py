from dataclasses import dataclass
from synthetic_models.common.grid import GridSpec
import numpy as np

@dataclass(frozen=True )
class RectangularBodySpec:
    """Definition of one rectangular constant-density body."""

    name: str

    z_start: int    
    z_end: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int

    density_contrast: float

    def validate(self, grid: GridSpec) -> None:
        """Check that the body is within the grid bounds."""
        if not (0 <= self.z_start < self.z_end <= grid.nz):
            raise ValueError(f"Body {self.name} z indices out of bounds: "
                             f"{self.z_start}-{self.z_end} not in [0, {grid.nz}]")
        if not (0 <= self.x_start < self.x_end <= grid.nx):
            raise ValueError(f"Body {self.name} x indices out of bounds: "
                             f"{self.x_start}-{self.x_end} not in [0, {grid.nx}]")
        if not (0 <= self.y_start < self.y_end <= grid.ny):
            raise ValueError(f"Body {self.name} y indices out of bounds: "
                             f"{self.y_start}-{self.y_end} not in [0, {grid.ny}]")
        if self.density_contrast == 0.0:
            raise ValueError(f"Body {self.name} has zero density contrast.")  

@dataclass(frozen=True)
class MultiBodyCaseSpec:
    """One synthetic case containing multiple rectangular bodies."""

    name: str
    bodies: tuple[RectangularBodySpec, ...]
    allow_overlap: bool = False

    def validate(
        self,
        grid: GridSpec,
    ) -> None:
        """Validate the case name, contained bodies, and overlap rules."""

        if not self.name.strip():
            raise ValueError(
                "Multi-body case name must not be empty."
            )

        if not self.bodies:
            raise ValueError(
                f"{self.name}: at least one body must be provided."
            )

        body_names = [
            body.name
            for body in self.bodies
        ]

        if len(body_names) != len(set(body_names)):
            raise ValueError(
                f"{self.name}: every body must have a unique name."
            )

        for body in self.bodies:
            body.validate(grid)

        if not self.allow_overlap:
            _validate_no_body_overlap(
                case_name=self.name,
                bodies=self.bodies,
                grid=grid,
            )

    @property
    def body_count(self) -> int:
        """Return the number of bodies in the case."""

        return len(self.bodies)

    @property
    def density_contrast(self) -> float:
        """
        Return the shared density contrast.

        Density-comparison threshold metrics currently assume that all
        bodies use the same density contrast.
        """

        contrasts = {
            float(body.density_contrast)
            for body in self.bodies
        }

        if len(contrasts) != 1:
            raise ValueError(
                f"{self.name}: all bodies must use the same density "
                "contrast for the current comparison metrics."
            )

        return contrasts.pop()

    @property
    def expected_nonzero_cells(self) -> int:
        """
        Return the expected occupied-cell count.

        This property is exact when body overlap is disabled.
        """

        if self.allow_overlap:
            raise ValueError(
                f"{self.name}: expected_nonzero_cells is ambiguous "
                "when overlap is allowed."
            )

        return sum(
            (
                body.x_end - body.x_start
            )
            * (
                body.y_end - body.y_start
            )
            * (
                body.z_end - body.z_start
            )
            for body in self.bodies
        )

@dataclass(frozen=True)
class DippingBodySpec:
    """
    Definition of one constant-density oriented rectangular prism.

    The body is defined in model-cell coordinates rather than physical
    distance units. Coordinates refer to cell centers, so ``center_x=10.5``
    places the body center at the center of x-index 10.

    The global array order remains ``(z, y, x)``. Strike is measured clockwise
    from the positive y direction, and dip is measured downward from horizontal.
    The body dips toward ``strike_degrees + 90`` degrees.

    ``strike_length`` is measured along strike, ``dip_length`` is measured
    down dip, and ``thickness`` is measured normal to the body plane.
    """

    name: str

    center_x: float
    center_y: float
    center_z: float

    strike_length: float
    dip_length: float
    thickness: float

    strike_degrees: float
    dip_degrees: float

    density_contrast: float

    def validate(self, grid: GridSpec) -> None:
        """Validate the body parameters and ensure the full prism fits."""
        if not self.name.strip():
            raise ValueError("Dipping-body name must not be empty.")

        for dimension_name, dimension in {
            "strike_length": self.strike_length,
            "dip_length": self.dip_length,
            "thickness": self.thickness,
        }.items():
            if dimension <= 0.0:
                raise ValueError(
                    f"{self.name}: {dimension_name} must be greater than zero."
                )

        if not 0.0 <= self.dip_degrees <= 90.0:
            raise ValueError(
                f"{self.name}: dip_degrees must be between 0 and 90."
            )

        if self.density_contrast == 0.0:
            raise ValueError(
                f"{self.name}: density_contrast must not be zero."
            )

        corners = _dipping_body_corners(self)

        x_coordinates = corners[:, 0]
        y_coordinates = corners[:, 1]
        z_coordinates = corners[:, 2]

        if np.any(x_coordinates < 0.0) or np.any(x_coordinates > grid.nx):
            raise ValueError(
                f"{self.name}: oriented body extends outside the x grid bounds."
            )

        if np.any(y_coordinates < 0.0) or np.any(y_coordinates > grid.ny):
            raise ValueError(
                f"{self.name}: oriented body extends outside the y grid bounds."
            )

        if np.any(z_coordinates < 0.0) or np.any(z_coordinates > grid.nz):
            raise ValueError(
                f"{self.name}: oriented body extends outside the z grid bounds."
            )

@dataclass(frozen=True)
class SaltDomeSpec:
    name:str
    center_x: float
    center_y: float

    top_depth: float
    bottom_depth: float

    stem_radius_x: float
    stem_radius_y: float

    bulb_additional_radius_x: float
    bulb_additional_radius_y: float

    bulb_center_depth: float
    bulb_vertical_scale: float

    taper_fraction: float
    density_contrast: float

    @property
    def height(self) -> float:
        """Return the dome's vertical height in grid-cell units."""
        return self.bottom_depth - self.top_depth

    @property
    def maximum_possible_radius_x(self) -> float:
        """
        Return a conservative upper bound on the dome's x radius.

        The Gaussian bulb weight cannot exceed one, and a nonnegative taper
        fraction cannot enlarge the stem beyond its radius at the dome top.
        """
        return (
            self.stem_radius_x
            + self.bulb_additional_radius_x
        )

    @property
    def maximum_possible_radius_y(self) -> float:
        """
        Return a conservative upper bound on the dome's y radius.

        The Gaussian bulb weight cannot exceed one, and a nonnegative taper
        fraction cannot enlarge the stem beyond its radius at the dome top.
        """
        return (
            self.stem_radius_y
            + self.bulb_additional_radius_y
        )

    def validate(self, grid: GridSpec) -> None:
        """
        Validate salt-dome geometry and ensure it can occupy the grid.

        Validation covers parameter values, vertical and horizontal bounds,
        and discrete voxel occupancy. A salt dome may use either a positive
        or negative density contrast.
        """
        _validate_salt_dome_finite_values(self)
        _validate_salt_dome_dimensions(self)
        _validate_salt_dome_vertical_bounds(
            body=self,
            grid=grid,
        )
        _validate_salt_dome_horizontal_bounds(
            body=self,
            grid=grid,
        )

        if not _salt_dome_contains_any_cell_center(
            body=self,
            grid=grid,
        ):
            raise ValueError(
                f"{self.name}: salt dome occupies no grid cells. "
                "Increase its dimensions or adjust its position."
            )

@dataclass(frozen=True)
class BasementReliefSpec:
    """
    Definition of a basement body beneath a variable-depth interface.

    The model contains zero density contrast above the interface and a
    constant basement density contrast below it.
    """

    name: str

    base_depth: float

    reference_x: float
    reference_y: float

    slope_x: float = 0.0
    slope_y: float = 0.0

    gaussian_amplitude: float = 0.0
    gaussian_center_x: float = 0.0
    gaussian_center_y: float = 0.0
    gaussian_scale_x: float = 1.0
    gaussian_scale_y: float = 1.0

    sinusoid_amplitude: float = 0.0
    sinusoid_wavelength: float = 1.0
    sinusoid_azimuth_degrees: float = 0.0
    sinusoid_phase_degrees: float = 0.0

    density_contrast: float = 0.5

    def validate(
        self,
        grid: GridSpec,
    ) -> None:
        """
        Validate the basement-relief specification against the model grid.
        """
        if not self.name.strip():
            raise ValueError(
                "Basement-relief name must not be empty."
            )

        scalar_values = {
            "base_depth": self.base_depth,
            "reference_x": self.reference_x,
            "reference_y": self.reference_y,
            "slope_x": self.slope_x,
            "slope_y": self.slope_y,
            "gaussian_amplitude": self.gaussian_amplitude,
            "gaussian_center_x": self.gaussian_center_x,
            "gaussian_center_y": self.gaussian_center_y,
            "gaussian_scale_x": self.gaussian_scale_x,
            "gaussian_scale_y": self.gaussian_scale_y,
            "sinusoid_amplitude": self.sinusoid_amplitude,
            "sinusoid_wavelength": self.sinusoid_wavelength,
            "sinusoid_azimuth_degrees": (
                self.sinusoid_azimuth_degrees
            ),
            "sinusoid_phase_degrees": (
                self.sinusoid_phase_degrees
            ),
            "density_contrast": self.density_contrast,
        }

        for parameter_name, value in scalar_values.items():
            if not np.isfinite(value):
                raise ValueError(
                    f"{self.name}: {parameter_name} must be finite."
                )

        if not 0.0 <= self.reference_x <= grid.nx:
            raise ValueError(
                f"{self.name}: reference_x must lie within "
                f"[0, {grid.nx}]."
            )

        if not 0.0 <= self.reference_y <= grid.ny:
            raise ValueError(
                f"{self.name}: reference_y must lie within "
                f"[0, {grid.ny}]."
            )

        if not 0.0 <= self.gaussian_center_x <= grid.nx:
            raise ValueError(
                f"{self.name}: gaussian_center_x must lie within "
                f"[0, {grid.nx}]."
            )

        if not 0.0 <= self.gaussian_center_y <= grid.ny:
            raise ValueError(
                f"{self.name}: gaussian_center_y must lie within "
                f"[0, {grid.ny}]."
            )

        if self.gaussian_scale_x <= 0.0:
            raise ValueError(
                f"{self.name}: gaussian_scale_x must be positive."
            )

        if self.gaussian_scale_y <= 0.0:
            raise ValueError(
                f"{self.name}: gaussian_scale_y must be positive."
            )

        if self.sinusoid_wavelength <= 0.0:
            raise ValueError(
                f"{self.name}: sinusoid_wavelength must be positive."
            )

        if self.density_contrast == 0.0:
            raise ValueError(
                f"{self.name}: density_contrast must not be zero."
            )

        interface_depths = calculate_basement_interface(
            grid=grid,
            body=self,
        )

        minimum_depth = float(np.min(interface_depths))
        maximum_depth = float(np.max(interface_depths))

        if minimum_depth <= 0.0:
            raise ValueError(
                f"{self.name}: basement interface reaches or crosses "
                f"the top of the grid. Minimum depth={minimum_depth:.4f}."
            )

        if maximum_depth >= grid.nz:
            raise ValueError(
                f"{self.name}: basement interface reaches or crosses "
                f"the bottom of the grid. Maximum depth="
                f"{maximum_depth:.4f}, grid.nz={grid.nz}."
            )

CaseSpec = (
    RectangularBodySpec
    | DippingBodySpec
    | MultiBodyCaseSpec
    | SaltDomeSpec
    | BasementReliefSpec
)

def build_rectangular_body(grid: GridSpec, body: RectangularBodySpec) -> np.ndarray:
    """Create a 3D numpy array representing the rectangular body."""
    body.validate(grid)
    model = np.zeros((grid.nz, grid.ny, grid.nx), dtype=np.float32)


    model[
        body.z_start:body.z_end,
        body.y_start:body.y_end,
        body.x_start:body.x_end,
    ] = body.density_contrast
        
    return model

def _validate_no_body_overlap(
    *,
    case_name: str,
    bodies: tuple[RectangularBodySpec, ...],
    grid: GridSpec,
) -> None:
    """Raise an error when two bodies occupy the same model cell."""

    occupancy = np.zeros(
        (
            grid.nz,
            grid.ny,
            grid.nx,
        ),
        dtype=np.uint8,
    )

    for body in bodies:
        body_slice = np.s_[
            body.z_start:body.z_end,
            body.y_start:body.y_end,
            body.x_start:body.x_end,
        ]

        if np.any(occupancy[body_slice] != 0):
            raise ValueError(
                f"{case_name}: body '{body.name}' overlaps another body."
            )

        occupancy[body_slice] = 1

def build_multi_body_model(
    grid: GridSpec,
    case: MultiBodyCaseSpec,
) -> np.ndarray:
    """
    Build one density model containing multiple rectangular bodies.

    Array order is ``(z, y, x)``.
    """

    case.validate(grid)

    model = np.zeros(
        (
            grid.nz,
            grid.ny,
            grid.nx,
        ),
        dtype=np.float32,
        order="C",
    )

    for body in case.bodies:
        body_slice = np.s_[
            body.z_start:body.z_end,
            body.y_start:body.y_end,
            body.x_start:body.x_end,
        ]

        if case.allow_overlap:
            model[body_slice] += np.float32(
                body.density_contrast
            )
        else:
            model[body_slice] = np.float32(
                body.density_contrast
            )

    return np.ascontiguousarray(
        model,
        dtype=np.float32,
    )

def build_density_model(
    grid: GridSpec,
    case: CaseSpec,
) -> np.ndarray:
    """Build a supported synthetic density model."""

    if isinstance(case, RectangularBodySpec):
        return build_rectangular_body(
            grid=grid,
            body=case,
        )

    if isinstance(case, DippingBodySpec):
        return build_dipping_body(
            grid=grid,
            body=case,
        )

    if isinstance(case, MultiBodyCaseSpec):
        return build_multi_body_model(
            grid=grid,
            case=case,
        )
    
    if isinstance(case, SaltDomeSpec):
        return build_salt_dome(
            grid=grid,
            body=case,
        )
    
    if isinstance(case, BasementReliefSpec):
        return build_basement_relief(
            grid=grid,
            body=case,
        )

    raise TypeError(
        "Unsupported case specification: "
        f"{type(case).__name__}."
    )

def build_dipping_body(
    grid: GridSpec,
    body: DippingBodySpec,
) -> np.ndarray:
    """
    Rasterize one oriented rectangular prism onto the model grid.

    A cell is occupied when its center lies inside the prism. The returned
    array is contiguous, uses ``float32``, and follows ``(z, y, x)`` order.
    """
    body.validate(grid)

    z_indices, y_indices, x_indices = np.indices(
        (grid.nz, grid.ny, grid.nx),
        dtype=np.float64,
    )

    x_centers = x_indices + 0.5
    y_centers = y_indices + 0.5
    z_centers = z_indices + 0.5

    strike_axis, dip_axis, normal_axis = _dipping_body_basis(body)

    displacement_x = x_centers - body.center_x
    displacement_y = y_centers - body.center_y
    displacement_z = z_centers - body.center_z

    local_strike = (
        displacement_x * strike_axis[0]
        + displacement_y * strike_axis[1]
        + displacement_z * strike_axis[2]
    )
    local_dip = (
        displacement_x * dip_axis[0]
        + displacement_y * dip_axis[1]
        + displacement_z * dip_axis[2]
    )
    local_normal = (
        displacement_x * normal_axis[0]
        + displacement_y * normal_axis[1]
        + displacement_z * normal_axis[2]
    )

    mask = (
        (np.abs(local_strike) <= body.strike_length / 2.0)
        & (np.abs(local_dip) <= body.dip_length / 2.0)
        & (np.abs(local_normal) <= body.thickness / 2.0)
    )

    if not np.any(mask):
        raise ValueError(
            f"{body.name}: body occupies no grid cells. Increase its dimensions "
            "or adjust its position."
        )

    model = np.zeros(
        (grid.nz, grid.ny, grid.nx),
        dtype=np.float32,
        order="C",
    )
    model[mask] = np.float32(body.density_contrast)

    return np.ascontiguousarray(model, dtype=np.float32)

def _dipping_body_basis(
    body: DippingBodySpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal strike, down-dip, and normal vectors."""
    strike_radians = np.deg2rad(body.strike_degrees % 360.0)
    dip_radians = np.deg2rad(body.dip_degrees)
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

    return strike_axis, dip_axis, normal_axis

def _dipping_body_corners(
    body: DippingBodySpec,
) -> np.ndarray:
    """Return the eight prism corners in continuous grid coordinates."""
    strike_axis, dip_axis, normal_axis = _dipping_body_basis(body)
    center = np.array(
        [body.center_x, body.center_y, body.center_z],
        dtype=np.float64,
    )

    half_strike = body.strike_length / 2.0
    half_dip = body.dip_length / 2.0
    half_thickness = body.thickness / 2.0

    return np.array(
        [
            center
            + strike_sign * half_strike * strike_axis
            + dip_sign * half_dip * dip_axis
            + normal_sign * half_thickness * normal_axis
            for strike_sign in (-1.0, 1.0)
            for dip_sign in (-1.0, 1.0)
            for normal_sign in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )

def _validate_salt_dome_finite_values(
    body: SaltDomeSpec,
) -> None:
    """Validate that every numeric salt-dome parameter is finite."""
    numeric_parameters = {
        "center_x": body.center_x,
        "center_y": body.center_y,
        "top_depth": body.top_depth,
        "bottom_depth": body.bottom_depth,
        "stem_radius_x": body.stem_radius_x,
        "stem_radius_y": body.stem_radius_y,
        "bulb_additional_radius_x": (
            body.bulb_additional_radius_x
        ),
        "bulb_additional_radius_y": (
            body.bulb_additional_radius_y
        ),
        "bulb_center_depth": body.bulb_center_depth,
        "bulb_vertical_scale": body.bulb_vertical_scale,
        "taper_fraction": body.taper_fraction,
        "density_contrast": body.density_contrast,
    }

    for parameter_name, parameter_value in numeric_parameters.items():
        if not np.isfinite(parameter_value):
            raise ValueError(
                f"{body.name or 'Salt dome'}: {parameter_name} "
                "must be finite."
            )

def _validate_salt_dome_dimensions(
    body: SaltDomeSpec,
) -> None:
    """Validate intrinsic salt-dome parameters."""
    if not body.name.strip():
        raise ValueError(
            "Salt-dome name must not be empty."
        )

    if body.top_depth >= body.bottom_depth:
        raise ValueError(
            f"{body.name}: top_depth must be less than bottom_depth."
        )

    for radius_name, radius in {
        "stem_radius_x": body.stem_radius_x,
        "stem_radius_y": body.stem_radius_y,
    }.items():
        if radius <= 0.0:
            raise ValueError(
                f"{body.name}: {radius_name} must be greater than zero."
            )

    for radius_name, radius in {
        "bulb_additional_radius_x": (
            body.bulb_additional_radius_x
        ),
        "bulb_additional_radius_y": (
            body.bulb_additional_radius_y
        ),
    }.items():
        if radius < 0.0:
            raise ValueError(
                f"{body.name}: {radius_name} must be nonnegative."
            )

    if body.bulb_vertical_scale <= 0.0:
        raise ValueError(
            f"{body.name}: bulb_vertical_scale must be greater than zero."
        )

    if not (
        body.top_depth
        <= body.bulb_center_depth
        <= body.bottom_depth
    ):
        raise ValueError(
            f"{body.name}: bulb_center_depth must lie between "
            "top_depth and bottom_depth."
        )

    if not 0.0 <= body.taper_fraction < 1.0:
        raise ValueError(
            f"{body.name}: taper_fraction must satisfy "
            "0 <= taper_fraction < 1."
        )

    if body.density_contrast == 0.0:
        raise ValueError(
            f"{body.name}: density_contrast must not be zero."
        )

def _validate_salt_dome_vertical_bounds(
    *,
    body: SaltDomeSpec,
    grid: GridSpec,
) -> None:
    """Ensure the dome's vertical interval lies inside the grid."""
    if body.top_depth < 0.0:
        raise ValueError(
            f"{body.name}: top_depth must be at least 0."
        )

    if body.bottom_depth > grid.nz:
        raise ValueError(
            f"{body.name}: bottom_depth must not exceed "
            f"the grid depth of {grid.nz} cells."
        )

def _validate_salt_dome_horizontal_bounds(
    *,
    body: SaltDomeSpec,
    grid: GridSpec,
) -> None:
    """
    Ensure a conservative horizontal dome envelope fits inside the grid.

    The bound uses the untapered stem radius plus the maximum possible
    Gaussian bulb addition. This may be slightly conservative when the bulb
    center occurs where the stem is already tapered.
    """
    minimum_x = (
        body.center_x
        - body.maximum_possible_radius_x
    )
    maximum_x = (
        body.center_x
        + body.maximum_possible_radius_x
    )

    minimum_y = (
        body.center_y
        - body.maximum_possible_radius_y
    )
    maximum_y = (
        body.center_y
        + body.maximum_possible_radius_y
    )

    if minimum_x < 0.0 or maximum_x > grid.nx:
        raise ValueError(
            f"{body.name}: salt dome extends outside the x grid bounds. "
            f"Conservative x extent is [{minimum_x:.3f}, "
            f"{maximum_x:.3f}], while the grid extent is "
            f"[0, {grid.nx}]."
        )

    if minimum_y < 0.0 or maximum_y > grid.ny:
        raise ValueError(
            f"{body.name}: salt dome extends outside the y grid bounds. "
            f"Conservative y extent is [{minimum_y:.3f}, "
            f"{maximum_y:.3f}], while the grid extent is "
            f"[0, {grid.ny}]."
        )

def _salt_dome_radii_at_depth(
    *,
    body: SaltDomeSpec,
    depth: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate salt-dome x and y radii at one or more depths.
    """
    depth_array = np.asarray(
        depth,
        dtype=np.float64,
    )

    normalized_depth = (
        depth_array - body.top_depth
    ) / body.height

    taper_multiplier = (
        1.0
        - body.taper_fraction * normalized_depth
    )

    bulb_weight = np.exp(
        -0.5
        * (
            (
                depth_array
                - body.bulb_center_depth
            )
            / body.bulb_vertical_scale
        )
        ** 2
    )

    radius_x = (
        body.stem_radius_x * taper_multiplier
        + body.bulb_additional_radius_x * bulb_weight
    )

    radius_y = (
        body.stem_radius_y * taper_multiplier
        + body.bulb_additional_radius_y * bulb_weight
    )

    return radius_x, radius_y

def _salt_dome_contains_any_cell_center(
    *,
    body: SaltDomeSpec,
    grid: GridSpec,
) -> bool:
    """
    Return whether at least one grid-cell center lies inside the dome.

    This check uses the same continuous cell-center convention that the model
    builder will use in Step 2.
    """
    z_centers = (
        np.arange(
            grid.nz,
            dtype=np.float64,
        )
        + 0.5
    )

    active_depth_mask = (
        (z_centers >= body.top_depth)
        & (z_centers <= body.bottom_depth)
    )

    active_depths = z_centers[active_depth_mask]

    if active_depths.size == 0:
        return False

    radius_x, radius_y = _salt_dome_radii_at_depth(
        body=body,
        depth=active_depths,
    )

    x_centers = (
        np.arange(
            grid.nx,
            dtype=np.float64,
        )
        + 0.5
    )
    y_centers = (
        np.arange(
            grid.ny,
            dtype=np.float64,
        )
        + 0.5
    )

    normalized_x_squared = (
        (
            x_centers
            - body.center_x
        )[None, :]
        / radius_x[:, None]
    ) ** 2

    normalized_y_squared = (
        (
            y_centers
            - body.center_y
        )[None, :]
        / radius_y[:, None]
    ) ** 2

    minimum_ellipse_value_by_depth = (
        normalized_x_squared.min(axis=1)
        + normalized_y_squared.min(axis=1)
    )

    return bool(
        np.any(
            minimum_ellipse_value_by_depth <= 1.0
        )
    )

def build_salt_dome(
    grid: GridSpec,
    body: SaltDomeSpec,
) -> np.ndarray:
    """
    Rasterize one vertical salt-dome-like body onto the model grid.

    The salt dome has an elliptical horizontal cross-section whose x and y
    radii vary continuously with depth. A cell is occupied when its center:

    1. Lies between ``top_depth`` and ``bottom_depth``.
    2. Lies inside the depth-dependent horizontal ellipse.

    Cell centers use continuous grid-cell coordinates:

    ``x = x_index + 0.5``
    ``y = y_index + 0.5``
    ``z = z_index + 0.5``

    The returned density model follows ``(z, y, x)`` array order.

    Parameters
    ----------
    grid
        Grid on which the salt dome is rasterized.
    body
        Salt-dome specification.

    Returns
    -------
    numpy.ndarray
        C-contiguous ``float32`` density model with shape
        ``(grid.nz, grid.ny, grid.nx)``.

    Raises
    ------
    ValueError
        If the specification is invalid or the rasterized body occupies no
        grid cells.
    """
    body.validate(grid)

    x_centers = (
        np.arange(
            grid.nx,
            dtype=np.float64,
        )
        + 0.5
    )

    y_centers = (
        np.arange(
            grid.ny,
            dtype=np.float64,
        )
        + 0.5
    )

    z_centers = (
        np.arange(
            grid.nz,
            dtype=np.float64,
        )
        + 0.5
    )

    radius_x, radius_y = _salt_dome_radii_at_depth(
        body=body,
        depth=z_centers,
    )

    normalized_x_squared = (
        (
            x_centers[None, None, :]
            - body.center_x
        )
        / radius_x[:, None, None]
    ) ** 2

    normalized_y_squared = (
        (
            y_centers[None, :, None]
            - body.center_y
        )
        / radius_y[:, None, None]
    ) ** 2

    horizontal_mask = (
        normalized_x_squared
        + normalized_y_squared
        <= 1.0
    )

    vertical_mask = (
        (z_centers >= body.top_depth)
        & (z_centers <= body.bottom_depth)
    )

    dome_mask = (
        horizontal_mask
        & vertical_mask[:, None, None]
    )

    if not np.any(dome_mask):
        raise ValueError(
            f"{body.name}: salt dome occupies no grid cells. "
            "Increase its dimensions or adjust its position."
        )

    model = np.zeros(
        (
            grid.nz,
            grid.ny,
            grid.nx,
        ),
        dtype=np.float32,
        order="C",
    )

    model[dome_mask] = np.float32(
        body.density_contrast
    )

    return np.ascontiguousarray(
        model,
        dtype=np.float32,
    )

def calculate_basement_interface(
    *,
    grid: GridSpec,
    body: BasementReliefSpec,
) -> np.ndarray:
    """
    Calculate the basement-interface depth over the horizontal grid.
    """
    y_indices, x_indices = np.indices(
        (
            grid.ny,
            grid.nx,
        ),
        dtype=np.float64,
    )

    x_centers = x_indices + 0.5
    y_centers = y_indices + 0.5

    planar_depth = (
        body.base_depth
        + body.slope_x
        * (
            x_centers
            - body.reference_x
        )
        + body.slope_y
        * (
            y_centers
            - body.reference_y
        )
    )

    gaussian_exponent = (
        (
            (
                x_centers
                - body.gaussian_center_x
            )
            / body.gaussian_scale_x
        )
        ** 2
        + (
            (
                y_centers
                - body.gaussian_center_y
            )
            / body.gaussian_scale_y
        )
        ** 2
    )

    gaussian_relief = (
        body.gaussian_amplitude
        * np.exp(
            -0.5
            * gaussian_exponent
        )
    )

    azimuth_radians = np.deg2rad(
        body.sinusoid_azimuth_degrees
    )
    phase_radians = np.deg2rad(
        body.sinusoid_phase_degrees
    )

    projected_distance = (
        (
            x_centers
            - body.reference_x
        )
        * np.sin(azimuth_radians)
        + (
            y_centers
            - body.reference_y
        )
        * np.cos(azimuth_radians)
    )

    sinusoidal_relief = (
        body.sinusoid_amplitude
        * np.cos(
            2.0
            * np.pi
            * projected_distance
            / body.sinusoid_wavelength
            + phase_radians
        )
    )

    interface_depths = (
        planar_depth
        + gaussian_relief
        + sinusoidal_relief
    )

    return np.ascontiguousarray(
        interface_depths,
        dtype=np.float64,
    )

def build_basement_relief(
    *,
    grid: GridSpec,
    body: BasementReliefSpec,
) -> np.ndarray:
    """
    Build a constant-density basement below a variable-depth interface.
    """
    body.validate(grid)

    interface_depths = calculate_basement_interface(
        grid=grid,
        body=body,
    )

    z_centers = (
        np.arange(
            grid.nz,
            dtype=np.float64,
        )
        + 0.5
    )

    basement_mask = (
        z_centers[:, np.newaxis, np.newaxis]
        >= interface_depths[np.newaxis, :, :]
    )

    if not np.any(basement_mask):
        raise ValueError(
            f"{body.name}: basement occupies no grid cells."
        )

    if np.all(basement_mask):
        raise ValueError(
            f"{body.name}: basement occupies the entire grid."
        )

    model = np.zeros(
        (
            grid.nz,
            grid.ny,
            grid.nx,
        ),
        dtype=np.float32,
        order="C",
    )

    model[basement_mask] = np.float32(
        body.density_contrast
    )

    return np.ascontiguousarray(
        model,
        dtype=np.float32,
    )


