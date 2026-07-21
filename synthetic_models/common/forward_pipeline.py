from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.metrics import gravity_response_metrics
from evaluation.plotting import (
    plot_gravity_anomaly,
    plot_gravity_center_profiles,
    plot_gravity_comparison,
)
from forward_modeling.forward_model import GravityForwardModel
from synthetic_models.common.bodies import (
    CaseSpec,
    DippingBodySpec,
    MultiBodyCaseSpec,
    RectangularBodySpec,
    SaltDomeSpec,
    BasementReliefSpec,
    calculate_basement_interface,
    build_density_model,
)
from synthetic_models.common.grid import GridSpec
from synthetic_models.common.model_io import (
    save_gravity_response,
    save_true_model,
)
from synthetic_models.common.output import (
    print_case_summary,
    print_saved_paths,
)
from synthetic_models.common.paths import ExperimentPaths
from synthetic_models.common.validation import validate_density_model


MetricValue = str | float | int
CaseMetrics = dict[str, MetricValue]


def _case_geometry_metrics(
    case: CaseSpec,
    grid: GridSpec,
) -> CaseMetrics:
    """
    Return geometry metadata for a supported synthetic case.

    Parameters
    ----------
    case
        Synthetic case specification.
    grid
        Grid specification used to convert cell coordinates into physical
        coordinates where appropriate.

    Returns
    -------
    dict
        Geometry metadata suitable for inclusion in the experiment metrics
        CSV.

    Raises
    ------
    TypeError
        If the case specification type is unsupported.
    """
    if isinstance(case, RectangularBodySpec):
        return _rectangular_geometry_metrics(
            case=case,
            grid=grid,
        )

    if isinstance(case, DippingBodySpec):
        return _dipping_geometry_metrics(
            case=case,
            grid=grid,
        )

    if isinstance(case, SaltDomeSpec):
        return _salt_dome_geometry_metrics(
            case=case,
            grid=grid,
        )

    if isinstance(case, MultiBodyCaseSpec):
        return _multi_body_geometry_metrics(
            case=case,
        )
    
    if isinstance(case, BasementReliefSpec):
        return _basement_relief_geometry_metrics(
            case=case,
            grid=grid,
        )

    raise TypeError(
        f"Unsupported case specification: {type(case).__name__}."
    )


def _rectangular_geometry_metrics(
    *,
    case: RectangularBodySpec,
    grid: GridSpec,
) -> CaseMetrics:
    """Return geometry metadata for one axis-aligned rectangular body."""
    center_x_cells = 0.5 * (
        case.x_start
        + case.x_end
    )

    center_y_cells = 0.5 * (
        case.y_start
        + case.y_end
    )

    center_z_cells = 0.5 * (
        case.z_start
        + case.z_end
    )

    return {
        "case_type": "single_rectangular_body",
        "body_count": 1,
        "x_start": case.x_start,
        "x_end": case.x_end,
        "y_start": case.y_start,
        "y_end": case.y_end,
        "z_start": case.z_start,
        "z_end": case.z_end,
        "x_width_cells": (
            case.x_end
            - case.x_start
        ),
        "y_width_cells": (
            case.y_end
            - case.y_start
        ),
        "z_thickness_cells": (
            case.z_end
            - case.z_start
        ),
        "body_center_x": (
            grid.x_min
            + center_x_cells * grid.dx
        ),
        "body_center_y": (
            grid.y_min
            + center_y_cells * grid.dy
        ),
        "body_center_depth": (
            grid.z_min
            + center_z_cells * grid.dz
        ),
        "strike_length_cells": "",
        "dip_length_cells": "",
        "body_thickness_cells": "",
        "strike_degrees": "",
        "dip_degrees": "",
        "density_contrast": case.density_contrast,
    }


def _dipping_geometry_metrics(
    *,
    case: DippingBodySpec,
    grid: GridSpec,
) -> CaseMetrics:
    """
    Return geometry metadata for one dipping or elongated body.

    The dipping-body center and dimensions are stored in grid-cell units.
    The physical center is calculated using the corresponding grid spacing.

    Parameters
    ----------
    case
        Dipping-body specification.
    grid
        Model-grid specification.

    Returns
    -------
    dict
        Dipping-body geometry metadata.
    """
    return {
        "case_type": "dipping_body",
        "body_count": 1,
        "x_start": "",
        "x_end": "",
        "y_start": "",
        "y_end": "",
        "z_start": "",
        "z_end": "",
        "x_width_cells": "",
        "y_width_cells": "",
        "z_thickness_cells": "",
        "body_center_x": (
            grid.x_min
            + case.center_x * grid.dx
        ),
        "body_center_y": (
            grid.y_min
            + case.center_y * grid.dy
        ),
        "body_center_depth": (
            grid.z_min
            + case.center_z * grid.dz
        ),
        "strike_length_cells": case.strike_length,
        "dip_length_cells": case.dip_length,
        "body_thickness_cells": case.thickness,
        "strike_degrees": case.strike_degrees,
        "dip_degrees": case.dip_degrees,
        "density_contrast": case.density_contrast,
    }


def _salt_dome_geometry_metrics(
    *,
    case: SaltDomeSpec,
    grid: GridSpec,
) -> CaseMetrics:
    """
    Return geometry metadata for one parameterized salt dome.

    Salt-dome coordinates and dimensions are defined in grid-cell units.
    This helper records both the original cell-based values and their
    corresponding physical values.

    The bulb-radius parameters are interpreted as additional radius added
    to the tapered stem radius by the Gaussian bulb profile.

    Parameters
    ----------
    case
        Salt-dome specification.
    grid
        Model-grid specification.

    Returns
    -------
    dict
        Salt-dome geometry metadata.
    """
    center_depth_cells = 0.5 * (
        case.top_depth
        + case.bottom_depth
    )

    dome_height_cells = (
        case.bottom_depth
        - case.top_depth
    )

    maximum_radius_x_cells = (
        case.stem_radius_x
        + case.bulb_additional_radius_x
    )

    maximum_radius_y_cells = (
        case.stem_radius_y
        + case.bulb_additional_radius_y
    )

    return {
        "case_type": "salt_dome",
        "body_count": 1,

        # Fields used by rectangular-body experiments.
        "x_start": "",
        "x_end": "",
        "y_start": "",
        "y_end": "",
        "z_start": "",
        "z_end": "",
        "x_width_cells": "",
        "y_width_cells": "",
        "z_thickness_cells": "",

        # General physical center fields.
        "body_center_x": (
            grid.x_min
            + case.center_x * grid.dx
        ),
        "body_center_y": (
            grid.y_min
            + case.center_y * grid.dy
        ),
        "body_center_depth": (
            grid.z_min
            + center_depth_cells * grid.dz
        ),

        # Fields used by dipping-body experiments.
        "strike_length_cells": "",
        "dip_length_cells": "",
        "body_thickness_cells": "",
        "strike_degrees": "",
        "dip_degrees": "",

        # Salt-dome coordinates in grid-cell units.
        "salt_center_x_cells": case.center_x,
        "salt_center_y_cells": case.center_y,
        "salt_top_depth_cells": case.top_depth,
        "salt_bottom_depth_cells": case.bottom_depth,
        "salt_center_depth_cells": center_depth_cells,
        "salt_height_cells": dome_height_cells,

        # Salt-dome physical coordinates and dimensions.
        "salt_center_x": (
            grid.x_min
            + case.center_x * grid.dx
        ),
        "salt_center_y": (
            grid.y_min
            + case.center_y * grid.dy
        ),
        "salt_top_depth": (
            grid.z_min
            + case.top_depth * grid.dz
        ),
        "salt_bottom_depth": (
            grid.z_min
            + case.bottom_depth * grid.dz
        ),
        "salt_center_depth": (
            grid.z_min
            + center_depth_cells * grid.dz
        ),
        "salt_height": (
            dome_height_cells
            * grid.dz
        ),

        # Stem radii.
        "salt_stem_radius_x_cells": (
            case.stem_radius_x
        ),
        "salt_stem_radius_y_cells": (
            case.stem_radius_y
        ),
        "salt_stem_radius_x": (
            case.stem_radius_x
            * grid.dx
        ),
        "salt_stem_radius_y": (
            case.stem_radius_y
            * grid.dy
        ),

        # Gaussian bulb parameters.
        "salt_bulb_additional_radius_x_cells": (
            case.bulb_additional_radius_x
        ),
        "salt_bulb_additional_radius_y_cells": (
            case.bulb_additional_radius_y
        ),
        "salt_bulb_center_depth_cells": (
            case.bulb_center_depth
        ),
        "salt_bulb_vertical_scale_cells": (
            case.bulb_vertical_scale
        ),
        "salt_bulb_additional_radius_x": (
            case.bulb_additional_radius_x
            * grid.dx
        ),
        "salt_bulb_additional_radius_y": (
            case.bulb_additional_radius_y
            * grid.dy
        ),
        "salt_bulb_center_depth": (
            grid.z_min
            + case.bulb_center_depth * grid.dz
        ),
        "salt_bulb_vertical_scale": (
            case.bulb_vertical_scale
            * grid.dz
        ),

        # Radius-profile controls.
        "salt_taper_fraction": case.taper_fraction,
        "salt_maximum_radius_x_cells": (
            maximum_radius_x_cells
        ),
        "salt_maximum_radius_y_cells": (
            maximum_radius_y_cells
        ),
        "salt_maximum_radius_x": (
            maximum_radius_x_cells
            * grid.dx
        ),
        "salt_maximum_radius_y": (
            maximum_radius_y_cells
            * grid.dy
        ),

        "density_contrast": case.density_contrast,
    }


def _multi_body_geometry_metrics(
    *,
    case: MultiBodyCaseSpec,
) -> CaseMetrics:
    """Return summary geometry metadata for a multi-body case."""
    return {
        "case_type": "multi_body",
        "body_count": case.body_count,
        "x_start": "",
        "x_end": "",
        "y_start": "",
        "y_end": "",
        "z_start": "",
        "z_end": "",
        "x_width_cells": "",
        "y_width_cells": "",
        "z_thickness_cells": "",
        "body_center_x": "",
        "body_center_y": "",
        "body_center_depth": "",
        "strike_length_cells": "",
        "dip_length_cells": "",
        "body_thickness_cells": "",
        "strike_degrees": "",
        "dip_degrees": "",
        "density_contrast": case.density_contrast,
    }

def _basement_relief_geometry_metrics(
    *,
    case: BasementReliefSpec,
    grid: GridSpec,
) -> CaseMetrics:
    """
    Return geometry metadata for one basement-relief model.
    """
    interface_depths = calculate_basement_interface(
        grid=grid,
        body=case,
    )

    minimum_depth = float(
        np.min(interface_depths)
    )
    maximum_depth = float(
        np.max(interface_depths)
    )
    mean_depth = float(
        np.mean(interface_depths)
    )
    depth_standard_deviation = float(
        np.std(interface_depths)
    )

    return {
        "case_type": "basement_relief",
        "body_count": 1,

        "x_start": "",
        "x_end": "",
        "y_start": "",
        "y_end": "",
        "z_start": "",
        "z_end": "",

        "x_width_cells": "",
        "y_width_cells": "",
        "z_thickness_cells": "",

        "body_center_x": "",
        "body_center_y": "",
        "body_center_depth": (
            grid.z_min
            + mean_depth * grid.dz
        ),

        "strike_length_cells": "",
        "dip_length_cells": "",
        "body_thickness_cells": "",
        "strike_degrees": "",
        "dip_degrees": "",

        "basement_base_depth_cells": (
            case.base_depth
        ),
        "basement_min_depth_cells": (
            minimum_depth
        ),
        "basement_max_depth_cells": (
            maximum_depth
        ),
        "basement_mean_depth_cells": (
            mean_depth
        ),
        "basement_depth_std_cells": (
            depth_standard_deviation
        ),

        "basement_slope_x": (
            case.slope_x
        ),
        "basement_slope_y": (
            case.slope_y
        ),

        "gaussian_amplitude_cells": (
            case.gaussian_amplitude
        ),
        "gaussian_center_x_cells": (
            case.gaussian_center_x
        ),
        "gaussian_center_y_cells": (
            case.gaussian_center_y
        ),
        "gaussian_scale_x_cells": (
            case.gaussian_scale_x
        ),
        "gaussian_scale_y_cells": (
            case.gaussian_scale_y
        ),

        "sinusoid_amplitude_cells": (
            case.sinusoid_amplitude
        ),
        "sinusoid_wavelength_cells": (
            case.sinusoid_wavelength
        ),
        "sinusoid_azimuth_degrees": (
            case.sinusoid_azimuth_degrees
        ),
        "sinusoid_phase_degrees": (
            case.sinusoid_phase_degrees
        ),

        "density_contrast": (
            case.density_contrast
        ),
    }

def _gravity_extremum_metrics(
    *,
    gravity_anomaly: np.ndarray,
) -> CaseMetrics:
    """
    Return the location and signed value of the strongest gravity anomaly.

    The strongest response is located using absolute magnitude so the
    calculation works for both positive and negative density contrasts.
    The original signed anomaly value is retained in the returned metrics.

    Parameters
    ----------
    gravity_anomaly
        Two-dimensional gravity anomaly in ``(y, x)`` array order.

    Returns
    -------
    dict
        Maximum-magnitude value and its array indices.

    Raises
    ------
    ValueError
        If the gravity anomaly is not a nonempty two-dimensional array.
    """
    anomaly = np.asarray(gravity_anomaly)

    if anomaly.ndim != 2:
        raise ValueError(
            "gravity_anomaly must be a two-dimensional array in "
            "(y, x) order."
        )

    if anomaly.size == 0:
        raise ValueError(
            "gravity_anomaly must not be empty."
        )

    if not np.all(np.isfinite(anomaly)):
        raise ValueError(
            "gravity_anomaly contains nonfinite values."
        )

    peak_y_index, peak_x_index = np.unravel_index(
        np.argmax(np.abs(anomaly)),
        anomaly.shape,
    )

    signed_peak_value = float(
        anomaly[
            peak_y_index,
            peak_x_index,
        ]
    )

    return {
        "gravity_maximum_magnitude": abs(
            signed_peak_value
        ),
        "gravity_maximum_magnitude_signed_value": (
            signed_peak_value
        ),
        "gravity_maximum_magnitude_x_index": int(
            peak_x_index
        ),
        "gravity_maximum_magnitude_y_index": int(
            peak_y_index
        ),
    }

def build_case_metrics(
    case: CaseSpec,
    gravity_anomaly: np.ndarray,
    grid: GridSpec,
    nonzero_cells: int,
) -> CaseMetrics:
    """
    Combine case geometry with original-gravity metrics.

    Parameters
    ----------
    case
        Synthetic case specification.
    gravity_anomaly
        Forward-modeled gravity anomaly.
    grid
        Grid specification.
    nonzero_cells
        Number of occupied cells in the true density model.

    Returns
    -------
    dict
        Combined case, geometry, and gravity metrics.
    """
    return {
        "case_name": case.name,
        **_case_geometry_metrics(
            case=case,
            grid=grid,
        ),
        "nonzero_cells": nonzero_cells,
        **gravity_response_metrics(
            anomaly=gravity_anomaly,
        ),
        **_gravity_extremum_metrics(
            gravity_anomaly=gravity_anomaly,
        ),
    }

def _validate_true_case_model(
    *,
    model: np.ndarray,
    case: CaseSpec,
    grid: GridSpec,
) -> int:
    """
    Validate a true model and return its occupied-cell count.

    Exact occupied-cell counts are validated for axis-aligned rectangular
    bodies and nonoverlapping multi-body cases. Dipping bodies and salt domes
    are voxelized from continuous geometry, so their exact occupied-cell
    counts are not known before rasterization.

    Parameters
    ----------
    model
        Generated true density model.
    case
        Synthetic case specification.
    grid
        Model-grid specification.

    Returns
    -------
    int
        Number of occupied cells in the generated model.

    Raises
    ------
    RuntimeError
        If validation does not return an occupied-cell count.
    """
    nonzero_cells = validate_density_model(
        model=model,
        body=case,
        grid=grid,
        model_label="true",
        check_rectangular_body=True,
    )

    if nonzero_cells is None:
        raise RuntimeError(
            f"{case.name}: validation did not return "
            "the occupied-cell count."
        )

    return nonzero_cells

def run_forward_experiment(
    body: CaseSpec,
    grid: GridSpec,
    forward_model: GravityForwardModel,
    paths: ExperimentPaths,
) -> tuple[np.ndarray, np.ndarray, CaseMetrics]:
    """
    Build and forward model one supported synthetic case.

    Parameters
    ----------
    body
        Synthetic case specification.
    grid
        Grid specification.
    forward_model
        Gravity forward-modeling interface.
    paths
        Experiment input and output paths.

    Returns
    -------
    tuple
        True density model, gravity anomaly, and case metrics.
    """
    true_model = build_density_model(
        grid=grid,
        case=body,
    )

    nonzero_cells = _validate_true_case_model(
        model=true_model,
        case=body,
        grid=grid,
    )

    gravity_anomaly = forward_model.calculate(
        model=true_model,
    )

    metrics = build_case_metrics(
        case=body,
        gravity_anomaly=gravity_anomaly,
        grid=grid,
        nonzero_cells=nonzero_cells,
    )

    print_case_summary(
        body=body,
        grid=grid,
        true_model=true_model,
        gravity_anomaly=gravity_anomaly,
        nonzero_cells=nonzero_cells,
    )

    true_model_path, metadata_path = save_true_model(
        model=true_model,
        grid=grid,
        body=body,
        output_directory=paths.true_models,
    )

    gravity_path = save_gravity_response(
        gravity_anomaly=gravity_anomaly,
        case_name=body.name,
        output_directory=paths.forward_responses,
    )

    gravity_figure_path = (
        paths.figures
        / f"{body.name}_gravity_anomaly.png"
    )

    plot_gravity_anomaly(
        anomaly=gravity_anomaly,
        grid=grid,
        case_name=body.name,
        output_path=gravity_figure_path,
    )

    print_saved_paths(
        paths={
            "True model": true_model_path,
            "True-model metadata": metadata_path,
            "Gravity": gravity_path,
            "Gravity figure": gravity_figure_path,
        },
    )

    return true_model, gravity_anomaly, metrics


def create_comparison_figures(
    anomalies: dict[str, np.ndarray],
    grid: GridSpec,
    figures_directory: Path,
) -> None:
    """
    Create cross-case gravity comparison figures.

    Parameters
    ----------
    anomalies
        Mapping from case names to gravity anomalies.
    grid
        Grid specification.
    figures_directory
        Destination directory for comparison figures.
    """
    if len(anomalies) < 2:
        return

    comparison_path = (
        figures_directory
        / "gravity_comparison.png"
    )

    plot_gravity_comparison(
        anomalies=anomalies,
        grid=grid,
        output_path=comparison_path,
    )

    profile_path = (
        figures_directory
        / "gravity_center_profiles.png"
    )

    plot_gravity_center_profiles(
        anomalies=anomalies,
        grid=grid,
        output_path=profile_path,
    )

    print_saved_paths(
        paths={
            "Cross-case gravity comparison": comparison_path,
            "Cross-case center profiles": profile_path,
        },
    )