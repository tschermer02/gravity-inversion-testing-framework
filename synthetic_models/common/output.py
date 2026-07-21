from __future__ import annotations

from pathlib import Path

import numpy as np

from synthetic_models.common.bodies import (
    BasementReliefSpec,
    CaseSpec,
    DippingBodySpec,
    MultiBodyCaseSpec,
    RectangularBodySpec,
    SaltDomeSpec,
    calculate_basement_interface,
)
from synthetic_models.common.grid import GridSpec


def print_case_summary(
    *,
    body: CaseSpec,
    grid: GridSpec,
    true_model: np.ndarray,
    gravity_anomaly: np.ndarray,
    nonzero_cells: int,
) -> None:
    """
    Print diagnostics for a true density model and gravity response.

    Parameters
    ----------
    body
        Synthetic case specification used to generate the model.
    grid
        Shared model-grid specification.
    true_model
        True density model in ``(z, y, x)`` array order.
    gravity_anomaly
        Forward-modeled gravity anomaly.
    nonzero_cells
        Number of occupied cells in the true density model.
    """
    peak_y_index, peak_x_index = np.unravel_index(
        np.argmax(np.abs(gravity_anomaly)),
        gravity_anomaly.shape,
    )

    peak_gravity_value = float(
        gravity_anomaly[
            peak_y_index,
            peak_x_index,
        ]
    )

    print(f"\nProcessing case: {body.name}")
    print("-" * 60)

    _print_case_geometry_summary(
        body=body,
        grid=grid,
    )

    print("\nTrue model")
    print("  Axis order: (z, y, x)")
    print(f"  Shape: {true_model.shape}")
    print(f"  Data type: {true_model.dtype}")
    print(
        "  C-contiguous: "
        f"{true_model.flags['C_CONTIGUOUS']}"
    )
    print(f"  Nonzero cells: {nonzero_cells}")
    print(f"  Minimum: {true_model.min():.8f}")
    print(f"  Maximum: {true_model.max():.8f}")

    print("\nGravity response")
    print(f"  Shape: {gravity_anomaly.shape}")
    print(f"  Data type: {gravity_anomaly.dtype}")
    print(f"  Minimum: {gravity_anomaly.min():.8f}")
    print(f"  Maximum: {gravity_anomaly.max():.8f}")
    print(f"  Mean: {gravity_anomaly.mean():.8f}")
    print(
        "  Maximum-magnitude index: "
        f"x={peak_x_index}, y={peak_y_index}"
    )
    print(
        "  Maximum-magnitude value: "
        f"{peak_gravity_value:.8f}"
    )


def _print_case_geometry_summary(
    *,
    body: CaseSpec,
    grid: GridSpec,
) -> None:
    """
    Print geometry-specific information for one synthetic case.

    Parameters
    ----------
    body
        Synthetic case specification to summarize.
    grid
        Shared model-grid specification.

    Raises
    ------
    TypeError
        If the case type is not supported.
    """
    if isinstance(
        body,
        RectangularBodySpec,
    ):
        _print_rectangular_body_summary(
            body,
        )
        return

    if isinstance(
        body,
        DippingBodySpec,
    ):
        _print_dipping_body_summary(
            body,
        )
        return

    if isinstance(
        body,
        MultiBodyCaseSpec,
    ):
        _print_multi_body_summary(
            body,
        )
        return

    if isinstance(
        body,
        SaltDomeSpec,
    ):
        _print_salt_dome_summary(
            body,
        )
        return

    if isinstance(
        body,
        BasementReliefSpec,
    ):
        _print_basement_relief_summary(
            body=body,
            grid=grid,
        )
        return

    raise TypeError(
        "Unsupported case specification: "
        f"{type(body).__name__}."
    )


def _print_rectangular_body_summary(
    body: RectangularBodySpec,
) -> None:
    """Print one axis-aligned rectangular-body summary."""
    print("  Type: Single rectangular body")
    print(
        f"  x indices: "
        f"{body.x_start}:{body.x_end}"
    )
    print(
        f"  y indices: "
        f"{body.y_start}:{body.y_end}"
    )
    print(
        f"  z indices: "
        f"{body.z_start}:{body.z_end}"
    )
    print(
        "  Density contrast: "
        f"{body.density_contrast}"
    )


def _print_dipping_body_summary(
    body: DippingBodySpec,
) -> None:
    """Print one dipping-body geometry summary."""
    print("  Type: Dipping or elongated body")
    print(
        "  Center indices: "
        f"x={body.center_x:.3f}, "
        f"y={body.center_y:.3f}, "
        f"z={body.center_z:.3f}"
    )
    print(
        "  Strike length: "
        f"{body.strike_length:.3f} cells"
    )
    print(
        "  Dip length: "
        f"{body.dip_length:.3f} cells"
    )
    print(
        "  Thickness: "
        f"{body.thickness:.3f} cells"
    )
    print(
        "  Strike: "
        f"{body.strike_degrees:.3f} degrees"
    )
    print(
        "  Dip: "
        f"{body.dip_degrees:.3f} degrees"
    )
    print(
        "  Density contrast: "
        f"{body.density_contrast}"
    )


def _print_multi_body_summary(
    body: MultiBodyCaseSpec,
) -> None:
    """Print a multi-body case summary."""
    print("  Type: Multi-body")
    print(f"  Bodies: {body.body_count}")
    print(
        "  Allow overlap: "
        f"{body.allow_overlap}"
    )

    for index, sub_body in enumerate(
        body.bodies,
        start=1,
    ):
        print(
            f"    Body {index}: "
            f"{sub_body.name}"
        )
        print(
            "      x: "
            f"{sub_body.x_start}:"
            f"{sub_body.x_end}"
        )
        print(
            "      y: "
            f"{sub_body.y_start}:"
            f"{sub_body.y_end}"
        )
        print(
            "      z: "
            f"{sub_body.z_start}:"
            f"{sub_body.z_end}"
        )
        print(
            "      density: "
            f"{sub_body.density_contrast}"
        )


def _print_salt_dome_summary(
    body: SaltDomeSpec,
) -> None:
    """Print one salt-dome geometry summary."""
    print("  Type: Salt dome")
    print(
        "  Horizontal center: "
        f"x={body.center_x:.3f}, "
        f"y={body.center_y:.3f}"
    )
    print(
        "  Depth interval: "
        f"{body.top_depth:.3f} to "
        f"{body.bottom_depth:.3f} cells"
    )
    print(
        "  Vertical height: "
        f"{body.height:.3f} cells"
    )
    print(
        "  Stem radii: "
        f"x={body.stem_radius_x:.3f}, "
        f"y={body.stem_radius_y:.3f} cells"
    )
    print(
        "  Additional bulb radii: "
        f"x={body.bulb_additional_radius_x:.3f}, "
        f"y={body.bulb_additional_radius_y:.3f} "
        "cells"
    )
    print(
        "  Bulb center depth: "
        f"{body.bulb_center_depth:.3f} cells"
    )
    print(
        "  Bulb vertical scale: "
        f"{body.bulb_vertical_scale:.3f} cells"
    )
    print(
        "  Taper fraction: "
        f"{body.taper_fraction:.3f}"
    )
    print(
        "  Maximum possible radii: "
        f"x={body.maximum_possible_radius_x:.3f}, "
        f"y={body.maximum_possible_radius_y:.3f} "
        "cells"
    )
    print(
        "  Density contrast: "
        f"{body.density_contrast}"
    )


def _print_basement_relief_summary(
    *,
    body: BasementReliefSpec,
    grid: GridSpec,
) -> None:
    """Print one basement-relief geometry summary."""
    interface_depths = calculate_basement_interface(
        grid=grid,
        body=body,
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

    print("  Type: Basement relief")
    print(
        "  Base interface depth: "
        f"{body.base_depth:.3f} cells"
    )
    print(
        "  Interface depth range: "
        f"{minimum_depth:.3f} to "
        f"{maximum_depth:.3f} cells"
    )
    print(
        "  Mean interface depth: "
        f"{mean_depth:.3f} cells"
    )
    print(
        "  Interface depth standard deviation: "
        f"{depth_standard_deviation:.3f} cells"
    )
    print(
        "  Reference position: "
        f"x={body.reference_x:.3f}, "
        f"y={body.reference_y:.3f}"
    )
    print(
        "  Planar slopes: "
        f"x={body.slope_x:.4f}, "
        f"y={body.slope_y:.4f}"
    )
    print(
        "  Gaussian amplitude: "
        f"{body.gaussian_amplitude:.3f} cells"
    )
    print(
        "  Gaussian center: "
        f"x={body.gaussian_center_x:.3f}, "
        f"y={body.gaussian_center_y:.3f}"
    )
    print(
        "  Gaussian scales: "
        f"x={body.gaussian_scale_x:.3f}, "
        f"y={body.gaussian_scale_y:.3f} cells"
    )
    print(
        "  Sinusoidal amplitude: "
        f"{body.sinusoid_amplitude:.3f} cells"
    )
    print(
        "  Sinusoidal wavelength: "
        f"{body.sinusoid_wavelength:.3f} cells"
    )
    print(
        "  Sinusoidal azimuth: "
        f"{body.sinusoid_azimuth_degrees:.3f} "
        "degrees"
    )
    print(
        "  Sinusoidal phase: "
        f"{body.sinusoid_phase_degrees:.3f} "
        "degrees"
    )
    print(
        "  Density contrast: "
        f"{body.density_contrast}"
    )


def print_recovered_model_summary(
    recovered_model: np.ndarray,
) -> None:
    """
    Print diagnostics for a CNN-recovered density model.

    Parameters
    ----------
    recovered_model
        CNN-recovered density model in ``(z, y, x)`` array order.
    """
    peak_z, peak_y, peak_x = np.unravel_index(
        np.argmax(np.abs(recovered_model)),
        recovered_model.shape,
    )

    peak_value = float(
        recovered_model[
            peak_z,
            peak_y,
            peak_x,
        ]
    )

    print("\nRecovered CNN model")
    print(f"  Shape: {recovered_model.shape}")
    print(f"  Data type: {recovered_model.dtype}")
    print(
        "  C-contiguous: "
        f"{recovered_model.flags['C_CONTIGUOUS']}"
    )
    print(f"  Minimum: {recovered_model.min():.8f}")
    print(f"  Maximum: {recovered_model.max():.8f}")
    print(f"  Mean: {recovered_model.mean():.8f}")
    print(
        "  Maximum-magnitude index: "
        f"x={peak_x}, y={peak_y}, z={peak_z}"
    )
    print(
        "  Maximum-magnitude value: "
        f"{peak_value:.8f}"
    )


def print_gravity_fit_summary(
    gravity_metrics: dict[str, str | float | int],
) -> None:
    """
    Print the main recovered-gravity fit metrics.

    Parameters
    ----------
    gravity_metrics
        Dictionary containing recovered-gravity comparison metrics.
    """
    print("\nRecovered-model gravity fit")
    print(
        "  RMSE: "
        f"{gravity_metrics['cnn_gravity_rmse']:.8f}"
    )
    print(
        "  Relative L2 error: "
        f"{gravity_metrics['cnn_gravity_relative_l2']:.8f}"
    )
    print(
        "  Correlation: "
        f"{gravity_metrics['cnn_gravity_correlation']:.8f}"
    )
    print(
        "  Maximum absolute error: "
        f"{gravity_metrics['cnn_gravity_max_absolute_error']:.8f}"
    )
    print(
        "  Gravity peak ratio: "
        f"{gravity_metrics['gravity_peak_ratio']:.8f}"
    )


def print_saved_paths(
    *,
    paths: dict[str, Path],
) -> None:
    """
    Print a labeled collection of saved output paths.

    Parameters
    ----------
    paths
        Mapping from output labels to saved file paths.
    """
    print("\nSaved outputs")

    for label, path in paths.items():
        print(f"  {label}: {path}")