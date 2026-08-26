from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.axes import Axes

from synthetic_models.common.bodies import RectangularBodySpec, CaseSpec, MultiBodyCaseSpec, DippingBodySpec, SaltDomeSpec, BasementReliefSpec, calculate_basement_interface
from synthetic_models.common.grid import GridSpec

def plot_gravity_anomaly(
    anomaly: np.ndarray,
    grid: GridSpec,
    case_name: str,
    output_path: Path,
    *,
    cmap: str = "RdBu_r",
    zero_centered: bool = True,
) -> None:
    """
    Plot a 2-D surface gravity anomaly map.

    The dominant anomaly is identified using absolute magnitude, allowing
    both positive and negative gravity responses to be displayed correctly.
    A zero-centered diverging color scale is used to preserve polarity.
    """
    expected_shape = (
        grid.ny,
        grid.nx,
    )

    if anomaly.shape != expected_shape:
        raise ValueError(
            f"Expected gravity shape {expected_shape}, "
            f"but received {anomaly.shape}."
        )

    if not np.all(np.isfinite(anomaly)):
        raise ValueError(
            "Gravity anomaly contains NaN or infinite values."
        )

    if zero_centered:
        anomaly_limit = float(np.max(np.abs(anomaly)))
        if np.isclose(anomaly_limit, 0.0):
            anomaly_limit = 1.0
        color_minimum = -anomaly_limit
        color_maximum = anomaly_limit
    else:
        color_minimum = float(np.min(anomaly))
        color_maximum = float(np.max(anomaly))
        if np.isclose(color_minimum, color_maximum):
            color_maximum = color_minimum + 1.0

    figure, axis = plt.subplots(
        figsize=(8, 7),
        constrained_layout=True,
    )

    image = axis.imshow(
        anomaly,
        origin="lower",
        extent=(
            grid.x_min,
            grid.x_max,
            grid.y_min,
            grid.y_max,
        ),
        aspect="equal",
        cmap=cmap,
        vmin=color_minimum,
        vmax=color_maximum,
    )

    peak_y_index, peak_x_index = np.unravel_index(
        np.argmax(np.abs(anomaly)),
        anomaly.shape,
    )

    peak_x = (
        grid.x_min
        + (peak_x_index + 0.5) * grid.dx
    )

    peak_y = (
        grid.y_min
        + (peak_y_index + 0.5) * grid.dy
    )

    signed_peak = float(
        anomaly[
            peak_y_index,
            peak_x_index,
        ]
    )

    axis.plot(
        peak_x,
        peak_y,
        marker="x",
        markersize=9,
        markeredgewidth=2,
        linestyle="None",
        label=(
            "Dominant anomaly "
            f"({signed_peak:.5g})"
        ),
    )

    axis.set_title(
        f"{case_name}: synthetic gravity anomaly"
    )

    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.legend()

    figure.colorbar(
        image,
        ax=axis,
        label="Gravity anomaly",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

def plot_gravity_comparison(
    anomalies: dict[str, np.ndarray],
    grid: GridSpec,
    output_path: Path,
) -> None:
    """
    Plot several gravity anomalies using one zero-centered color scale.
    """
    if not anomalies:
        raise ValueError(
            "No gravity anomalies were provided."
        )

    expected_shape = (
        grid.ny,
        grid.nx,
    )

    for case_name, anomaly in anomalies.items():
        if anomaly.shape != expected_shape:
            raise ValueError(
                f"{case_name}: expected gravity shape "
                f"{expected_shape}, but received "
                f"{anomaly.shape}."
            )

        if not np.all(np.isfinite(anomaly)):
            raise ValueError(
                f"{case_name}: gravity anomaly contains "
                "NaN or infinite values."
            )

    global_limit = max(
        float(np.max(np.abs(anomaly)))
        for anomaly in anomalies.values()
    )

    if np.isclose(global_limit, 0.0):
        global_limit = 1.0

    case_names = list(
        anomalies.keys()
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(case_names),
        figsize=(
            5 * len(case_names),
            5,
        ),
        constrained_layout=True,
    )

    if len(case_names) == 1:
        axes = [axes]

    image = None

    for axis, case_name in zip(
        axes,
        case_names,
    ):
        anomaly = anomalies[case_name]

        image = axis.imshow(
            anomaly,
            origin="lower",
            extent=(
                grid.x_min,
                grid.x_max,
                grid.y_min,
                grid.y_max,
            ),
            aspect="equal",
            vmin=-global_limit,
            vmax=global_limit,
            cmap="viridis",
        )

        peak_y_index, peak_x_index = np.unravel_index(
            np.argmax(np.abs(anomaly)),
            anomaly.shape,
        )

        peak_x = (
            grid.x_min
            + (peak_x_index + 0.5) * grid.dx
        )

        peak_y = (
            grid.y_min
            + (peak_y_index + 0.5) * grid.dy
        )

        signed_peak = float(
            anomaly[
                peak_y_index,
                peak_x_index,
            ]
        )

        axis.plot(
            peak_x,
            peak_y,
            marker="x",
            markersize=8,
            markeredgewidth=2,
            linestyle="None",
        )

        axis.set_title(
            f"{case_name}\n"
            f"dominant = {signed_peak:.5f}"
        )

        axis.set_xlabel("x")
        axis.set_ylabel("y")

    if image is None:
        raise RuntimeError(
            "Comparison image was not created."
        )

    figure.colorbar(
        image,
        ax=axes,
        label="Gravity anomaly",
        shrink=0.85,
    )

    figure.suptitle(
        "Synthetic gravity-response comparison",
        fontsize=14,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

def plot_gravity_center_profiles(
    anomalies: dict[str, np.ndarray],
    grid: GridSpec,
    output_path: Path,
) -> None:
    """
    Plot center-line gravity profiles for multiple cases.

    Each anomaly is sampled along x through the middle y row.
    """

    if not anomalies:
        raise ValueError("No gravity anomalies were provided.")

    expected_shape = (grid.ny, grid.nx)

    for case_name, anomaly in anomalies.items():
        if anomaly.shape != expected_shape:
            raise ValueError(
                f"{case_name}: expected gravity shape {expected_shape}, "
                f"but received {anomaly.shape}."
            )

        if not np.all(np.isfinite(anomaly)):
            raise ValueError(
                f"{case_name}: gravity anomaly contains NaN or infinite values."
            )

    x_coordinates = np.linspace(
        grid.x_min,
        grid.x_max,
        grid.nx,
    )

    center_y_index = grid.ny // 2

    figure, axis = plt.subplots(
        figsize=(9, 6),
        constrained_layout=True,
    )

    for case_name, anomaly in anomalies.items():
        center_profile = anomaly[center_y_index, :]

        axis.plot(
            x_coordinates,
            center_profile,
            label=case_name,
            linewidth=2,
        )

    axis.set_title(
        "Center-line gravity profiles versus source depth"
    )

    axis.set_xlabel("x")
    axis.set_ylabel("Gravity anomaly")
    axis.grid(True, alpha=0.3)
    axis.legend()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

def plot_gravity_fit_comparison(
    original_gravity: np.ndarray,
    recovered_gravity: np.ndarray,
    grid: GridSpec,
    case_name: str,
    output_path: Path,
    *,
    gravity_cmap: str = "RdBu_r",
    gravity_zero_centered: bool = True,
) -> None:
    """
    Compare original gravity, recovered-model gravity, and their residual.

    The original and recovered maps use one symmetric, zero-centered color
    scale so amplitude and polarity can be compared directly.
    """
    expected_shape = (
        grid.ny,
        grid.nx,
    )

    for name, anomaly in {
        "original": original_gravity,
        "recovered": recovered_gravity,
    }.items():
        if anomaly.shape != expected_shape:
            raise ValueError(
                f"{name} gravity has shape {anomaly.shape}; "
                f"expected {expected_shape}."
            )

        if not np.all(np.isfinite(anomaly)):
            raise ValueError(
                f"{name} gravity contains NaN or infinite values."
            )

    residual = (
        recovered_gravity
        - original_gravity
    )

    if gravity_zero_centered:
        gravity_limit = float(
            max(
                np.max(np.abs(original_gravity)),
                np.max(np.abs(recovered_gravity)),
            )
        )
        if np.isclose(gravity_limit, 0.0):
            gravity_limit = 1.0
        gravity_minimum = -gravity_limit
        gravity_maximum = gravity_limit
    else:
        gravity_minimum = float(
            min(np.min(original_gravity), np.min(recovered_gravity))
        )
        gravity_maximum = float(
            max(np.max(original_gravity), np.max(recovered_gravity))
        )
        if np.isclose(gravity_minimum, gravity_maximum):
            gravity_maximum = gravity_minimum + 1.0

    residual_limit = float(
        np.max(np.abs(residual))
    )

    if np.isclose(residual_limit, 0.0):
        residual_limit = 1.0

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(16, 5),
        constrained_layout=True,
    )

    original_image = axes[0].imshow(
        original_gravity,
        origin="lower",
        extent=(
            grid.x_min,
            grid.x_max,
            grid.y_min,
            grid.y_max,
        ),
        aspect="equal",
        cmap=gravity_cmap,
        vmin=gravity_minimum,
        vmax=gravity_maximum,
    )

    axes[0].set_title(
        "Original synthetic gravity"
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    axes[1].imshow(
        recovered_gravity,
        origin="lower",
        extent=(
            grid.x_min,
            grid.x_max,
            grid.y_min,
            grid.y_max,
        ),
        aspect="equal",
        cmap=gravity_cmap,
        vmin=gravity_minimum,
        vmax=gravity_maximum,
    )

    axes[1].set_title(
        "Recovered-model gravity"
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    residual_image = axes[2].imshow(
        residual,
        origin="lower",
        extent=(
            grid.x_min,
            grid.x_max,
            grid.y_min,
            grid.y_max,
        ),
        aspect="equal",
        cmap="RdBu_r",
        vmin=-residual_limit,
        vmax=residual_limit,
    )

    axes[2].set_title(
        "Gravity residual"
    )
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    figure.colorbar(
        original_image,
        ax=axes[:2],
        label="Gravity anomaly",
        shrink=0.85,
    )

    figure.colorbar(
        residual_image,
        ax=axes[2],
        label="Recovered − original",
        shrink=0.85,
    )

    figure.suptitle(
        f"{case_name}: gravity-data reproduction"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

def _signed_max_projection(
    array: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Project using the value with the greatest absolute magnitude.

    Unlike ``numpy.max``, this preserves the sign of negative density
    or residual values.

    Parameters
    ----------
    array
        Three-dimensional model array.
    axis
        Axis to collapse.

    Returns
    -------
    numpy.ndarray
        Two-dimensional signed maximum-magnitude projection.
    """
    maximum_indices = np.argmax(
        np.abs(array),
        axis=axis,
    )

    expanded_indices = np.expand_dims(
        maximum_indices,
        axis=axis,
    )

    return np.take_along_axis(
        array,
        expanded_indices,
        axis=axis,
    ).squeeze(axis=axis)

def _rectangular_body_views(
    *,
    true_model: np.ndarray,
    recovered_model: np.ndarray,
    residual: np.ndarray,
    body: RectangularBodySpec,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
]:
    """Extract orthogonal center slices for one rectangular body."""
    z_index = (
        body.z_start
        + body.z_end
        - 1
    ) // 2

    y_index = (
        body.y_start
        + body.y_end
        - 1
    ) // 2

    x_index = (
        body.x_start
        + body.x_end
        - 1
    ) // 2

    true_xy = true_model[z_index, :, :]
    recovered_xy = recovered_model[z_index, :, :]
    residual_xy = residual[z_index, :, :]

    true_xz = true_model[:, y_index, :]
    recovered_xz = recovered_model[:, y_index, :]
    residual_xz = residual[:, y_index, :]

    true_yz = true_model[:, :, x_index]
    recovered_yz = recovered_model[:, :, x_index]
    residual_yz = residual[:, :, x_index]

    slice_description = (
        f"XY(z={z_index})  "
        f"XZ(y={y_index})  "
        f"YZ(x={x_index})"
    )

    return (
        true_xy,
        recovered_xy,
        residual_xy,
        true_xz,
        recovered_xz,
        residual_xz,
        true_yz,
        recovered_yz,
        residual_yz,
        slice_description,
    )

def _projection_views(
    *,
    true_model: np.ndarray,
    recovered_model: np.ndarray,
    residual: np.ndarray,
    description: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
]:
    """Create signed maximum-magnitude XY, XZ, and YZ projections.

    Parameters
    ----------
    true_model
        True density model in ``(z, y, x)`` order.
    recovered_model
        CNN-recovered density model.
    residual
        Recovered model minus true model.
    description
        Text describing the projection strategy.

    Returns
    -------
    tuple
        True, recovered, and residual views for the XY, XZ, and YZ
        orientations, followed by a descriptive title.
    """
    true_xy = _signed_max_projection(
        true_model,
        axis=0,
    )
    recovered_xy = _signed_max_projection(
        recovered_model,
        axis=0,
    )
    residual_xy = _signed_max_projection(
        residual,
        axis=0,
    )

    true_xz = _signed_max_projection(
        true_model,
        axis=1,
    )
    recovered_xz = _signed_max_projection(
        recovered_model,
        axis=1,
    )
    residual_xz = _signed_max_projection(
        residual,
        axis=1,
    )

    true_yz = _signed_max_projection(
        true_model,
        axis=2,
    )
    recovered_yz = _signed_max_projection(
        recovered_model,
        axis=2,
    )
    residual_yz = _signed_max_projection(
        residual,
        axis=2,
    )

    return (
        true_xy,
        recovered_xy,
        residual_xy,
        true_xz,
        recovered_xz,
        residual_xz,
        true_yz,
        recovered_yz,
        residual_yz,
        description,
    )

def extract_apparent_interface_depth(
    *,
    model: np.ndarray,
    density_contrast: float,
    threshold_fraction: float = 0.5,
) -> np.ndarray:
    """
    Extract the shallowest significant-density cell in each model column.

    The returned interface is an ``(y, x)`` array. Each finite value gives
    the depth, in grid-cell coordinates, of the shallowest voxel whose
    density magnitude exceeds the selected threshold.

    Columns containing no significant recovered density are assigned NaN.

    This is an apparent interface rather than proof that the recovered model
    represents a true basement half-space. A compact CNN-recovered body may
    therefore produce a shallow apparent interface over only part of the
    horizontal domain.

    Parameters
    ----------
    model
        Density model in ``(z, y, x)`` array order.
    density_contrast
        Reference density contrast used to define the threshold magnitude.
    threshold_fraction
        Fraction of the reference density-contrast magnitude used as the
        significance threshold.

    Returns
    -------
    np.ndarray
        Apparent interface depths with shape ``(y, x)``. Depth values refer
        to cell centers in grid-cell coordinates.

    Raises
    ------
    ValueError
        If the input model or threshold parameters are invalid.
    """
    if model.ndim != 3:
        raise ValueError(
            "model must be a three-dimensional array in (z, y, x) order."
        )

    if not np.all(np.isfinite(model)):
        raise ValueError(
            "model contains nonfinite values."
        )

    if not np.isfinite(density_contrast):
        raise ValueError(
            "density_contrast must be finite."
        )

    if density_contrast == 0.0:
        raise ValueError(
            "density_contrast must not be zero."
        )

    if not 0.0 < threshold_fraction <= 1.0:
        raise ValueError(
            "threshold_fraction must lie in the interval (0, 1]."
        )

    threshold = (
        threshold_fraction
        * abs(density_contrast)
    )

    occupied = (
        np.abs(model)
        >= threshold
    )

    column_has_density = np.any(
        occupied,
        axis=0,
    )

    first_occupied_index = np.argmax(
        occupied,
        axis=0,
    )

    interface_depth = (
        first_occupied_index.astype(np.float64)
        + 0.5
    )

    interface_depth[
        ~column_has_density
    ] = np.nan

    return np.ascontiguousarray(
        interface_depth,
        dtype=np.float64,
    )

def plot_basement_interface_comparison(
    *,
    body: BasementReliefSpec,
    true_model: np.ndarray,
    recovered_model: np.ndarray,
    grid: GridSpec,
    output_path: Path,
    threshold_fraction: float = 0.5,
) -> None:
    """
    Plot true and recovered basement-interface depths and their difference.

    The true interface is evaluated directly from the basement-relief
    specification. The recovered apparent interface is defined as the
    shallowest cell in each vertical column whose recovered density
    magnitude exceeds a selected fraction of the true density contrast.

    Difference is calculated as:

        recovered depth - true depth

    Therefore:

    - negative values indicate an interface recovered too shallowly;
    - positive values indicate an interface recovered too deeply;
    - gray or blank regions indicate no recovered density above threshold.

    Parameters
    ----------
    body
        Basement-relief specification.
    true_model
        True density model in ``(z, y, x)`` order. It is included for shape
        validation and consistency with other plotting functions.
    recovered_model
        CNN-recovered density model in ``(z, y, x)`` order.
    grid
        Shared model-grid specification.
    output_path
        Destination image path.
    threshold_fraction
        Fraction of ``abs(body.density_contrast)`` used to identify
        significant recovered density.

    Raises
    ------
    ValueError
        If model shapes do not agree with the grid.
    """
    expected_shape = (
        grid.nz,
        grid.ny,
        grid.nx,
    )

    if true_model.shape != expected_shape:
        raise ValueError(
            f"{body.name}: true_model has shape "
            f"{true_model.shape}, expected {expected_shape}."
        )

    if recovered_model.shape != expected_shape:
        raise ValueError(
            f"{body.name}: recovered_model has shape "
            f"{recovered_model.shape}, expected {expected_shape}."
        )

    true_interface_cells = calculate_basement_interface(
        grid=grid,
        body=body,
    )

    recovered_interface_cells = extract_apparent_interface_depth(
        model=recovered_model,
        density_contrast=body.density_contrast,
        threshold_fraction=threshold_fraction,
    )

    true_interface_depth = (
        grid.z_min
        + true_interface_cells
        * grid.dz
    )

    recovered_interface_depth = (
        grid.z_min
        + recovered_interface_cells
        * grid.dz
    )

    interface_difference = (
        recovered_interface_depth
        - true_interface_depth
    )

    valid_recovered_mask = np.isfinite(
        recovered_interface_depth
    )

    recovered_coverage_fraction = float(
        np.mean(valid_recovered_mask)
    )

    true_minimum = float(
        np.nanmin(true_interface_depth)
    )
    true_maximum = float(
        np.nanmax(true_interface_depth)
    )

    if np.any(valid_recovered_mask):
        recovered_minimum = float(
            np.nanmin(recovered_interface_depth)
        )
        recovered_maximum = float(
            np.nanmax(recovered_interface_depth)
        )

        depth_minimum = min(
            true_minimum,
            recovered_minimum,
        )
        depth_maximum = max(
            true_maximum,
            recovered_maximum,
        )
    else:
        depth_minimum = true_minimum
        depth_maximum = true_maximum

    finite_difference = interface_difference[
        np.isfinite(interface_difference)
    ]

    if finite_difference.size > 0:
        difference_limit = float(
            np.max(
                np.abs(
                    finite_difference
                )
            )
        )
    else:
        difference_limit = grid.dz

    if difference_limit == 0.0:
        difference_limit = grid.dz

    horizontal_extent = [
        grid.x_min,
        grid.x_min + grid.nx * grid.dx,
        grid.y_min,
        grid.y_min + grid.ny * grid.dy,
    ]

    recovered_for_plot = np.ma.masked_invalid(
        recovered_interface_depth
    )

    difference_for_plot = np.ma.masked_invalid(
        interface_difference
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(17, 5.5),
        constrained_layout=True,
    )

    true_image = axes[0].imshow(
        true_interface_depth,
        origin="lower",
        extent=horizontal_extent,
        aspect="equal",
        cmap="viridis_r",
        vmin=depth_minimum,
        vmax=depth_maximum,
    )

    axes[0].set_title(
        "True interface depth"
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    recovered_image = axes[1].imshow(
        recovered_for_plot,
        origin="lower",
        extent=horizontal_extent,
        aspect="equal",
        cmap="viridis_r",
        vmin=depth_minimum,
        vmax=depth_maximum,
    )

    axes[1].set_title(
        "Recovered apparent interface\n"
        f"coverage={100.0 * recovered_coverage_fraction:.1f}%"
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    difference_image = axes[2].imshow(
        difference_for_plot,
        origin="lower",
        extent=horizontal_extent,
        aspect="equal",
        cmap="RdBu",
        vmin=-difference_limit,
        vmax=difference_limit,
    )

    axes[2].set_title(
        "Recovered − true depth\n"
        "red = too shallow, blue = too deep"
    )
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    depth_colorbar = figure.colorbar(
        true_image,
        ax=[
            axes[0],
            axes[1],
        ],
        shrink=0.85,
        pad=0.03,
    )

    depth_colorbar.set_label(
        "Interface depth"
    )

    difference_colorbar = figure.colorbar(
        difference_image,
        ax=axes[2],
        shrink=0.85,
        pad=0.03,
    )

    difference_colorbar.set_label(
        "Depth difference"
    )

    figure.suptitle(
        f"{body.name}\n"
        "Basement-interface comparison "
        f"(threshold={threshold_fraction:.2f} × "
        "|density contrast|)",
        fontsize=14,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

def plot_true_recovered_density_comparison(
    true_model: np.ndarray,
    recovered_model: np.ndarray,
    body: CaseSpec,
    grid: GridSpec,
    case_name: str,
    output_path: Path,
) -> None:
    """
    Compare true and CNN-recovered density models.
    """

    expected_shape = (
        grid.nz,
        grid.ny,
        grid.nx,
    )

    for model_name, model in {
        "true": true_model,
        "recovered": recovered_model,
    }.items():
        if model.shape != expected_shape:
            raise ValueError(
                f"{case_name}: expected {model_name} model shape "
                f"{expected_shape}, but received {model.shape}."
            )

        if not np.all(np.isfinite(model)):
            raise ValueError(
                f"{case_name}: {model_name} model contains "
                "NaN or infinite values."
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    residual = recovered_model - true_model

    if isinstance(body, RectangularBodySpec):
        (
            true_xy,
            recovered_xy,
            residual_xy,
            true_xz,
            recovered_xz,
            residual_xz,
            true_yz,
            recovered_yz,
            residual_yz,
            slice_description,
        ) = _rectangular_body_views(
            true_model=true_model,
            recovered_model=recovered_model,
            residual=residual,
            body=body,
        )

    elif isinstance(body, DippingBodySpec):
        (
            true_xy,
            recovered_xy,
            residual_xy,
            true_xz,
            recovered_xz,
            residual_xz,
            true_yz,
            recovered_yz,
            residual_yz,
            slice_description,
        ) = _projection_views(
            true_model=true_model,
            recovered_model=recovered_model,
            residual=residual,
            description=(
                "Maximum-magnitude projections "
                f"(strike={body.strike_degrees:.1f}°, "
                f"dip={body.dip_degrees:.1f}°)"
            ),
        )

    elif isinstance(body, MultiBodyCaseSpec):
        (
            true_xy,
            recovered_xy,
            residual_xy,
            true_xz,
            recovered_xz,
            residual_xz,
            true_yz,
            recovered_yz,
            residual_yz,
            slice_description,
        ) = _projection_views(
            true_model=true_model,
            recovered_model=recovered_model,
            residual=residual,
            description=(
                "Maximum-magnitude projections "
                f"({body.body_count} bodies)"
            ),
        )

    elif isinstance(body, SaltDomeSpec):
        (
            true_xy,
            recovered_xy,
            residual_xy,
            true_xz,
            recovered_xz,
            residual_xz,
            true_yz,
            recovered_yz,
            residual_yz,
            slice_description,
        ) = _projection_views(
            true_model=true_model,
            recovered_model=recovered_model,
            residual=residual,
            description=(
                "Maximum-magnitude projections "
                f"(salt dome, density contrast="
                f"{body.density_contrast:.4g})"
            ),
        )

    elif isinstance(
        body,
        BasementReliefSpec,
    ):
        (
            true_xy,
            recovered_xy,
            residual_xy,
            true_xz,
            recovered_xz,
            residual_xz,
            true_yz,
            recovered_yz,
            residual_yz,
            slice_description,
        ) = _projection_views(
            true_model=true_model,
            recovered_model=recovered_model,
            residual=residual,
            description=(
                "Maximum-magnitude projections "
                f"(basement relief, density contrast="
                f"{body.density_contrast:.4g})"
            ),
        )

    else:
        raise TypeError(
            f"{case_name}: unsupported case type "
            f"{type(body).__name__}."
        )

    density_limit = float(
        max(
            np.max(np.abs(true_model)),
            np.max(np.abs(recovered_model)),
        )
    )

    if np.isclose(
        density_limit,
        0.0,
    ):
        density_limit = 1.0

    residual_limit = float(
        max(
            np.max(np.abs(residual_xy)),
            np.max(np.abs(residual_xz)),
            np.max(np.abs(residual_yz)),
        )
    )

    if np.isclose(
        residual_limit,
        0.0,
    ):
        residual_limit = 1.0

    xy_extent = (
        grid.x_min,
        grid.x_max,
        grid.y_min,
        grid.y_max,
    )

    xz_extent = (
        grid.x_min,
        grid.x_max,
        grid.z_max,
        grid.z_min,
    )

    yz_extent = (
        grid.y_min,
        grid.y_max,
        grid.z_max,
        grid.z_min,
    )

    figure, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(12, 12),
        constrained_layout=True,
    )

    density_images = [
        (true_xy, recovered_xy),
        (true_xz, recovered_xz),
        (true_yz, recovered_yz),
    ]

    residual_images = [
        residual_xy,
        residual_xz,
        residual_yz,
    ]

    extents = [
        xy_extent,
        xz_extent,
        yz_extent,
    ]

    aspects = [
        "equal",
        "auto",
        "auto",
    ]

    row_labels = [
        "XY",
        "XZ",
        "YZ",
    ]

    density_image = None
    residual_image = None

    for row_index in range(3):
        true_image, recovered_image = density_images[row_index]
        current_residual = residual_images[row_index]

        density_image = axes[row_index, 0].imshow(
            true_image,
            origin="upper" if row_index > 0 else "lower",
            extent=extents[row_index],
            aspect=aspects[row_index],
            cmap="viridis",
            vmin=-density_limit,
            vmax=density_limit,   
        )

        axes[row_index, 1].imshow(
            recovered_image,
            origin="upper" if row_index > 0 else "lower",
            extent=extents[row_index],
            aspect=aspects[row_index],
            cmap="viridis",
            vmin=-density_limit,
            vmax=density_limit,
        )

        residual_image = axes[row_index, 2].imshow(
            current_residual,
            origin="upper" if row_index > 0 else "lower",
            extent=extents[row_index],
            aspect=aspects[row_index],
            cmap="RdBu_r",
            vmin=-residual_limit,
            vmax=residual_limit,
        )

        axes[row_index, 0].set_ylabel(
            "y" if row_index == 0 else "Depth"
        )

        axes[row_index, 0].text(
            -0.18,
            0.5,
            row_labels[row_index],
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            verticalalignment="center",
            fontweight="bold",
        )

    axes[0, 0].set_title("True")
    axes[0, 1].set_title("Recovered")
    axes[0, 2].set_title("Residual")

    for axis in axes[0, :]:
        axis.set_xlabel("x")
        axis.set_ylabel("y")

    for axis in axes[1, :]:
        axis.set_xlabel("x")
        axis.set_ylabel("Depth")

    for axis in axes[2, :]:
        axis.set_xlabel("y")
        axis.set_ylabel("Depth")

    if density_image is None or residual_image is None:
        raise RuntimeError(
            "Density comparison images were not created."
        )

    figure.colorbar(
        density_image,
        ax=axes[:, :2],
        shrink=0.85,
        label="Density contrast",
    )

    figure.colorbar(
        residual_image,
        ax=axes[:, 2],
        shrink=0.85,
        label="Recovered - true",
    )

    figure.suptitle(
        f"{case_name}\n{slice_description}",
        fontsize=14,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    if isinstance(
        body,
        BasementReliefSpec,
    ):
        basement_interface_path = (
            output_path.parent
            / f"{body.name}_basement_interface_comparison.png"
        )

        plot_basement_interface_comparison(
            body=body,
            true_model=true_model,
            recovered_model=recovered_model,
            grid=grid,
            output_path=basement_interface_path,
            threshold_fraction=0.5,
        )

def plot_noise_inversion_comparison(
    *,
    clean_gravity: np.ndarray,
    noisy_gravity: np.ndarray,
    recovered_gravity: np.ndarray,
    recovered_model: np.ndarray,
    grid: GridSpec,
    case_name: str,
    noise_percent: float,
    output_path: Path,
) -> None:
    """
    Plot the complete noisy-gravity inversion workflow.

    The four panels show:

    1. Clean gravity generated from the true density model.
    2. Noisy gravity supplied directly to the pretrained CNN.
    3. Gravity generated by forward modeling the CNN-recovered density.
    4. Maximum-magnitude vertical projection of the recovered density.

    Parameters
    ----------
    clean_gravity
        Noise-free gravity response generated from the true model.

    noisy_gravity
        Gravity response after noise was added. This is the array supplied
        directly to the pretrained CNN.

    recovered_gravity
        Gravity response obtained by forward modeling the CNN-recovered
        density model.

    recovered_model
        CNN-recovered density model in ``(z, y, x)`` array order.

    grid
        Shared model-grid specification.

    case_name
        Name of the current noise case.

    noise_percent
        Noise standard deviation expressed as a percentage of the maximum
        absolute clean gravity anomaly.

    output_path
        Location where the completed figure will be saved.
    """

    expected_gravity_shape = (
        grid.ny,
        grid.nx,
    )

    for array_name, array in {
        "clean_gravity": clean_gravity,
        "noisy_gravity": noisy_gravity,
        "recovered_gravity": recovered_gravity,
    }.items():
        if array.shape != expected_gravity_shape:
            raise ValueError(
                f"{array_name} must have shape "
                f"{expected_gravity_shape}, but received "
                f"{array.shape}."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{array_name} contains NaN or infinite values."
            )

    expected_model_shape = (
        grid.nz,
        grid.ny,
        grid.nx,
    )

    if recovered_model.shape != expected_model_shape:
        raise ValueError(
            "recovered_model must have shape "
            f"{expected_model_shape}, but received "
            f"{recovered_model.shape}."
        )

    if not np.all(np.isfinite(recovered_model)):
        raise ValueError(
            "recovered_model contains NaN or infinite values."
        )

    # Use one common scale for all three gravity panels so their
    # amplitudes can be compared directly.
    gravity_minimum = float(
        min(
            np.min(clean_gravity),
            np.min(noisy_gravity),
            np.min(recovered_gravity),
        )
    )

    gravity_maximum = float(
        max(
            np.max(clean_gravity),
            np.max(noisy_gravity),
            np.max(recovered_gravity),
        )
    )

    # Preserve the sign of the strongest-magnitude recovered density
    # voxel in each vertical column.
    maximum_indices = np.argmax(
        np.abs(recovered_model),
        axis=0,
    )

    recovered_projection = np.take_along_axis(
        recovered_model,
        maximum_indices[np.newaxis, :, :],
        axis=0,
    )[0]

    density_limit = float(
        np.max(
            np.abs(recovered_projection)
        )
    )

    if np.isclose(density_limit, 0.0):
        density_limit = 1.0

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(13, 11),
        constrained_layout=True,
    )

    extent = (
        grid.x_min,
        grid.x_max,
        grid.y_min,
        grid.y_max,
    )

    clean_image = axes[0, 0].imshow(
        clean_gravity,
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=gravity_minimum,
        vmax=gravity_maximum,
    )

    axes[0, 0].set_title(
        "Clean gravity\nfrom true density model"
    )
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")

    axes[0, 1].imshow(
        noisy_gravity,
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=gravity_minimum,
        vmax=gravity_maximum,
    )

    axes[0, 1].set_title(
        f"Noisy gravity supplied to CNN\n"
        f"{noise_percent:.1f}% noise"
    )
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")

    axes[1, 0].imshow(
        recovered_gravity,
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=gravity_minimum,
        vmax=gravity_maximum,
    )

    axes[1, 0].set_title(
        "Gravity from CNN-recovered density"
    )
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")

    density_image = axes[1, 1].imshow(
        recovered_projection,
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=-density_limit,
        vmax=density_limit,
    )

    axes[1, 1].set_title(
        "CNN-recovered density\n"
        "maximum-magnitude vertical projection"
    )
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")

    figure.colorbar(
        clean_image,
        ax=[
            axes[0, 0],
            axes[0, 1],
            axes[1, 0],
        ],
        label="Gravity anomaly",
        shrink=0.85,
    )

    figure.colorbar(
        density_image,
        ax=axes[1, 1],
        label="Recovered density contrast",
        shrink=0.85,
    )

    figure.suptitle(
        f"{case_name}\n"
        "Noisy-gravity CNN inversion workflow",
        fontsize=15,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

def plot_noise_robustness_summary(
    *,
    metrics_csv: Path,
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """
    Plot a publication-quality summary of CNN robustness to gravity noise.
    """
    metrics_csv = Path(metrics_csv)
    output_path = Path(output_path)

    if not metrics_csv.is_file():
        raise FileNotFoundError(
            f"Noise metrics CSV does not exist: {metrics_csv}"
        )

    if output_path.suffix.lower() not in {
        ".png",
        ".pdf",
        ".svg",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }:
        raise ValueError(
            "output_path must use a supported figure extension."
        )

    if dpi <= 0:
        raise ValueError("dpi must be greater than zero.")

    required_columns = (
        "noise_percent",
        "model_correlation",
        "model_relative_l2",
        "model_iou",
        "clean_cnn_gravity_correlation",
        "clean_cnn_gravity_relative_l2",
        "snr_db",
    )

    metric_rows = _read_noise_robustness_metrics(
        metrics_csv=metrics_csv,
        required_columns=required_columns,
    )

    metric_rows.sort(
        key=lambda row: float(row["noise_percent"])
    )

    noise_percent = _metric_column_to_array(
        rows=metric_rows,
        column_name="noise_percent",
    )

    model_correlation = _metric_column_to_array(
        rows=metric_rows,
        column_name="model_correlation",
    )

    model_relative_l2 = _metric_column_to_array(
        rows=metric_rows,
        column_name="model_relative_l2",
    )

    model_iou = _metric_column_to_array(
        rows=metric_rows,
        column_name="model_iou",
    )

    clean_gravity_correlation = _metric_column_to_array(
        rows=metric_rows,
        column_name="clean_cnn_gravity_correlation",
    )

    clean_gravity_relative_l2 = _metric_column_to_array(
        rows=metric_rows,
        column_name="clean_cnn_gravity_relative_l2",
    )

    snr_db = _metric_column_to_array(
        rows=metric_rows,
        column_name="snr_db",
        allow_infinite=True,
    )

    _validate_noise_summary_arrays(
        noise_percent=noise_percent,
        metric_arrays={
            "model_correlation": model_correlation,
            "model_relative_l2": model_relative_l2,
            "model_iou": model_iou,
            "clean_cnn_gravity_correlation": (
                clean_gravity_correlation
            ),
            "clean_cnn_gravity_relative_l2": (
                clean_gravity_relative_l2
            ),
            "snr_db": snr_db,
        },
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(13.0, 7.5),
        constrained_layout=True,
    )

    flat_axes = axes.ravel()

    _plot_robustness_metric(
        axis=flat_axes[0],
        noise_percent=noise_percent,
        metric_values=model_correlation,
        title="Density Correlation",
        y_label="Correlation",
        y_limits=(-0.05, 1.05),
    )

    _plot_robustness_metric(
        axis=flat_axes[1],
        noise_percent=noise_percent,
        metric_values=model_relative_l2,
        title="Density Relative L2 Error",
        y_label="Relative L2 error",
        y_minimum=0.0,
    )

    _plot_robustness_metric(
        axis=flat_axes[2],
        noise_percent=noise_percent,
        metric_values=model_iou,
        title="Density Intersection over Union",
        y_label="IoU",
        y_limits=(0.0, 1.0),
    )

    _plot_robustness_metric(
        axis=flat_axes[3],
        noise_percent=noise_percent,
        metric_values=clean_gravity_correlation,
        title="Clean-Gravity Correlation",
        y_label="Correlation",
        y_limits=(-0.05, 1.05),
    )

    _plot_robustness_metric(
        axis=flat_axes[4],
        noise_percent=noise_percent,
        metric_values=clean_gravity_relative_l2,
        title="Clean-Gravity Relative L2 Error",
        y_label="Relative L2 error",
        y_minimum=0.0,
    )

    _plot_snr_metric(
        axis=flat_axes[5],
        noise_percent=noise_percent,
        snr_db=snr_db,
    )

    for axis in flat_axes:
        axis.set_xlabel("Noise level (%)")
        axis.set_xticks(noise_percent)
        axis.tick_params(
            axis="both",
            which="major",
            labelsize=9,
        )

    figure.suptitle(
        "CNN Gravity-Inversion Robustness to Gaussian Noise",
        fontsize=15,
        fontweight="bold",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path

def _read_noise_robustness_metrics(
    *,
    metrics_csv: Path,
    required_columns: tuple[str, ...],
) -> list[dict[str, str]]:
    """
    Read and validate rows from a noise-robustness metrics CSV.
    """
    with metrics_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"{metrics_csv}: CSV does not contain a header."
            )

        missing_columns = [
            column
            for column in required_columns
            if column not in reader.fieldnames
        ]

        if missing_columns:
            raise ValueError(
                f"{metrics_csv}: missing required columns: "
                f"{', '.join(missing_columns)}."
            )

        rows = list(reader)

    if not rows:
        raise ValueError(
            f"{metrics_csv}: CSV does not contain any data rows."
        )

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        for column_name in required_columns:
            value = row.get(column_name)

            if value is None or not value.strip():
                raise ValueError(
                    f"{metrics_csv}: row {row_number}, column "
                    f"'{column_name}' is blank."
                )

    return rows

def _metric_column_to_array(
    *,
    rows: list[dict[str, str]],
    column_name: str,
    allow_infinite: bool = False,
) -> np.ndarray:
    """
    Convert one CSV metric column into a one-dimensional NumPy array.
    """
    values: list[float] = []

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        raw_value = row[column_name]

        try:
            numeric_value = float(raw_value)
        except ValueError as error:
            raise ValueError(
                f"Row {row_number}, column '{column_name}' must be "
                f"numeric, but received {raw_value!r}."
            ) from error

        if np.isnan(numeric_value):
            raise ValueError(
                f"Row {row_number}, column '{column_name}' contains NaN."
            )

        if (
            not allow_infinite
            and not np.isfinite(numeric_value)
        ):
            raise ValueError(
                f"Row {row_number}, column '{column_name}' must be finite."
            )

        values.append(numeric_value)

    return np.asarray(
        values,
        dtype=np.float64,
    )

def _validate_noise_summary_arrays(
    *,
    noise_percent: np.ndarray,
    metric_arrays: dict[str, np.ndarray],
) -> None:
    """
    Validate arrays used in the noise-robustness summary figure.
    """
    if noise_percent.ndim != 1:
        raise ValueError(
            "noise_percent must be a one-dimensional array."
        )

    if noise_percent.size == 0:
        raise ValueError(
            "At least one noise level must be provided."
        )

    if not np.all(np.isfinite(noise_percent)):
        raise ValueError(
            "noise_percent contains nonfinite values."
        )

    if np.any(noise_percent < 0.0):
        raise ValueError(
            "noise_percent must contain only nonnegative values."
        )

    if np.unique(noise_percent).size != noise_percent.size:
        raise ValueError(
            "noise_percent contains duplicate noise levels."
        )

    for metric_name, metric_values in metric_arrays.items():
        if metric_values.ndim != 1:
            raise ValueError(
                f"{metric_name} must be a one-dimensional array."
            )

        if metric_values.shape != noise_percent.shape:
            raise ValueError(
                f"{metric_name} has shape {metric_values.shape}, but "
                f"noise_percent has shape {noise_percent.shape}."
            )

    bounded_metrics = {
        "model_correlation": (-1.0, 1.0),
        "model_iou": (0.0, 1.0),
        "clean_cnn_gravity_correlation": (-1.0, 1.0),
    }

    for metric_name, bounds in bounded_metrics.items():
        metric_values = metric_arrays[metric_name]
        lower_bound, upper_bound = bounds

        if np.any(
            (metric_values < lower_bound)
            | (metric_values > upper_bound)
        ):
            raise ValueError(
                f"{metric_name} must remain within "
                f"[{lower_bound}, {upper_bound}]."
            )

    nonnegative_metrics = (
        "model_relative_l2",
        "clean_cnn_gravity_relative_l2",
    )

    for metric_name in nonnegative_metrics:
        if np.any(metric_arrays[metric_name] < 0.0):
            raise ValueError(
                f"{metric_name} must be nonnegative."
            )

def _plot_robustness_metric(
    *,
    axis: Axes,
    noise_percent: np.ndarray,
    metric_values: np.ndarray,
    title: str,
    y_label: str,
    y_limits: tuple[float, float] | None = None,
    y_minimum: float | None = None,
) -> None:
    """
    Plot one robustness metric against noise percentage.
    """
    axis.plot(
        noise_percent,
        metric_values,
        marker="o",
        linewidth=2.0,
        markersize=7.0,
    )

    axis.set_title(
        title,
        fontsize=11,
        fontweight="bold",
    )

    axis.set_ylabel(y_label)

    axis.grid(
        visible=True,
        alpha=0.3,
        linewidth=0.7,
    )

    if y_limits is not None:
        axis.set_ylim(y_limits)
    elif y_minimum is not None:
        current_bottom, current_top = axis.get_ylim()

        axis.set_ylim(
            bottom=y_minimum,
            top=current_top,
        )

def _plot_snr_metric(
    *,
    axis: Axes,
    noise_percent: np.ndarray,
    snr_db: np.ndarray,
) -> None:
    """
    Plot finite signal-to-noise ratios and annotate infinite SNR cases.

    Parameters
    ----------
    axis
        Matplotlib axis on which to draw.
    noise_percent
        Noise percentages used as x coordinates.
    snr_db
        Signal-to-noise ratio values in decibels.
    """
    finite_mask = np.isfinite(snr_db)

    if np.any(finite_mask):
        axis.plot(
            noise_percent[finite_mask],
            snr_db[finite_mask],
            marker="o",
            linewidth=2.0,
            markersize=7.0,
        )

    infinite_mask = np.isposinf(snr_db)

    if np.any(infinite_mask):
        finite_values = snr_db[finite_mask]

        if finite_values.size:
            annotation_height = float(
                np.max(finite_values)
            )
        else:
            annotation_height = 0.0

        for noise_value in noise_percent[infinite_mask]:
            axis.annotate(
                "∞",
                xy=(
                    noise_value,
                    annotation_height,
                ),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
            )

    axis.set_title(
        "Signal-to-Noise Ratio",
        fontsize=11,
        fontweight="bold",
    )

    axis.set_ylabel("SNR (dB)")

    axis.grid(
        visible=True,
        alpha=0.3,
        linewidth=0.7,
    )

def plot_published_example_density_comparison(
    true_model: np.ndarray,
    recovered_model: np.ndarray,
    grid: GridSpec,
    output_path: Path,
    case_name: str = "Published Example",
) -> None:
    """
    Plot true, recovered, and residual density-model projections.

    This function is intended for the published reproduction example, where
    true and recovered density arrays are available but no synthetic
    ``CaseSpec`` exists.

    The figure contains three columns:

    1. True density model.
    2. CNN-recovered density model.
    3. Residual, defined as recovered minus true.

    The rows show maximum-magnitude projections onto the XY, XZ, and YZ
    planes.

    Parameters
    ----------
    true_model
        True density model with shape ``(nz, ny, nx)``.

    recovered_model
        CNN-recovered density model with shape ``(nz, ny, nx)``.

    grid
        Grid specification defining model dimensions and physical extents.

    output_path
        Path where the figure will be saved.

    case_name
        Name displayed in the figure title.

    Raises
    ------
    ValueError
        If either model has an incorrect shape or contains non-finite values.
    RuntimeError
        If the plotting images are not successfully created.
    """
    expected_shape = (
        grid.nz,
        grid.ny,
        grid.nx,
    )

    normalized_models: dict[str, np.ndarray] = {}

    for model_name, model in {
        "true": true_model,
        "recovered": recovered_model,
    }.items():
        normalized_model = np.ascontiguousarray(
            np.asarray(
                model,
                dtype=np.float32,
            )
        )

        if normalized_model.shape != expected_shape:
            raise ValueError(
                f"{case_name}: expected {model_name} model shape "
                f"{expected_shape}, but received "
                f"{normalized_model.shape}."
            )

        if not np.all(np.isfinite(normalized_model)):
            raise ValueError(
                f"{case_name}: {model_name} model contains "
                "NaN or infinite values."
            )

        normalized_models[model_name] = normalized_model

    true_model = normalized_models["true"]
    recovered_model = normalized_models["recovered"]

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Positive residual values indicate overprediction by the CNN.
    # Negative values indicate underprediction.
    residual = recovered_model - true_model

    true_xy = np.max(true_model, axis=0)
    recovered_xy = np.max(recovered_model, axis=0)
    residual_xy = recovered_xy - true_xy

    true_xz = np.max(true_model, axis=2)
    recovered_xz = np.max(recovered_model, axis=2)
    residual_xz = recovered_xz - true_xz

    true_yz = np.max(true_model, axis=1)
    recovered_yz = np.max(recovered_model, axis=1)
    residual_yz = recovered_yz - true_yz

    slice_description = "Maximum-value projections"

    density_limit = float(
        max(
            np.max(np.abs(true_model)),
            np.max(np.abs(recovered_model)),
        )
    )

    if np.isclose(density_limit, 0.0):
        density_limit = 1.0

    residual_limit = float(
        max(
            np.max(np.abs(residual_xy)),
            np.max(np.abs(residual_xz)),
            np.max(np.abs(residual_yz)),
        )
    )

    if np.isclose(residual_limit, 0.0):
        residual_limit = 1.0

    xy_extent = (
        grid.x_min,
        grid.x_max,
        grid.y_min,
        grid.y_max,
    )

    xz_extent = (
        grid.x_min,
        grid.x_max,
        grid.z_max,
        grid.z_min,
    )

    yz_extent = (
        grid.y_min,
        grid.y_max,
        grid.z_max,
        grid.z_min,
    )

    density_images = [
        (true_xy, recovered_xy),
        (true_xz, recovered_xz),
        (true_yz, recovered_yz),
    ]

    residual_images = [
        residual_xy,
        residual_xz,
        residual_yz,
    ]

    extents = [
        xy_extent,
        xz_extent,
        yz_extent,
    ]

    aspects = [
        "equal",
        "auto",
        "auto",
    ]

    axis_labels = [
        ("x", "y"),
        ("x", "Depth"),
        ("y", "Depth"),
    ]

    row_labels = [
        "XY",
        "XZ",
        "YZ",
    ]

    figure, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(12, 12),
        constrained_layout=True,
    )

    density_image = None
    residual_image = None

    for row_index in range(3):
        true_view, recovered_view = density_images[row_index]
        residual_view = residual_images[row_index]

        origin = (
            "lower"
            if row_index == 0
            else "upper"
        )

        density_image = axes[row_index, 0].imshow(
            true_view,
            origin=origin,
            extent=extents[row_index],
            aspect=aspects[row_index],
            cmap="viridis",
            vmin=0,
            vmax=density_limit,
        )

        axes[row_index, 1].imshow(
            recovered_view,
            origin=origin,
            extent=extents[row_index],
            aspect=aspects[row_index],
            cmap="viridis",
            vmin=0,
            vmax=density_limit,
        )

        residual_image = axes[row_index, 2].imshow(
            residual_view,
            origin=origin,
            extent=extents[row_index],
            aspect=aspects[row_index],
            cmap="RdBu_r",
            vmin=-residual_limit,
            vmax=residual_limit,
        )

        x_label, y_label = axis_labels[row_index]

        for column_index in range(3):
            axes[row_index, column_index].set_xlabel(
                x_label
            )
            axes[row_index, column_index].set_ylabel(
                y_label
            )

        axes[row_index, 0].text(
            -0.18,
            0.5,
            row_labels[row_index],
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            verticalalignment="center",
            horizontalalignment="center",
            fontweight="bold",
        )

    axes[0, 0].set_title("True")
    axes[0, 1].set_title("Recovered")
    axes[0, 2].set_title("Residual")

    if density_image is None or residual_image is None:
        plt.close(figure)
        raise RuntimeError(
            "Published-example comparison images were not created."
        )

    figure.colorbar(
        density_image,
        ax=axes[:, :2],
        shrink=0.85,
        label="Density contrast",
    )

    figure.colorbar(
        residual_image,
        ax=axes[:, 2],
        shrink=0.85,
        label="Recovered - true",
    )

    figure.suptitle(
        f"{case_name}\n{slice_description}",
        fontsize=14,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)
