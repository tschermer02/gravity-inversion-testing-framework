from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from forward_modeling.matlab_fwd3d.example_config import (
    MATLAB_EXAMPLE_CHANNELS,
    build_matlab_example_density,
    build_matlab_example_grid,
    build_matlab_example_receivers,
)
from forward_modeling.matlab_fwd3d.gravity import (
    calculate_gravity_table,
)
from forward_modeling.matlab_fwd3d.io import (
    load_matlab_gravity_table,
    save_gravity_table,
    sort_gravity_table,
)


FloatArray = npt.NDArray[np.float64]


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    default_reference = (
        Path(__file__).resolve().parent
        / "reference_data"
        / "obsData1.dat"
    )

    default_output = (
        Path(__file__).resolve().parent
        / "validation_output"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Validate the Python FWD3D gravity implementation "
            "against MATLAB obsData1.dat."
        )
    )

    parser.add_argument(
        "--matlab-output",
        type=Path,
        default=default_reference,
        help=(
            "Path to MATLAB obsData1.dat. "
            f"Default: {default_reference}"
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=default_output,
        help=(
            "Directory used for validation tables and figures. "
            f"Default: {default_output}"
        ),
    )

    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=1.0e-6,
        help="Absolute validation tolerance.",
    )

    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=1.0e-6,
        help="Relative validation tolerance.",
    )

    return parser


def _validate_coordinate_columns(
    matlab_table: FloatArray,
    python_table: FloatArray,
) -> None:
    """Confirm that both tables describe the same observations."""

    if matlab_table.shape != python_table.shape:
        raise ValueError(
            "MATLAB and Python tables have different shapes: "
            f"{matlab_table.shape} and {python_table.shape}."
        )

    coordinate_match = np.allclose(
        matlab_table[:, :3],
        python_table[:, :3],
        rtol=0.0,
        atol=1.0e-10,
    )

    channel_match = np.array_equal(
        matlab_table[:, 3].astype(np.int64),
        python_table[:, 3].astype(np.int64),
    )

    if not coordinate_match:
        raise ValueError(
            "MATLAB and Python receiver coordinates do not match."
        )

    if not channel_match:
        raise ValueError(
            "MATLAB and Python channel columns do not match."
        )


def _calculate_error_summary(
    matlab_values: FloatArray,
    python_values: FloatArray,
) -> dict[str, float]:
    """Calculate numerical validation metrics."""

    differences = (
        python_values
        - matlab_values
    )

    absolute_errors = np.abs(
        differences
    )

    nonzero_reference = (
        np.abs(matlab_values)
        > np.finfo(np.float64).eps
    )

    relative_errors = np.zeros_like(
        absolute_errors
    )

    relative_errors[nonzero_reference] = (
        absolute_errors[nonzero_reference]
        / np.abs(
            matlab_values[nonzero_reference]
        )
    )

    correlation = float(
        np.corrcoef(
            matlab_values,
            python_values,
        )[0, 1]
    )

    return {
        "maximum_absolute_error": float(
            np.max(absolute_errors)
        ),
        "mean_absolute_error": float(
            np.mean(absolute_errors)
        ),
        "root_mean_squared_error": float(
            np.sqrt(
                np.mean(
                    differences**2
                )
            )
        ),
        "maximum_relative_error": float(
            np.max(relative_errors)
        ),
        "mean_relative_error": float(
            np.mean(relative_errors)
        ),
        "correlation": correlation,
    }


def _reshape_single_plane(
    table: FloatArray,
    *,
    channel: int,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
]:
    """
    Reshape one channel and receiver elevation into a Y-X map.
    """

    channel_rows = table[
        table[:, 3].astype(np.int64)
        == channel
    ]

    unique_z = np.unique(
        channel_rows[:, 2]
    )

    if unique_z.size != 1:
        raise ValueError(
            "Validation plotting expects one receiver Z level."
        )

    x_coordinates = np.unique(
        channel_rows[:, 0]
    )

    y_coordinates = np.unique(
        channel_rows[:, 1]
    )

    value_map = np.empty(
        (
            y_coordinates.size,
            x_coordinates.size,
        ),
        dtype=np.float64,
    )

    x_indices = {
        value: index
        for index, value in enumerate(
            x_coordinates
        )
    }

    y_indices = {
        value: index
        for index, value in enumerate(
            y_coordinates
        )
    }

    for row in channel_rows:
        x_value = row[0]
        y_value = row[1]

        value_map[
            y_indices[y_value],
            x_indices[x_value],
        ] = row[4]

    return (
        x_coordinates,
        y_coordinates,
        value_map,
    )


def _save_map(
    *,
    values: FloatArray,
    x: FloatArray,
    y: FloatArray,
    title: str,
    colorbar_label: str,
    output_path: Path,
) -> None:
    """Save one gravity validation map."""

    figure, axis = plt.subplots(
        figsize=(7.0, 6.0)
    )

    image = axis.imshow(
        values,
        origin="lower",
        extent=(
            float(x.min()),
            float(x.max()),
            float(y.min()),
            float(y.max()),
        ),
        aspect="equal",
    )

    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_title(title)

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        colorbar_label
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def _create_validation_figures(
    *,
    matlab_table: FloatArray,
    python_table: FloatArray,
    output_directory: Path,
) -> None:
    """Create MATLAB, Python, and difference maps for Gz."""

    x, y, matlab_map = _reshape_single_plane(
        matlab_table,
        channel=4,
    )

    _, _, python_map = _reshape_single_plane(
        python_table,
        channel=4,
    )

    difference_map = (
        python_map
        - matlab_map
    )

    _save_map(
        values=matlab_map,
        x=x,
        y=y,
        title="MATLAB Gz",
        colorbar_label="mGal",
        output_path=(
            output_directory
            / "matlab_gz.png"
        ),
    )

    _save_map(
        values=python_map,
        x=x,
        y=y,
        title="Python Gz",
        colorbar_label="mGal",
        output_path=(
            output_directory
            / "python_gz.png"
        ),
    )

    _save_map(
        values=difference_map,
        x=x,
        y=y,
        title="Python minus MATLAB Gz",
        colorbar_label="mGal",
        output_path=(
            output_directory
            / "gz_difference.png"
        ),
    )


def main() -> None:
    """Run the MATLAB-to-Python gravity validation."""

    parser = _build_argument_parser()
    arguments = parser.parse_args()

    output_directory = (
        arguments.output_directory
        .expanduser()
        .resolve()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_grid = (
        build_matlab_example_grid()
    )

    density_model = (
        build_matlab_example_density(
            model_grid
        )
    )

    receiver_grid = (
        build_matlab_example_receivers()
    )

    python_table = calculate_gravity_table(
        model=density_model,
        model_grid=model_grid,
        receiver_points=receiver_grid.points(),
        channels=MATLAB_EXAMPLE_CHANNELS,
    )

    matlab_table = load_matlab_gravity_table(
        arguments.matlab_output
    )

    python_table = sort_gravity_table(
        python_table
    )

    matlab_table = sort_gravity_table(
        matlab_table
    )

    _validate_coordinate_columns(
        matlab_table=matlab_table,
        python_table=python_table,
    )

    matlab_values = matlab_table[:, 4]
    python_values = python_table[:, 4]

    summary = _calculate_error_summary(
        matlab_values=matlab_values,
        python_values=python_values,
    )

    passed = np.allclose(
        python_values,
        matlab_values,
        rtol=arguments.relative_tolerance,
        atol=arguments.absolute_tolerance,
    )

    python_output_path = save_gravity_table(
        output_directory
        / "python_obsData1.dat",
        python_table,
    )

    difference_table = np.column_stack(
        (
            matlab_table[:, :4],
            python_values
            - matlab_values,
        )
    )

    difference_output_path = (
        save_gravity_table(
            output_directory
            / "python_minus_matlab.dat",
            difference_table,
        )
    )

    _create_validation_figures(
        matlab_table=matlab_table,
        python_table=python_table,
        output_directory=output_directory,
    )

    print()
    print("MATLAB FWD3D gravity validation")
    print("=" * 38)
    print(
        f"Number of rows: "
        f"{matlab_table.shape[0]}"
    )
    print(
        f"Number of model cells: "
        f"{model_grid.number_of_cells}"
    )
    print(
        f"Model shape: "
        f"{model_grid.model_shape}"
    )
    print(
        f"Receiver volume shape: "
        f"{receiver_grid.volume_shape}"
    )
    print()

    for name, value in summary.items():
        print(
            f"{name}: {value:.12e}"
        )

    print()
    print(
        "Tolerance check: "
        + (
            "PASSED"
            if passed
            else "FAILED"
        )
    )
    print()
    print(
        f"Python table: {python_output_path}"
    )
    print(
        f"Difference table: "
        f"{difference_output_path}"
    )
    print(
        f"Figures: {output_directory}"
    )

    if not passed:
        raise SystemExit(
            "Python results did not match MATLAB "
            "within the requested tolerances."
        )


if __name__ == "__main__":
    main()