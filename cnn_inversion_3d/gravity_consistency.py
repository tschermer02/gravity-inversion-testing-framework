from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

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


class GravityForwardOperator(Protocol):
    """Interface required for CNN gravity-consistency evaluation."""

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Return the density shape in ``(z, y, x)`` order."""

    @property
    def output_shape(self) -> tuple[int, int, int]:
        """Return gravity shape in ``(receiver_z, y, x)`` order."""

    def calculate(
        self,
        model: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Forward model one density volume."""


@dataclass(frozen=True)
class GravityConsistencyResult:
    """Results from forward modeling a recovered density volume."""

    true_gravity: FloatArray
    recovered_gravity: FloatArray
    residual: FloatArray
    metrics: dict[str, float]


@dataclass(frozen=True)
class CNNForwardModelContext:
    """Physical configuration reconstructed from dataset metadata."""

    forward_model: GravityForwardOperator
    receiver_levels: FloatArray
    x_coordinates: FloatArray
    y_coordinates: FloatArray
    gravity_unit: str


@dataclass(frozen=True)
class SinglePlaneForwardOperator:
    """Expose one FWD3D receiver level as a two-dimensional operator."""

    forward_model: FWD3DGravityForwardModel

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Return the density input shape."""

        return self.forward_model.input_shape

    @property
    def output_shape(self) -> tuple[int, int, int]:
        """Return one receiver level followed by the surface grid."""

        return self.forward_model.output_shape

    def calculate(self, model: npt.ArrayLike) -> FloatArray:
        """Calculate and remove only the singleton receiver-level axis."""

        return np.asarray(self.forward_model.calculate(model))


@dataclass(frozen=True)
class GravityConsistencyBatchResult:
    """Summary of a prediction-directory consistency evaluation."""

    rows: list[dict[str, Any]]
    completed: int
    skipped: int
    failed: int


def remove_final_singleton_channel(
    array: npt.ArrayLike,
    *,
    name: str,
    expected_shape: tuple[int, int, int],
) -> FloatArray:
    """
    Validate a volume and remove only its final singleton channel.

    Parameters
    ----------
    array
        Three-dimensional volume or volume with one final channel.
    name
        Array name used in validation errors.
    expected_shape
        Required three-dimensional shape.

    Returns
    -------
    numpy.ndarray
        Finite float64 volume with ``expected_shape``.
    """

    values = np.asarray(
        array,
        dtype=np.float64,
    )

    if values.shape == (
        *expected_shape,
        1,
    ):
        values = values[
            ...,
            0,
        ]

    if (
        len(expected_shape) == 3
        and expected_shape[0] == 1
        and values.shape == expected_shape[1:]
    ):
        values = values[np.newaxis, ...]

    if values.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape} or "
            f"{(*expected_shape, 1)}, received {values.shape}."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return values


def gravity_correlation(
    true_gravity: npt.ArrayLike,
    recovered_gravity: npt.ArrayLike,
) -> float:
    """
    Calculate Pearson correlation with safe constant-array handling.

    Correlation is returned as ``NaN`` when either input is constant,
    because Pearson correlation is then mathematically undefined.
    """

    true_values = np.asarray(
        true_gravity,
        dtype=np.float64,
    ).ravel()
    recovered_values = np.asarray(
        recovered_gravity,
        dtype=np.float64,
    ).ravel()

    if (
        np.isclose(
            np.std(true_values),
            0.0,
        )
        or np.isclose(
            np.std(recovered_values),
            0.0,
        )
    ):
        return float("nan")

    return float(
        np.corrcoef(
            true_values,
            recovered_values,
        )[0, 1]
    )


def compute_gravity_consistency_metrics(
    true_gravity: npt.ArrayLike,
    recovered_gravity: npt.ArrayLike,
    *,
    epsilon: float = 1.0e-12,
) -> dict[str, float]:
    """
    Compute data-space agreement metrics for two gravity volumes.

    The residual convention is always:

    ``recovered_gravity - true_gravity``.

    Parameters
    ----------
    true_gravity, recovered_gravity
        Matching volumes in ``(receiver_z, y, x)`` order.
    epsilon
        Positive lower bound for relative-L2 denominators.

    Returns
    -------
    dict
        Global and per-receiver gravity metrics.
    """

    if epsilon <= 0.0:
        raise ValueError(
            "epsilon must be greater than zero."
        )

    true_values = np.asarray(
        true_gravity,
        dtype=np.float64,
    )
    recovered_values = np.asarray(
        recovered_gravity,
        dtype=np.float64,
    )

    if true_values.ndim == 2:
        true_values = true_values[np.newaxis, ...]
        recovered_values = recovered_values[np.newaxis, ...]
    elif true_values.ndim != 3:
        raise ValueError(
            "true_gravity must be two- or three-dimensional, received "
            f"{true_values.shape}."
        )

    if recovered_values.shape != true_values.shape:
        raise ValueError(
            "recovered_gravity must match true_gravity shape "
            f"{true_values.shape}, received {recovered_values.shape}."
        )

    if not (
        np.all(
            np.isfinite(true_values)
        )
        and np.all(
            np.isfinite(recovered_values)
        )
    ):
        raise ValueError(
            "Gravity volumes must contain only finite values."
        )

    residual = (
        recovered_values
        - true_values
    )
    gravity_mse = float(
        np.mean(
            residual**2
        )
    )
    true_norm = float(
        np.linalg.norm(
            true_values.ravel()
        )
    )

    metrics = {
        "gravity_mse": gravity_mse,
        "gravity_rmse": float(
            np.sqrt(
                gravity_mse
            )
        ),
        "gravity_mae": float(
            np.mean(
                np.abs(residual)
            )
        ),
        "gravity_relative_l2": float(
            np.linalg.norm(
                residual.ravel()
            )
            / max(
                true_norm,
                epsilon,
            )
        ),
        "gravity_correlation": gravity_correlation(
            true_values,
            recovered_values,
        ),
        "gravity_max_abs_residual": float(
            np.max(
                np.abs(residual)
            )
        ),
        "gravity_mean_residual": float(
            np.mean(residual)
        ),
        "gravity_residual_std": float(
            np.std(residual)
        ),
    }

    for receiver_index in range(
        true_values.shape[0]
    ):
        true_receiver = true_values[
            receiver_index
        ]
        recovered_receiver = recovered_values[
            receiver_index
        ]
        receiver_residual = (
            recovered_receiver
            - true_receiver
        )
        receiver_mse = float(
            np.mean(
                receiver_residual**2
            )
        )
        receiver_prefix = (
            f"gravity_receiver_{receiver_index:02d}"
        )

        metrics[
            f"{receiver_prefix}_rmse"
        ] = float(
            np.sqrt(
                receiver_mse
            )
        )
        metrics[
            f"{receiver_prefix}_relative_l2"
        ] = float(
            np.linalg.norm(
                receiver_residual.ravel()
            )
            / max(
                float(
                    np.linalg.norm(
                        true_receiver.ravel()
                    )
                ),
                epsilon,
            )
        )
        metrics[
            f"{receiver_prefix}_correlation"
        ] = gravity_correlation(
            true_receiver,
            recovered_receiver,
        )

    return metrics


def forward_model_cnn_density(
    density: npt.ArrayLike,
    *,
    forward_model: GravityForwardOperator,
) -> FloatArray:
    """
    Forward model a CNN density prediction with the configured solver.

    Parameters
    ----------
    density
        Predicted density in ``(z, y, x)`` order, optionally with one
        final singleton channel.
    forward_model
        Existing validated forward operator.

    Returns
    -------
    numpy.ndarray
        Recovered gravity in ``(receiver_z, y, x)`` order.
    """

    density_volume = remove_final_singleton_channel(
        density,
        name="predicted_density",
        expected_shape=forward_model.input_shape,
    )
    recovered_gravity = np.asarray(
        forward_model.calculate(
            density_volume
        ),
        dtype=np.float64,
    )

    if recovered_gravity.shape != forward_model.output_shape:
        raise RuntimeError(
            "Forward model returned shape "
            f"{recovered_gravity.shape}; expected "
            f"{forward_model.output_shape}."
        )

    return recovered_gravity


def evaluate_gravity_consistency(
    true_gravity: npt.ArrayLike,
    predicted_density: npt.ArrayLike,
    *,
    forward_model: GravityForwardOperator,
    epsilon: float = 1.0e-12,
) -> GravityConsistencyResult:
    """
    Forward model recovered density and compare gravity volumes.

    Parameters
    ----------
    true_gravity
        Original synthetic gravity volume, optionally with one final
        singleton channel.
    predicted_density
        CNN-recovered density volume, optionally with one final singleton
        channel.
    forward_model
        Existing validated forward operator.
    epsilon
        Positive lower bound for relative-L2 denominators.

    Returns
    -------
    GravityConsistencyResult
        True gravity, recovered gravity, residual, and metrics.
    """

    true_volume = remove_final_singleton_channel(
        true_gravity,
        name="true_gravity",
        expected_shape=forward_model.output_shape,
    )
    recovered_volume = forward_model_cnn_density(
        predicted_density,
        forward_model=forward_model,
    )
    residual = (
        recovered_volume
        - true_volume
    )
    metrics = compute_gravity_consistency_metrics(
        true_volume,
        recovered_volume,
        epsilon=epsilon,
    )

    return GravityConsistencyResult(
        true_gravity=true_volume,
        recovered_gravity=recovered_volume,
        residual=residual,
        metrics=metrics,
    )


def build_cnn_forward_model_context(
    metadata_path: Path,
) -> CNNForwardModelContext:
    """
    Reconstruct the dataset's exact FWD3D physical configuration.

    Parameters
    ----------
    metadata_path
        Dataset-level metadata JSON produced during generation.

    Returns
    -------
    CNNForwardModelContext
        Validated solver, receiver levels, coordinates, and units.
    """

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Dataset metadata is required for gravity consistency:\n"
            f"{metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as metadata_file:
        metadata = json.load(
            metadata_file
        )

    if metadata.get(
        "density_array_order"
    ) != "density[z, y, x]":
        raise ValueError(
            "Unsupported or missing density array order in metadata."
        )

    gravity_array_order = metadata.get("gravity_array_order")
    if gravity_array_order not in {
        "gravity[z_receiver, y_receiver, x_receiver]",
        "gravity[y, x]",
    }:
        raise ValueError(
            "Unsupported or missing gravity array order in metadata."
        )

    gravity_channel = int(
        metadata.get(
            "gravity_channel",
            -1,
        )
    )
    gravity_unit = str(
        metadata.get(
            "gravity_unit",
            "",
        )
    )

    if gravity_channel != 4 or gravity_unit != "mGal":
        raise ValueError(
            "CNN gravity consistency currently requires FWD3D Gz "
            "channel 4 in mGal, matching dataset generation."
        )

    grid_values = metadata.get(
        "grid"
    )
    configuration = metadata.get(
        "configuration"
    )

    if not isinstance(
        grid_values,
        dict,
    ) or not isinstance(
        configuration,
        dict,
    ):
        raise ValueError(
            "Dataset metadata lacks grid or generation configuration."
        )

    receiver_values = configuration.get("receivers")

    if gravity_array_order == "gravity[y, x]":
        receiver_values = {
            "number_of_levels": 1,
            "first_level_z": metadata["observation_z_m"],
            "level_spacing": 1.0,
        }

    if not isinstance(
        receiver_values,
        dict,
    ):
        raise ValueError(
            "Dataset metadata lacks receiver configuration."
        )

    grid = MatlabCompatibleGridSpec(
        nx=int(
            grid_values["nx"]
        ),
        ny=int(
            grid_values["ny"]
        ),
        nz=int(
            grid_values["nz"]
        ),
        x_min=float(
            grid_values["x_min"]
        ),
        x_max=float(
            grid_values["x_max"]
        ),
        y_min=float(
            grid_values["y_min"]
        ),
        y_max=float(
            grid_values["y_max"]
        ),
        z_min=float(
            grid_values["z_min"]
        ),
        z_max=float(
            grid_values["z_max"]
        ),
    )

    for spacing_name in (
        "dx",
        "dy",
        "dz",
    ):
        if not np.isclose(
            getattr(
                grid,
                spacing_name,
            ),
            float(
                grid_values[
                    spacing_name
                ]
            ),
        ):
            raise ValueError(
                f"Metadata {spacing_name} is inconsistent with grid bounds."
            )

    number_of_levels = int(
        receiver_values[
            "number_of_levels"
        ]
    )
    first_level_z = float(
        receiver_values[
            "first_level_z"
        ]
    )
    level_spacing = float(
        receiver_values[
            "level_spacing"
        ]
    )
    receiver_levels = (
        first_level_z
        - level_spacing
        * np.arange(
            number_of_levels,
            dtype=np.float64,
        )
    )
    if gravity_array_order == "gravity[y, x]":
        x_coordinates = np.asarray(
            metadata["observation_x_coordinates_m"], dtype=np.float64
        )
        y_coordinates = np.asarray(
            metadata["observation_y_coordinates_m"], dtype=np.float64
        )
    else:
        x_coordinates = np.linspace(
            grid.x_min, grid.x_max, grid.nx, dtype=np.float64
        )
        y_coordinates = np.linspace(
            grid.y_min, grid.y_max, grid.ny, dtype=np.float64
        )
    receiver_grid = ReceiverGrid(
        x=x_coordinates,
        y=y_coordinates,
        z=receiver_levels,
    )
    model_grid = model_grid_from_grid_spec(
        grid
    )
    forward_model = FWD3DGravityForwardModel(
        model_grid=model_grid,
        receiver_grid=receiver_grid,
        channel=gravity_channel,
        receiver_chunk_size=int(
            configuration.get(
                "receiver_chunk_size",
                128,
            )
        ),
    )

    expected_density_shape = tuple(
        int(value)
        for value in metadata[
            "density_shape"
        ]
    )
    expected_gravity_shape = tuple(
        int(value)
        for value in metadata[
            "gravity_shape"
        ]
    )

    if forward_model.input_shape != expected_density_shape:
        raise ValueError(
            "Reconstructed density shape does not match metadata: "
            f"{forward_model.input_shape} != {expected_density_shape}."
        )

    exposed_forward_model: GravityForwardOperator = forward_model
    if gravity_array_order == "gravity[y, x]":
        exposed_forward_model = SinglePlaneForwardOperator(forward_model)
        expected_gravity_shape = (1, *expected_gravity_shape)

    if exposed_forward_model.output_shape != expected_gravity_shape:
        raise ValueError(
            "Reconstructed gravity shape does not match metadata: "
            f"{exposed_forward_model.output_shape} != {expected_gravity_shape}."
        )

    return CNNForwardModelContext(
        forward_model=exposed_forward_model,
        receiver_levels=receiver_levels,
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        gravity_unit=gravity_unit,
    )


def gravity_plot_limits(
    true_gravity: npt.ArrayLike,
    recovered_gravity: npt.ArrayLike,
) -> tuple[float, float, float]:
    """
    Return shared data limits and a symmetric residual magnitude.

    Returns
    -------
    tuple
        ``(data_minimum, data_maximum, residual_limit)``.
    """

    true_values = np.asarray(
        true_gravity,
        dtype=np.float64,
    )
    recovered_values = np.asarray(
        recovered_gravity,
        dtype=np.float64,
    )

    if true_values.shape != recovered_values.shape:
        raise ValueError(
            "Gravity volumes must have matching shapes for plotting."
        )

    residual = (
        recovered_values
        - true_values
    )
    residual_limit = float(
        np.max(
            np.abs(residual)
        )
    )

    if residual_limit == 0.0:
        residual_limit = float(
            np.finfo(
                np.float64
            ).eps
        )

    data_minimum = float(
        min(
            np.min(true_values),
            np.min(recovered_values),
        )
    )
    data_maximum = float(
        max(
            np.max(true_values),
            np.max(recovered_values),
        )
    )

    if data_minimum == data_maximum:
        data_maximum = (
            data_minimum
            + float(
                np.finfo(
                    np.float64
                ).eps
            )
        )

    return (
        data_minimum,
        data_maximum,
        residual_limit,
    )


def plot_cnn_gravity_comparison(
    true_gravity: npt.ArrayLike,
    recovered_gravity: npt.ArrayLike,
    receiver_levels: Sequence[float],
    output_path: Path,
    *,
    selected_indices: Sequence[int] | None = None,
    x_coordinates: Sequence[float] | None = None,
    y_coordinates: Sequence[float] | None = None,
    units: str = "mGal",
) -> None:
    """
    Plot true, recovered, and residual gravity by receiver level.

    Parameters
    ----------
    true_gravity, recovered_gravity
        Matching three-dimensional gravity volumes.
    receiver_levels
        Receiver elevations in first-axis order.
    output_path
        Figure destination.
    selected_indices
        Receiver indices to plot. The shallowest, middle, and deepest
        levels are selected by default.
    x_coordinates, y_coordinates
        Optional physical coordinates in meters.
    units
        Gravity units displayed on colorbars.
    """

    true_values = np.asarray(
        true_gravity,
        dtype=np.float64,
    )
    recovered_values = np.asarray(
        recovered_gravity,
        dtype=np.float64,
    )

    if (
        true_values.ndim != 3
        or recovered_values.shape != true_values.shape
    ):
        raise ValueError(
            "Gravity plot inputs must be matching three-dimensional volumes."
        )

    levels = np.asarray(
        receiver_levels,
        dtype=np.float64,
    )

    if levels.shape != (
        true_values.shape[0],
    ):
        raise ValueError(
            "receiver_levels must contain one value per gravity level."
        )

    if selected_indices is None:
        selected_indices = tuple(
            dict.fromkeys(
                (
                    0,
                    true_values.shape[0] // 2,
                    true_values.shape[0] - 1,
                )
            )
        )

    indices = tuple(
        int(index)
        for index in selected_indices
    )

    if not indices:
        raise ValueError(
            "At least one receiver index must be selected."
        )

    for index in indices:
        if not 0 <= index < true_values.shape[0]:
            raise ValueError(
                f"Receiver index {index} lies outside "
                f"[0, {true_values.shape[0] - 1}]."
            )

    if x_coordinates is None:
        x_values = np.arange(
            true_values.shape[2],
            dtype=np.float64,
        )
        x_label = "X index"
    else:
        x_values = np.asarray(
            x_coordinates,
            dtype=np.float64,
        )
        x_label = "X (m)"

    if y_coordinates is None:
        y_values = np.arange(
            true_values.shape[1],
            dtype=np.float64,
        )
        y_label = "Y index"
    else:
        y_values = np.asarray(
            y_coordinates,
            dtype=np.float64,
        )
        y_label = "Y (m)"

    if x_values.shape != (
        true_values.shape[2],
    ) or y_values.shape != (
        true_values.shape[1],
    ):
        raise ValueError(
            "Coordinate lengths must match gravity X and Y dimensions."
        )

    (
        data_minimum,
        data_maximum,
        residual_limit,
    ) = gravity_plot_limits(
        true_values,
        recovered_values,
    )
    residual = (
        recovered_values
        - true_values
    )
    extent = (
        float(x_values[0]),
        float(x_values[-1]),
        float(y_values[0]),
        float(y_values[-1]),
    )

    figure, axes = plt.subplots(
        nrows=len(indices),
        ncols=3,
        figsize=(
            13.0,
            4.0 * len(indices),
        ),
        squeeze=False,
    )
    data_image = None
    residual_image = None

    for row_index, receiver_index in enumerate(
        indices
    ):
        for column_index, (
            values,
            title,
        ) in enumerate(
            (
                (
                    true_values[
                        receiver_index
                    ],
                    "True Gravity",
                ),
                (
                    recovered_values[
                        receiver_index
                    ],
                    "Recovered Gravity",
                ),
                (
                    residual[
                        receiver_index
                    ],
                    "Gravity Residual",
                ),
            )
        ):
            axis = axes[
                row_index,
                column_index,
            ]
            is_residual = (
                column_index == 2
            )
            image = axis.imshow(
                values,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap=(
                    "RdBu_r"
                    if is_residual
                    else "viridis"
                ),
                vmin=(
                    -residual_limit
                    if is_residual
                    else data_minimum
                ),
                vmax=(
                    residual_limit
                    if is_residual
                    else data_maximum
                ),
            )

            if is_residual:
                residual_image = image
            else:
                data_image = image

            axis.set_title(
                title
            )
            axis.set_xlabel(
                x_label
            )
            axis.set_ylabel(
                y_label
            )

        axes[
            row_index,
            0,
        ].text(
            -0.23,
            0.5,
            f"Receiver z = {levels[receiver_index]:g} m",
            transform=axes[
                row_index,
                0,
            ].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontweight="bold",
        )

    if data_image is None or residual_image is None:
        raise RuntimeError(
            "Gravity comparison figure contains no images."
        )

    figure.colorbar(
        data_image,
        ax=axes[:, :2].ravel().tolist(),
        label=f"Gravity ({units})",
        shrink=0.9,
    )
    figure.colorbar(
        residual_image,
        ax=axes[:, 2].ravel().tolist(),
        label=f"Recovered - true ({units})",
        shrink=0.9,
    )
    figure.suptitle(
        "CNN Gravity Forward-Consistency Evaluation",
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.90,
        bottom=0.07,
        top=0.93,
        wspace=0.28,
        hspace=0.30,
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
    plt.close(
        figure
    )


def summarize_gravity_metric_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Summarize numeric gravity metrics over evaluated samples.

    Returns
    -------
    dict
        Count and descriptive statistics for each gravity metric.
    """

    if not rows:
        return {
            "number_of_samples": 0,
            "metrics": {},
        }

    metric_names = [
        name
        for name in rows[0]
        if name.startswith(
            "gravity_"
        )
        and name != "gravity_consistency_status"
    ]
    summaries: dict[str, Any] = {}

    for name in metric_names:
        values = np.asarray(
            [
                float(
                    row[name]
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        finite_values = values[
            np.isfinite(values)
        ]

        if finite_values.size == 0:
            summaries[name] = {
                "mean": float("nan"),
                "median": float("nan"),
                "standard_deviation": float("nan"),
                "minimum": float("nan"),
                "maximum": float("nan"),
            }
            continue

        summaries[name] = {
            "mean": float(
                np.mean(finite_values)
            ),
            "median": float(
                np.median(finite_values)
            ),
            "standard_deviation": float(
                np.std(finite_values)
            ),
            "minimum": float(
                np.min(finite_values)
            ),
            "maximum": float(
                np.max(finite_values)
            ),
        }

    return {
        "number_of_samples": len(rows),
        "metrics": summaries,
    }


def evaluate_prediction_directory(
    *,
    prediction_directory: Path,
    metric_rows: list[dict[str, Any]],
    context: CNNForwardModelContext,
    selected_receiver_indices: Sequence[int] | None,
    save_gravity_volumes: bool,
    overwrite: bool,
) -> GravityConsistencyBatchResult:
    """
    Evaluate and cache gravity consistency for prediction NPZ files.

    Existing per-sample metrics are loaded and skipped unless ``overwrite``
    is true. Failures are reported and do not prevent other samples from
    being processed.
    """

    output_root = (
        prediction_directory
        / "gravity_consistency"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    rows: list[dict[str, Any]] = []
    completed = 0
    skipped = 0
    failed = 0

    for row_index, metric_row in enumerate(
        metric_rows
    ):
        prediction_name = metric_row.get(
            "prediction_path"
        )

        if not prediction_name:
            failed += 1
            print(
                f"[{row_index + 1:>3}/{len(metric_rows)}] FAILED: "
                "metric row has no prediction_path."
            )
            continue

        prediction_path = (
            prediction_directory
            / str(prediction_name)
        )
        sample_name = prediction_path.stem.removesuffix(
            "_prediction"
        )
        sample_output = (
            output_root
            / sample_name
        )
        metrics_path = (
            sample_output
            / "gravity_consistency_metrics.json"
        )
        figure_path = (
            sample_output
            / "gravity_comparison.png"
        )
        volume_paths = (
            sample_output
            / "true_gravity.npy",
            sample_output
            / "recovered_gravity.npy",
            sample_output
            / "gravity_residual.npy",
        )
        cache_complete = (
            metrics_path.exists()
            and figure_path.exists()
            and (
                not save_gravity_volumes
                or all(
                    path.exists()
                    for path in volume_paths
                )
            )
        )

        if cache_complete and not overwrite:
            with metrics_path.open(
                "r",
                encoding="utf-8",
            ) as metrics_file:
                cached = json.load(
                    metrics_file
                )
            rows.append(
                dict(cached)
            )
            skipped += 1
            print(
                f"[{row_index + 1:>3}/{len(metric_rows)}] "
                f"SKIPPED {sample_name}"
            )
            continue

        try:
            with np.load(
                prediction_path,
                allow_pickle=False,
            ) as prediction_file:
                true_gravity = np.asarray(
                    prediction_file[
                        "gravity"
                    ]
                )
                predicted_density = np.asarray(
                    prediction_file[
                        "predicted_density"
                    ]
                )

            result = evaluate_gravity_consistency(
                true_gravity,
                predicted_density,
                forward_model=(
                    context.forward_model
                ),
            )
            sample_output.mkdir(
                parents=True,
                exist_ok=True,
            )
            result_row: dict[str, Any] = {
                "sample_path": metric_row.get(
                    "sample_path"
                ),
                "prediction_path": str(
                    prediction_name
                ),
                **result.metrics,
            }

            with metrics_path.open(
                "w",
                encoding="utf-8",
            ) as metrics_file:
                json.dump(
                    result_row,
                    metrics_file,
                    indent=2,
                    allow_nan=True,
                )

            if save_gravity_volumes:
                np.save(
                    sample_output
                    / "true_gravity.npy",
                    result.true_gravity,
                )
                np.save(
                    sample_output
                    / "recovered_gravity.npy",
                    result.recovered_gravity,
                )
                np.save(
                    sample_output
                    / "gravity_residual.npy",
                    result.residual,
                )

            plot_cnn_gravity_comparison(
                result.true_gravity,
                result.recovered_gravity,
                context.receiver_levels,
                figure_path,
                selected_indices=(
                    selected_receiver_indices
                ),
                x_coordinates=(
                    context.x_coordinates
                ),
                y_coordinates=(
                    context.y_coordinates
                ),
                units=context.gravity_unit,
            )
            rows.append(
                result_row
            )
            completed += 1
            print(
                f"[{row_index + 1:>3}/{len(metric_rows)}] "
                f"COMPLETED {sample_name}"
            )
        except Exception as error:
            failed += 1
            print(
                f"[{row_index + 1:>3}/{len(metric_rows)}] "
                f"FAILED {sample_name}: {error}"
            )

    if rows:
        csv_path = (
            prediction_directory
            / "gravity_consistency_metrics.csv"
        )
        with csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=list(
                    rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(rows)

    return GravityConsistencyBatchResult(
        rows=rows,
        completed=completed,
        skipped=skipped,
        failed=failed,
    )
