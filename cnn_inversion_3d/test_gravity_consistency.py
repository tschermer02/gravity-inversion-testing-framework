from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cnn_inversion_3d.gravity_consistency import (
    CNNForwardModelContext,
    compute_gravity_consistency_metrics,
    evaluate_gravity_consistency,
    evaluate_prediction_directory,
    gravity_plot_limits,
    plot_cnn_gravity_comparison,
    remove_final_singleton_channel,
)
from forward_modeling.matlab_fwd3d.forward_model import (
    FWD3DGravityForwardModel,
)
from forward_modeling.matlab_fwd3d.grid import (
    ModelGrid,
)
from forward_modeling.matlab_fwd3d.receivers import (
    ReceiverGrid,
)


class SumForwardOperator:
    """Small deterministic forward operator for consistency tests."""

    input_shape = (
        2,
        2,
        2,
    )
    output_shape = (
        3,
        2,
        2,
    )

    def calculate(
        self,
        model,
    ) -> np.ndarray:
        """Return a deterministic gravity volume."""

        density = np.asarray(
            model,
            dtype=np.float64,
        )
        horizontal_sum = np.sum(
            density,
            axis=0,
        )
        return np.stack(
            [
                horizontal_sum,
                0.5 * horizontal_sum,
                0.25 * horizontal_sum,
            ],
            axis=0,
        )


def test_identical_gravity_metrics() -> None:
    """Verify zero residual metrics for identical nonconstant volumes."""

    gravity = np.arange(
        24,
        dtype=np.float64,
    ).reshape(
        2,
        3,
        4,
    )
    metrics = compute_gravity_consistency_metrics(
        gravity,
        gravity,
    )

    assert metrics[
        "gravity_mse"
    ] == pytest.approx(0.0)
    assert metrics[
        "gravity_rmse"
    ] == pytest.approx(0.0)
    assert metrics[
        "gravity_mae"
    ] == pytest.approx(0.0)
    assert metrics[
        "gravity_relative_l2"
    ] == pytest.approx(0.0)
    assert metrics[
        "gravity_correlation"
    ] == pytest.approx(1.0)


def test_known_gravity_metrics() -> None:
    """Verify metrics for a known residual."""

    true_gravity = np.asarray(
        [[[1.0, 2.0]]]
    )
    recovered_gravity = np.asarray(
        [[[2.0, 4.0]]]
    )
    metrics = compute_gravity_consistency_metrics(
        true_gravity,
        recovered_gravity,
    )

    assert metrics[
        "gravity_mse"
    ] == pytest.approx(2.5)
    assert metrics[
        "gravity_rmse"
    ] == pytest.approx(
        np.sqrt(2.5)
    )
    assert metrics[
        "gravity_mae"
    ] == pytest.approx(1.5)
    assert metrics[
        "gravity_relative_l2"
    ] == pytest.approx(1.0)
    assert metrics[
        "gravity_max_abs_residual"
    ] == pytest.approx(2.0)
    assert metrics[
        "gravity_mean_residual"
    ] == pytest.approx(1.5)


def test_constant_gravity_correlation_is_nan() -> None:
    """Verify documented undefined correlation for constant arrays."""

    metrics = compute_gravity_consistency_metrics(
        np.ones(
            (2, 2, 2)
        ),
        np.ones(
            (2, 2, 2)
        ),
    )

    assert np.isnan(
        metrics[
            "gravity_correlation"
        ]
    )


@pytest.mark.parametrize(
    "receiver_count",
    [
        1,
        3,
        5,
    ],
)
def test_metrics_support_multiple_receiver_counts(
    receiver_count: int,
) -> None:
    """Verify per-receiver metrics do not assume eight levels."""

    true_gravity = np.arange(
        receiver_count * 4,
        dtype=np.float64,
    ).reshape(
        receiver_count,
        2,
        2,
    )
    recovered_gravity = (
        true_gravity
        + 1.0
    )
    metrics = compute_gravity_consistency_metrics(
        true_gravity,
        recovered_gravity,
    )

    assert (
        f"gravity_receiver_{receiver_count - 1:02d}_rmse"
        in metrics
    )


@pytest.mark.parametrize(
    ("shape", "expected_shape"),
    [
        (
            (24, 64, 64),
            (24, 64, 64),
        ),
        (
            (24, 64, 64, 1),
            (24, 64, 64),
        ),
        (
            (8, 64, 64),
            (8, 64, 64),
        ),
        (
            (8, 64, 64, 1),
            (8, 64, 64),
        ),
    ],
)
def test_expected_singleton_channel_shapes(
    shape: tuple[int, ...],
    expected_shape: tuple[int, int, int],
) -> None:
    """Verify accepted density and gravity channel layouts."""

    result = remove_final_singleton_channel(
        np.zeros(
            shape,
            dtype=np.float32,
        ),
        name="volume",
        expected_shape=expected_shape,
    )

    assert result.shape == expected_shape


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2),
        (2, 2, 2, 2),
        (1, 2, 2, 2, 1),
    ],
)
def test_invalid_extra_dimensions_raise(
    shape: tuple[int, ...],
) -> None:
    """Verify that arbitrary squeezing is not performed."""

    with pytest.raises(
        ValueError,
        match="must have shape",
    ):
        remove_final_singleton_channel(
            np.zeros(shape),
            name="volume",
            expected_shape=(
                2,
                2,
                2,
            ),
        )


def test_mismatched_gravity_shapes_raise() -> None:
    """Verify clear rejection of mismatched receiver counts."""

    with pytest.raises(
        ValueError,
        match="must match",
    ):
        compute_gravity_consistency_metrics(
            np.zeros(
                (2, 2, 2)
            ),
            np.zeros(
                (3, 2, 2)
            ),
        )


def test_perfect_density_recovery() -> None:
    """Verify zero residual when predicted density is exact."""

    operator = SumForwardOperator()
    density = np.arange(
        8,
        dtype=np.float64,
    ).reshape(
        operator.input_shape
    )
    true_gravity = operator.calculate(
        density
    )
    result = evaluate_gravity_consistency(
        true_gravity,
        density,
        forward_model=operator,
    )

    assert np.allclose(
        result.recovered_gravity,
        true_gravity,
    )
    assert np.allclose(
        result.residual,
        0.0,
    )


def test_zero_density_prediction() -> None:
    """Verify zero recovered gravity and residual equal to negative truth."""

    operator = SumForwardOperator()
    true_density = np.ones(
        operator.input_shape
    )
    true_gravity = operator.calculate(
        true_density
    )
    result = evaluate_gravity_consistency(
        true_gravity,
        np.zeros_like(
            true_density
        ),
        forward_model=operator,
    )

    assert np.allclose(
        result.recovered_gravity,
        0.0,
    )
    assert np.allclose(
        result.residual,
        -true_gravity,
    )


def test_validated_fwd3d_perfect_recovery() -> None:
    """Verify the CNN adapter with the real FWD3D solver."""

    model_grid = ModelGrid(
        x_centers=np.asarray(
            [5.0, 15.0]
        ),
        y_centers=np.asarray(
            [5.0, 15.0]
        ),
        z_centers=np.asarray(
            [5.0, 15.0]
        ),
        dx=10.0,
        dy=10.0,
        dz=np.asarray(
            [10.0, 10.0]
        ),
    )
    receiver_grid = ReceiverGrid(
        x=np.asarray(
            [5.0, 15.0]
        ),
        y=np.asarray(
            [5.0, 15.0]
        ),
        z=np.asarray(
            [-10.0, -20.0]
        ),
    )
    forward_model = FWD3DGravityForwardModel(
        model_grid=model_grid,
        receiver_grid=receiver_grid,
        channel=4,
        receiver_chunk_size=4,
    )
    density = np.zeros(
        forward_model.input_shape
    )
    density[
        0,
        0,
        0,
    ] = 0.5
    true_gravity = forward_model.calculate(
        density
    )
    result = evaluate_gravity_consistency(
        true_gravity,
        density,
        forward_model=forward_model,
    )

    assert np.allclose(
        result.recovered_gravity,
        true_gravity,
    )
    assert np.allclose(
        result.residual,
        0.0,
    )


def test_plot_limits_are_shared_and_symmetric() -> None:
    """Verify common data limits and residual magnitude."""

    true_gravity = np.asarray(
        [[[0.0, 2.0]]]
    )
    recovered_gravity = np.asarray(
        [[[-1.0, 4.0]]]
    )
    minimum, maximum, residual_limit = (
        gravity_plot_limits(
            true_gravity,
            recovered_gravity,
        )
    )

    assert minimum == pytest.approx(
        -1.0
    )
    assert maximum == pytest.approx(
        4.0
    )
    assert residual_limit == pytest.approx(
        2.0
    )


@pytest.mark.parametrize(
    "receiver_count",
    [
        1,
        3,
    ],
)
def test_gravity_comparison_plot_created(
    tmp_path: Path,
    receiver_count: int,
) -> None:
    """Verify plot creation for one and multiple receiver levels."""

    true_gravity = np.arange(
        receiver_count * 16,
        dtype=np.float64,
    ).reshape(
        receiver_count,
        4,
        4,
    )
    recovered_gravity = (
        true_gravity
        + 0.5
    )
    output_path = (
        tmp_path
        / f"gravity_{receiver_count}.png"
    )

    plot_cnn_gravity_comparison(
        true_gravity,
        recovered_gravity,
        np.arange(
            receiver_count,
            dtype=np.float64,
        ),
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_prediction_directory_outputs_and_cache(
    tmp_path: Path,
) -> None:
    """Verify per-sample outputs and default cache skipping."""

    operator = SumForwardOperator()
    density = np.ones(
        operator.input_shape,
        dtype=np.float32,
    )
    true_gravity = operator.calculate(
        density
    ).astype(
        np.float32
    )
    prediction_path = (
        tmp_path
        / "sample_000001_prediction.npz"
    )
    np.savez_compressed(
        prediction_path,
        gravity=true_gravity,
        predicted_density=density,
    )
    context = CNNForwardModelContext(
        forward_model=operator,
        receiver_levels=np.asarray(
            [-10.0, -20.0, -30.0]
        ),
        x_coordinates=np.asarray(
            [0.0, 10.0]
        ),
        y_coordinates=np.asarray(
            [0.0, 10.0]
        ),
        gravity_unit="mGal",
    )
    metric_rows = [
        {
            "sample_path": (
                "samples/sample_000001.npz"
            ),
            "prediction_path": (
                prediction_path.name
            ),
        }
    ]

    first = evaluate_prediction_directory(
        prediction_directory=tmp_path,
        metric_rows=metric_rows,
        context=context,
        selected_receiver_indices=(
            0,
            2,
        ),
        save_gravity_volumes=True,
        overwrite=False,
    )
    second = evaluate_prediction_directory(
        prediction_directory=tmp_path,
        metric_rows=metric_rows,
        context=context,
        selected_receiver_indices=(
            0,
            2,
        ),
        save_gravity_volumes=True,
        overwrite=False,
    )
    sample_output = (
        tmp_path
        / "gravity_consistency"
        / "sample_000001"
    )

    assert first.completed == 1
    assert first.skipped == 0
    assert first.failed == 0
    assert second.completed == 0
    assert second.skipped == 1
    assert second.failed == 0
    assert (
        sample_output
        / "gravity_consistency_metrics.json"
    ).exists()
    assert (
        sample_output
        / "gravity_comparison.png"
    ).exists()
    assert (
        sample_output
        / "true_gravity.npy"
    ).exists()
    assert (
        sample_output
        / "recovered_gravity.npy"
    ).exists()
    assert (
        sample_output
        / "gravity_residual.npy"
    ).exists()
