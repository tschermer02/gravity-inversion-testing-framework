"""Tests for the analysis-only canonical experiment comparison."""

from __future__ import annotations

import numpy as np
import pytest

from cnn_inversion_3d.final_experiment_comparison import (
    extract_occupied_geometry,
    validate_sample_alignment,
)
from cnn_inversion_3d.single_plane_review import SinglePlaneReviewConfig


def test_extract_occupied_geometry_uses_zyx_and_physical_edges() -> None:
    """Verify top/bottom and width conversion on the canonical grid."""

    config = SinglePlaneReviewConfig()
    density = np.zeros(config.density_shape, dtype=np.float32)
    density[2:6, 8:20, 10:25] = 0.5
    geometry = extract_occupied_geometry(density, config=config)
    assert geometry == {
        "predicted_top_depth_m": 20.0,
        "predicted_bottom_depth_m": 60.0,
        "predicted_thickness_m": 40.0,
        "predicted_width_x_m": 150.0,
        "predicted_width_y_m": 120.0,
    }


def test_extract_empty_geometry_returns_nan() -> None:
    """Verify empty predictions are reportable rather than fatal."""

    config = SinglePlaneReviewConfig()
    result = extract_occupied_geometry(np.zeros(config.density_shape))
    assert all(np.isnan(value) for value in result.values())


def test_paired_alignment_accepts_same_ids_in_different_order() -> None:
    """Verify canonical manifest order controls the paired output."""

    ids = [f"sample_{index:06d}" for index in range(100)]
    result = validate_sample_alignment(
        ids, {"E05": reversed(ids), "E06": ids, "E07": ids[::-1]}
    )
    assert result == ids


def test_paired_alignment_fails_loudly_on_mismatch() -> None:
    """Verify a single mismatched ID stops paired analysis."""

    ids = [f"sample_{index:06d}" for index in range(100)]
    wrong = [*ids[:-1], "sample_999999"]
    with pytest.raises(ValueError, match="alignment differs"):
        validate_sample_alignment(
            ids, {"E05": ids, "E06": wrong, "E07": ids}
        )

