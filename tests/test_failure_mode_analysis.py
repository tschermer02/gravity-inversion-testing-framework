from __future__ import annotations

import numpy as np
import pytest

from cnn_inversion_3d.failure_mode_analysis import geometry, validate_alignment


def test_geometry_uses_zyx_and_ten_meter_cells() -> None:
    density = np.zeros((24, 64, 64), dtype=np.float32)
    density[2:6, 4:9, 10:17] = 0.5

    result = geometry(density)

    assert result["top_depth_m"] == 20.0
    assert result["bottom_depth_m"] == 60.0
    assert result["thickness_m"] == 40.0
    assert result["width_x_m"] == 70.0
    assert result["width_y_m"] == 50.0


def test_geometry_returns_nan_for_empty_prediction() -> None:
    result = geometry(np.zeros((24, 64, 64), dtype=np.float32))
    assert all(np.isnan(value) for value in result.values())


def test_alignment_matches_ids_not_order() -> None:
    ids = [f"sample_{index:06d}" for index in range(100)]
    result = validate_alignment(ids, {"E05": ids[::-1], "E06": ids[1:] + ids[:1], "E07": ids})
    assert result == ids


def test_alignment_rejects_missing_sample() -> None:
    ids = [f"sample_{index:06d}" for index in range(100)]
    with pytest.raises(ValueError, match="Test-set mismatch"):
        validate_alignment(ids, {"E05": ids[:-1], "E06": ids, "E07": ids})
