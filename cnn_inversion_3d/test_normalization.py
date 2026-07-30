from __future__ import annotations

import json
from pathlib import Path

import pytest

from cnn_inversion_3d.normalization import (
    load_gravity_normalization,
)


def write_summary(
    path: Path,
) -> None:
    """
    Write a minimal valid gravity-distribution summary.
    """

    summary = {
        "recommended_absolute_max_scale": 0.25,
        "recommended_percentile_99_scale": 0.10,
        "recommended_standard_deviation_scale": 0.03,
    }

    path.write_text(
        json.dumps(
            summary
        ),
        encoding="utf-8",
    )


def test_load_percentile_99_scale(
    tmp_path: Path,
) -> None:
    """
    Verify loading the 99th-percentile gravity scale.
    """

    summary_path = (
        tmp_path
        / "summary.json"
    )

    write_summary(
        summary_path
    )

    normalization = load_gravity_normalization(
        summary_path=summary_path,
        method="percentile_99",
    )

    assert normalization.method == "percentile_99"
    assert normalization.scale == 0.10
    assert normalization.source_path == summary_path.resolve()


def test_load_absolute_maximum_scale(
    tmp_path: Path,
) -> None:
    """
    Verify loading the absolute-maximum gravity scale.
    """

    summary_path = (
        tmp_path
        / "summary.json"
    )

    write_summary(
        summary_path
    )

    normalization = load_gravity_normalization(
        summary_path=summary_path,
        method="absolute_maximum",
    )

    assert normalization.scale == 0.25


def test_load_standard_deviation_scale(
    tmp_path: Path,
) -> None:
    """
    Verify loading the standard-deviation gravity scale.
    """

    summary_path = (
        tmp_path
        / "summary.json"
    )

    write_summary(
        summary_path
    )

    normalization = load_gravity_normalization(
        summary_path=summary_path,
        method="standard_deviation",
    )

    assert normalization.scale == 0.03


def test_missing_summary_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Verify that a missing summary file raises an error.
    """

    with pytest.raises(
        FileNotFoundError
    ):
        load_gravity_normalization(
            summary_path=(
                tmp_path
                / "missing.json"
            ),
            method="percentile_99",
        )


def test_nonpositive_scale_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Verify that a nonpositive loaded scale is rejected.
    """

    summary_path = (
        tmp_path
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            {
                "recommended_percentile_99_scale": 0.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        load_gravity_normalization(
            summary_path=summary_path,
            method="percentile_99",
        )