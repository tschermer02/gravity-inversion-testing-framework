from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


GravityScaleMethod = Literal[
    "absolute_maximum",
    "percentile_99",
    "standard_deviation",
]


@dataclass(frozen=True)
class GravityNormalization:
    """
    Training-derived gravity normalization configuration.

    Parameters
    ----------
    method
        Statistic used to define the global gravity scale.
    scale
        Positive constant by which all gravity values are divided.
    source_path
        JSON file from which the training-only statistic was loaded.
    """

    method: GravityScaleMethod
    scale: float
    source_path: Path

    def validate(self) -> None:
        """
        Validate the normalization configuration.
        """

        if self.method not in {
            "absolute_maximum",
            "percentile_99",
            "standard_deviation",
        }:
            raise ValueError(
                f"Unsupported gravity normalization method: {self.method}."
            )

        if self.scale <= 0.0:
            raise ValueError(
                "Gravity normalization scale must be greater than zero."
            )


def load_gravity_normalization(
    *,
    summary_path: str | Path,
    method: GravityScaleMethod,
) -> GravityNormalization:
    """
    Load a gravity scale from a training-distribution summary.

    The summary must have been calculated using only the training split.

    Parameters
    ----------
    summary_path
        Path to ``training_gravity_distribution.json``.
    method
        Statistic used as the normalization scale.

    Returns
    -------
    GravityNormalization
        Validated normalization configuration.

    Raises
    ------
    FileNotFoundError
        If the summary file does not exist.
    KeyError
        If the requested statistic is missing.
    ValueError
        If the loaded scale is invalid.
    """

    path = Path(
        summary_path
    ).resolve()

    if not path.exists():
        raise FileNotFoundError(
            "Gravity-distribution summary does not exist:\n"
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as summary_file:
        summary = json.load(
            summary_file
        )

    field_by_method = {
        "absolute_maximum": (
            "recommended_absolute_max_scale"
        ),
        "percentile_99": (
            "recommended_percentile_99_scale"
        ),
        "standard_deviation": (
            "recommended_standard_deviation_scale"
        ),
    }

    field_name = field_by_method[
        method
    ]

    if field_name not in summary:
        raise KeyError(
            f"Normalization summary contains no '{field_name}' value."
        )

    normalization = GravityNormalization(
        method=method,
        scale=float(
            summary[field_name]
        ),
        source_path=path,
    )

    normalization.validate()

    return normalization