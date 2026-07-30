from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GravityDistributionSummary:
    """
    Summary statistics for one training gravity dataset.

    All statistics are calculated only from samples listed in the training
    manifest.
    """

    number_of_samples: int
    number_of_values: int

    global_minimum: float
    global_maximum: float
    global_absolute_maximum: float

    global_mean: float
    global_standard_deviation: float

    absolute_value_median: float
    absolute_value_percentile_90: float
    absolute_value_percentile_95: float
    absolute_value_percentile_99: float
    absolute_value_percentile_99_9: float

    sample_maximum_median: float
    sample_maximum_minimum: float
    sample_maximum_maximum: float

    sample_standard_deviation_median: float
    sample_standard_deviation_minimum: float
    sample_standard_deviation_maximum: float

    recommended_absolute_max_scale: float
    recommended_percentile_99_scale: float
    recommended_standard_deviation_scale: float


def find_repository_root() -> Path:
    """
    Return the repository root.
    """

    return Path(__file__).resolve().parents[1]


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Analyze gravity magnitudes using only a dataset's "
            "training split."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=(
            "Dataset directory. Relative paths are interpreted "
            "from the repository root."
        ),
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default="train_manifest.csv",
        help="Training manifest filename.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional JSON output path. By default, the summary is "
            "saved inside the dataset directory."
        ),
    )

    return parser


def resolve_path(
    *,
    repository_root: Path,
    path: Path,
) -> Path:
    """
    Resolve a path relative to the repository root.
    """

    if not path.is_absolute():
        path = repository_root / path

    return path.resolve()


def load_manifest_rows(
    *,
    dataset_directory: Path,
    manifest_name: str,
) -> list[dict[str, str]]:
    """
    Load sample rows from a dataset manifest.

    Parameters
    ----------
    dataset_directory
        Dataset root directory.
    manifest_name
        Manifest filename.

    Returns
    -------
    list of dict
        Manifest rows.
    """

    manifest_path = (
        dataset_directory
        / manifest_name
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest does not exist:\n{manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        rows = list(
            csv.DictReader(
                manifest_file
            )
        )

    if not rows:
        raise ValueError(
            f"{manifest_name} contains no sample rows."
        )

    return rows


def load_training_gravity_arrays(
    *,
    dataset_directory: Path,
    manifest_rows: list[dict[str, str]],
) -> list[np.ndarray]:
    """
    Load gravity arrays referenced by the training manifest.

    Parameters
    ----------
    dataset_directory
        Dataset root directory.
    manifest_rows
        Training-manifest rows.

    Returns
    -------
    list of numpy.ndarray
        Gravity arrays as float64 values.

    Raises
    ------
    ValueError
        If a gravity array has an unexpected shape or invalid values.
    """

    gravity_arrays: list[np.ndarray] = []

    expected_shape = (
        8,
        64,
        64,
    )

    for row in manifest_rows:
        relative_path = row.get(
            "relative_path"
        )

        if relative_path is None:
            raise ValueError(
                "Manifest row has no relative_path value."
            )

        sample_path = (
            dataset_directory
            / relative_path
        )

        if not sample_path.exists():
            raise FileNotFoundError(
                f"Sample does not exist:\n{sample_path}"
            )

        with np.load(
            sample_path
        ) as sample:
            if "gravity" not in sample:
                raise KeyError(
                    f"{sample_path.name} contains no gravity array."
                )

            gravity = np.asarray(
                sample["gravity"],
                dtype=np.float64,
            )

        if gravity.shape != expected_shape:
            raise ValueError(
                f"{sample_path.name}: expected gravity shape "
                f"{expected_shape}, received {gravity.shape}."
            )

        if not np.all(
            np.isfinite(
                gravity
            )
        ):
            raise ValueError(
                f"{sample_path.name}: gravity contains invalid values."
            )

        gravity_arrays.append(
            gravity
        )

    return gravity_arrays


def calculate_distribution_summary(
    gravity_arrays: list[np.ndarray],
) -> GravityDistributionSummary:
    """
    Calculate training-only gravity distribution statistics.

    Parameters
    ----------
    gravity_arrays
        Training gravity volumes.

    Returns
    -------
    GravityDistributionSummary
        Global and per-sample distribution statistics.
    """

    if not gravity_arrays:
        raise ValueError(
            "At least one gravity array is required."
        )

    flattened = np.concatenate(
        [
            gravity.reshape(-1)
            for gravity in gravity_arrays
        ]
    )

    absolute_values = np.abs(
        flattened
    )

    sample_absolute_maxima = np.asarray(
        [
            np.max(
                np.abs(
                    gravity
                )
            )
            for gravity in gravity_arrays
        ],
        dtype=np.float64,
    )

    sample_standard_deviations = np.asarray(
        [
            np.std(
                gravity
            )
            for gravity in gravity_arrays
        ],
        dtype=np.float64,
    )

    global_standard_deviation = float(
        np.std(
            flattened
        )
    )

    global_absolute_maximum = float(
        np.max(
            absolute_values
        )
    )

    percentile_99 = float(
        np.percentile(
            absolute_values,
            99.0,
        )
    )

    if global_absolute_maximum <= 0.0:
        raise ValueError(
            "Training gravity is identically zero."
        )

    if percentile_99 <= 0.0:
        raise ValueError(
            "The 99th-percentile gravity magnitude is zero."
        )

    if global_standard_deviation <= 0.0:
        raise ValueError(
            "Training gravity has zero standard deviation."
        )

    return GravityDistributionSummary(
        number_of_samples=len(
            gravity_arrays
        ),
        number_of_values=int(
            flattened.size
        ),
        global_minimum=float(
            np.min(
                flattened
            )
        ),
        global_maximum=float(
            np.max(
                flattened
            )
        ),
        global_absolute_maximum=(
            global_absolute_maximum
        ),
        global_mean=float(
            np.mean(
                flattened
            )
        ),
        global_standard_deviation=(
            global_standard_deviation
        ),
        absolute_value_median=float(
            np.percentile(
                absolute_values,
                50.0,
            )
        ),
        absolute_value_percentile_90=float(
            np.percentile(
                absolute_values,
                90.0,
            )
        ),
        absolute_value_percentile_95=float(
            np.percentile(
                absolute_values,
                95.0,
            )
        ),
        absolute_value_percentile_99=(
            percentile_99
        ),
        absolute_value_percentile_99_9=float(
            np.percentile(
                absolute_values,
                99.9,
            )
        ),
        sample_maximum_median=float(
            np.median(
                sample_absolute_maxima
            )
        ),
        sample_maximum_minimum=float(
            np.min(
                sample_absolute_maxima
            )
        ),
        sample_maximum_maximum=float(
            np.max(
                sample_absolute_maxima
            )
        ),
        sample_standard_deviation_median=float(
            np.median(
                sample_standard_deviations
            )
        ),
        sample_standard_deviation_minimum=float(
            np.min(
                sample_standard_deviations
            )
        ),
        sample_standard_deviation_maximum=float(
            np.max(
                sample_standard_deviations
            )
        ),
        recommended_absolute_max_scale=(
            global_absolute_maximum
        ),
        recommended_percentile_99_scale=(
            percentile_99
        ),
        recommended_standard_deviation_scale=(
            global_standard_deviation
        ),
    )


def print_summary(
    summary: GravityDistributionSummary,
) -> None:
    """
    Print a readable distribution report.
    """

    print()
    print("Training gravity distribution")
    print("=" * 29)

    print(
        f"Samples: {summary.number_of_samples:,}"
    )
    print(
        f"Gravity values: {summary.number_of_values:,}"
    )

    print()
    print("Global signed distribution")
    print("-" * 26)
    print(
        f"Minimum: {summary.global_minimum:.12e} mGal"
    )
    print(
        f"Maximum: {summary.global_maximum:.12e} mGal"
    )
    print(
        "Absolute maximum: "
        f"{summary.global_absolute_maximum:.12e} mGal"
    )
    print(
        f"Mean: {summary.global_mean:.12e} mGal"
    )
    print(
        "Standard deviation: "
        f"{summary.global_standard_deviation:.12e} mGal"
    )

    print()
    print("Absolute-value percentiles")
    print("-" * 26)
    print(
        f"50%: {summary.absolute_value_median:.12e} mGal"
    )
    print(
        f"90%: {summary.absolute_value_percentile_90:.12e} mGal"
    )
    print(
        f"95%: {summary.absolute_value_percentile_95:.12e} mGal"
    )
    print(
        f"99%: {summary.absolute_value_percentile_99:.12e} mGal"
    )
    print(
        f"99.9%: "
        f"{summary.absolute_value_percentile_99_9:.12e} mGal"
    )

    print()
    print("Per-sample absolute maxima")
    print("-" * 27)
    print(
        f"Minimum: "
        f"{summary.sample_maximum_minimum:.12e} mGal"
    )
    print(
        f"Median: "
        f"{summary.sample_maximum_median:.12e} mGal"
    )
    print(
        f"Maximum: "
        f"{summary.sample_maximum_maximum:.12e} mGal"
    )

    print()
    print("Candidate global scales")
    print("-" * 23)
    print(
        "Absolute maximum: "
        f"{summary.recommended_absolute_max_scale:.12e}"
    )
    print(
        "99th percentile: "
        f"{summary.recommended_percentile_99_scale:.12e}"
    )
    print(
        "Standard deviation: "
        f"{summary.recommended_standard_deviation_scale:.12e}"
    )


def save_summary(
    *,
    summary: GravityDistributionSummary,
    output_path: Path,
) -> None:
    """
    Save the distribution summary as JSON.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            asdict(
                summary
            ),
            output_file,
            indent=2,
        )


def main() -> None:
    """
    Analyze the training gravity distribution.
    """

    parser = build_argument_parser()
    arguments = parser.parse_args()

    repository_root = find_repository_root()

    dataset_directory = resolve_path(
        repository_root=repository_root,
        path=arguments.dataset,
    )

    if not dataset_directory.exists():
        raise FileNotFoundError(
            f"Dataset does not exist:\n{dataset_directory}"
        )

    manifest_rows = load_manifest_rows(
        dataset_directory=dataset_directory,
        manifest_name=arguments.manifest,
    )

    gravity_arrays = load_training_gravity_arrays(
        dataset_directory=dataset_directory,
        manifest_rows=manifest_rows,
    )

    summary = calculate_distribution_summary(
        gravity_arrays
    )

    if arguments.output is None:
        output_path = (
            dataset_directory
            / "training_gravity_distribution.json"
        )
    else:
        output_path = resolve_path(
            repository_root=repository_root,
            path=arguments.output,
        )

    print_summary(
        summary
    )

    save_summary(
        summary=summary,
        output_path=output_path,
    )

    print()
    print(
        f"Saved summary: {output_path}"
    )


if __name__ == "__main__":
    main()