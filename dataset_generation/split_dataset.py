from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


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
            "Split a generated FWD3D dataset into training, "
            "validation, and test manifests."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "datasets/fwd3d_smoke_test"
        ),
        help=(
            "Dataset directory. Relative paths are interpreted "
            "from the repository root."
        ),
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction assigned to training.",
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction assigned to validation.",
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Fraction assigned to testing.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260727,
        help="Random seed used to shuffle samples.",
    )

    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help=(
            "Optional exact training count. Must be supplied together "
            "with --validation-count and --test-count."
        ),
    )

    parser.add_argument(
        "--validation-count",
        type=int,
        default=None,
        help=(
            "Optional exact validation count. Must be supplied together "
            "with --train-count and --test-count."
        ),
    )

    parser.add_argument(
        "--test-count",
        type=int,
        default=None,
        help=(
            "Optional exact test count. Must be supplied together "
            "with --train-count and --validation-count."
        ),
    )

    parser.add_argument(
        "--holdout-dataset",
        type=Path,
        default=None,
        help=(
            "Optional reference dataset whose validation and test sample "
            "identities are preserved. Requires explicit split counts."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing split manifest files.",
    )

    return parser


def resolve_dataset_directory(
    *,
    repository_root: Path,
    dataset_argument: Path,
) -> Path:
    """
    Resolve and validate the dataset directory.
    """

    dataset_directory = dataset_argument

    if not dataset_directory.is_absolute():
        dataset_directory = (
            repository_root
            / dataset_directory
        )

    dataset_directory = dataset_directory.resolve()

    if not dataset_directory.exists():
        raise FileNotFoundError(
            "Dataset directory does not exist:\n"
            f"{dataset_directory}"
        )

    return dataset_directory


def validate_split_fractions(
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> None:
    """
    Validate the requested split fractions.
    """

    fractions = {
        "train_fraction": train_fraction,
        "validation_fraction": (
            validation_fraction
        ),
        "test_fraction": test_fraction,
    }

    for name, value in fractions.items():
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"{name} must be between zero and one."
            )

    total = (
        train_fraction
        + validation_fraction
        + test_fraction
    )

    if not np.isclose(
        total,
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Split fractions must sum to one. "
            f"Received {total:.12f}."
        )


def load_manifest(
    manifest_path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Load the complete dataset manifest.

    Returns
    -------
    tuple
        Manifest field names and rows.
    """

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        reader = csv.DictReader(
            manifest_file
        )

        if reader.fieldnames is None:
            raise ValueError(
                "Manifest contains no header."
            )

        field_names = list(
            reader.fieldnames
        )

        rows = list(
            reader
        )

    if not rows:
        raise ValueError(
            "Manifest contains no samples."
        )

    return field_names, rows


def calculate_split_counts(
    *,
    number_of_samples: int,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[int, int, int]:
    """
    Calculate integer split sizes.

    Training and validation counts are rounded down. Remaining samples
    are assigned to testing so no samples are lost.
    """

    if number_of_samples < 3:
        raise ValueError(
            "At least three samples are required to create "
            "train, validation, and test splits."
        )

    train_count = int(
        np.floor(
            number_of_samples
            * train_fraction
        )
    )

    validation_count = int(
        np.floor(
            number_of_samples
            * validation_fraction
        )
    )

    test_count = (
        number_of_samples
        - train_count
        - validation_count
    )

    if train_count < 1:
        raise ValueError(
            "Training split contains no samples."
        )

    if validation_count < 1:
        raise ValueError(
            "Validation split contains no samples."
        )

    if test_count < 1:
        raise ValueError(
            "Test split contains no samples."
        )

    return (
        train_count,
        validation_count,
        test_count,
    )


def validate_explicit_split_counts(
    *,
    number_of_samples: int,
    train_count: int,
    validation_count: int,
    test_count: int,
) -> tuple[int, int, int]:
    """
    Validate exact requested dataset split counts.

    Parameters
    ----------
    number_of_samples
        Number of rows in the complete manifest.
    train_count, validation_count, test_count
        Requested positive sample counts.

    Returns
    -------
    tuple
        Validated training, validation, and test counts.
    """

    counts = {
        "train_count": train_count,
        "validation_count": (
            validation_count
        ),
        "test_count": test_count,
    }

    for name, value in counts.items():
        if value < 1:
            raise ValueError(
                f"{name} must be at least one."
            )

    requested_total = (
        train_count
        + validation_count
        + test_count
    )

    if requested_total != number_of_samples:
        raise ValueError(
            "Explicit split counts must sum to the manifest length. "
            f"Received {requested_total:,} for "
            f"{number_of_samples:,} samples."
        )

    return (
        train_count,
        validation_count,
        test_count,
    )


def write_manifest(
    *,
    output_path: Path,
    field_names: list[str],
    rows: list[dict[str, str]],
    overwrite: bool,
) -> None:
    """
    Write one split manifest.
    """

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            "Split manifest already exists:\n"
            f"{output_path}\n"
            "Use --overwrite to replace it."
        )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer: csv.DictWriter[str] = (
            csv.DictWriter(
                output_file,
                fieldnames=field_names,
            )
        )

        writer.writeheader()
        writer.writerows(rows)


def validate_no_overlap(
    *,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
) -> None:
    """
    Confirm that no sample occurs in more than one split.
    """

    train_paths = {
        row["relative_path"]
        for row in train_rows
    }

    validation_paths = {
        row["relative_path"]
        for row in validation_rows
    }

    test_paths = {
        row["relative_path"]
        for row in test_rows
    }

    if train_paths & validation_paths:
        raise RuntimeError(
            "Training and validation splits overlap."
        )

    if train_paths & test_paths:
        raise RuntimeError(
            "Training and test splits overlap."
        )

    if validation_paths & test_paths:
        raise RuntimeError(
            "Validation and test splits overlap."
        )


def write_split_metadata(
    *,
    output_path: Path,
    seed: int,
    total_samples: int,
    train_count: int,
    validation_count: int,
    test_count: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    requested_counts: dict[str, int] | None = None,
    holdout_dataset: Path | None = None,
) -> None:
    """
    Save split metadata as JSON.
    """

    metadata: dict[str, Any] = {
        "random_seed": seed,
        "total_samples": total_samples,
        "requested_fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "sample_counts": {
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
        },
        "actual_fractions": {
            "train": (
                train_count / total_samples
            ),
            "validation": (
                validation_count
                / total_samples
            ),
            "test": (
                test_count / total_samples
            ),
        },
        "split_manifests": {
            "train": "train_manifest.csv",
            "validation": (
                "validation_manifest.csv"
            ),
            "test": "test_manifest.csv",
        },
    }

    if requested_counts is not None:
        metadata[
            "split_method"
        ] = "explicit_counts"
        metadata[
            "requested_counts"
        ] = requested_counts

    if holdout_dataset is not None:
        metadata[
            "holdout_dataset"
        ] = str(
            holdout_dataset
        )
        metadata[
            "holdout_identity_preserved"
        ] = True

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=2,
        )


def main() -> None:
    """
    Create deterministic dataset splits.
    """

    parser = build_argument_parser()
    arguments = parser.parse_args()

    explicit_counts = (
        arguments.train_count,
        arguments.validation_count,
        arguments.test_count,
    )
    supplied_count_options = sum(
        value is not None
        for value in explicit_counts
    )

    if supplied_count_options not in {
        0,
        3,
    }:
        raise ValueError(
            "--train-count, --validation-count, and --test-count "
            "must be supplied together."
        )

    if (
        arguments.holdout_dataset is not None
        and supplied_count_options != 3
    ):
        raise ValueError(
            "--holdout-dataset requires explicit --train-count, "
            "--validation-count, and --test-count values."
        )

    if supplied_count_options == 0:
        validate_split_fractions(
            train_fraction=(
                arguments.train_fraction
            ),
            validation_fraction=(
                arguments.validation_fraction
            ),
            test_fraction=(
                arguments.test_fraction
            ),
        )

    repository_root = find_repository_root()

    dataset_directory = (
        resolve_dataset_directory(
            repository_root=repository_root,
            dataset_argument=(
                arguments.dataset
            ),
        )
    )

    manifest_path = (
        dataset_directory
        / "manifest.csv"
    )

    field_names, rows = load_manifest(
        manifest_path
    )

    if supplied_count_options == 3:
        (
            train_count,
            validation_count,
            test_count,
        ) = validate_explicit_split_counts(
            number_of_samples=len(rows),
            train_count=int(
                arguments.train_count
            ),
            validation_count=int(
                arguments.validation_count
            ),
            test_count=int(
                arguments.test_count
            ),
        )
    else:
        (
            train_count,
            validation_count,
            test_count,
        ) = calculate_split_counts(
            number_of_samples=len(rows),
            train_fraction=(
                arguments.train_fraction
            ),
            validation_fraction=(
                arguments.validation_fraction
            ),
        )

    train_fraction = (
        train_count / len(rows)
    )
    validation_fraction = (
        validation_count / len(rows)
    )
    test_fraction = (
        test_count / len(rows)
    )

    random_generator = (
        np.random.default_rng(
            arguments.seed
        )
    )

    holdout_dataset = None

    if arguments.holdout_dataset is not None:
        holdout_dataset = resolve_dataset_directory(
            repository_root=repository_root,
            dataset_argument=(
                arguments.holdout_dataset
            ),
        )
        _, reference_validation_rows = (
            load_manifest(
                holdout_dataset
                / "validation_manifest.csv"
            )
        )
        _, reference_test_rows = load_manifest(
            holdout_dataset
            / "test_manifest.csv"
        )

        if (
            len(reference_validation_rows)
            != validation_count
            or len(reference_test_rows)
            != test_count
        ):
            raise ValueError(
                "Reference holdout counts do not match requested "
                "validation and test counts."
            )

        rows_by_path = {
            row["relative_path"]: row
            for row in rows
        }
        validation_paths = [
            row["relative_path"]
            for row in reference_validation_rows
        ]
        test_paths = [
            row["relative_path"]
            for row in reference_test_rows
        ]
        missing_paths = [
            path
            for path in (
                validation_paths
                + test_paths
            )
            if path not in rows_by_path
        ]

        if missing_paths:
            raise ValueError(
                "Reference holdout contains sample paths absent from "
                f"the target dataset, including {missing_paths[0]!r}."
            )

        holdout_paths = set(
            validation_paths
            + test_paths
        )
        training_candidates = [
            row
            for row in rows
            if row["relative_path"]
            not in holdout_paths
        ]

        if len(training_candidates) != train_count:
            raise ValueError(
                "Samples remaining after fixed holdout selection do not "
                f"match train_count: {len(training_candidates)} != "
                f"{train_count}."
            )

        training_order = (
            random_generator.permutation(
                len(training_candidates)
            )
        )
        train_rows = [
            training_candidates[
                int(index)
            ]
            for index in training_order
        ]
        validation_rows = [
            rows_by_path[path]
            for path in validation_paths
        ]
        test_rows = [
            rows_by_path[path]
            for path in test_paths
        ]
    else:
        shuffled_indices = (
            random_generator.permutation(
                len(rows)
            )
        )
        shuffled_rows = [
            rows[int(index)]
            for index in shuffled_indices
        ]
        train_end = train_count
        validation_end = (
            train_count
            + validation_count
        )
        train_rows = shuffled_rows[
            :train_end
        ]
        validation_rows = shuffled_rows[
            train_end:validation_end
        ]
        test_rows = shuffled_rows[
            validation_end:
        ]

    if len(test_rows) != test_count:
        raise RuntimeError(
            "Calculated test count does not match the "
            "number of assigned test samples."
        )

    validate_no_overlap(
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
    )

    train_manifest_path = (
        dataset_directory
        / "train_manifest.csv"
    )

    validation_manifest_path = (
        dataset_directory
        / "validation_manifest.csv"
    )

    test_manifest_path = (
        dataset_directory
        / "test_manifest.csv"
    )

    write_manifest(
        output_path=train_manifest_path,
        field_names=field_names,
        rows=train_rows,
        overwrite=arguments.overwrite,
    )

    write_manifest(
        output_path=validation_manifest_path,
        field_names=field_names,
        rows=validation_rows,
        overwrite=arguments.overwrite,
    )

    write_manifest(
        output_path=test_manifest_path,
        field_names=field_names,
        rows=test_rows,
        overwrite=arguments.overwrite,
    )

    split_metadata_path = (
        dataset_directory
        / "split_metadata.json"
    )

    if (
        split_metadata_path.exists()
        and not arguments.overwrite
    ):
        raise FileExistsError(
            "Split metadata already exists:\n"
            f"{split_metadata_path}\n"
            "Use --overwrite to replace it."
        )

    write_split_metadata(
        output_path=split_metadata_path,
        seed=arguments.seed,
        total_samples=len(rows),
        train_count=train_count,
        validation_count=(
            validation_count
        ),
        test_count=test_count,
        train_fraction=(
            train_fraction
        ),
        validation_fraction=(
            validation_fraction
        ),
        test_fraction=test_fraction,
        requested_counts=(
            {
                "train": train_count,
                "validation": (
                    validation_count
                ),
                "test": test_count,
            }
            if supplied_count_options == 3
            else None
        ),
        holdout_dataset=holdout_dataset,
    )

    print()
    print("Dataset split complete")
    print("=" * 22)
    print(
        f"Dataset: {dataset_directory}"
    )
    print(
        f"Total samples: {len(rows):,}"
    )
    print(
        f"Training samples: "
        f"{len(train_rows):,}"
    )
    print(
        f"Validation samples: "
        f"{len(validation_rows):,}"
    )
    print(
        f"Test samples: "
        f"{len(test_rows):,}"
    )
    print()
    print(
        f"Training manifest: "
        f"{train_manifest_path}"
    )
    print(
        f"Validation manifest: "
        f"{validation_manifest_path}"
    )
    print(
        f"Test manifest: "
        f"{test_manifest_path}"
    )
    print(
        f"Split metadata: "
        f"{split_metadata_path}"
    )
    print()
    print("No split overlap: PASSED")


if __name__ == "__main__":
    main()
