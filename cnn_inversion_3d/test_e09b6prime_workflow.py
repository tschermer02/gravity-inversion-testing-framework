"""Lightweight controls for the E09B-6-prime dataset-only experiment."""
from __future__ import annotations

import csv

import numpy as np

from cnn_inversion_3d.e09b6prime_workflow import manifests_match
from cnn_inversion_3d.single_plane_review import SinglePlaneReviewConfig
from dataset_generation.generate_e09b6prime_dataset import (
    SIZE_LIMITS, balanced_schedule, sample_body, valid_dimension_combinations,
)


def test_size_strata_cover_every_valid_dimension_combination_once() -> None:
    groups = valid_dimension_combinations(SinglePlaneReviewConfig())
    combinations = [item for values in groups.values() for item in values]
    assert len(combinations) == 13 * 13 * 7
    assert len(set(combinations)) == len(combinations)
    assert all(np.prod(item) <= SIZE_LIMITS[0] for item in groups["small"])
    assert all(SIZE_LIMITS[0] < np.prod(item) <= SIZE_LIMITS[1] for item in groups["medium"])
    assert all(np.prod(item) > SIZE_LIMITS[1] for item in groups["large"])


def test_balanced_schedule_has_independent_nearly_equal_3x3_coverage() -> None:
    schedule = balanced_schedule(10000, np.random.default_rng(20260901))
    counts = {cell: schedule.count(cell) for cell in set(schedule)}
    assert len(counts) == 9
    assert max(counts.values()) - min(counts.values()) <= 1


def test_sample_body_preserves_canonical_ranges_and_requested_strata() -> None:
    config = SinglePlaneReviewConfig(); rng = np.random.default_rng(7)
    dimensions = valid_dimension_combinations(config)
    for size in ("small", "medium", "large"):
        for density in ("low", "medium", "high"):
            body = sample_body(rng, config, dimensions, size, density)
            assert body["body_size_group"] == size
            assert body["density_group"] == density
            assert 20 <= body["top_depth_m"] <= 80
            assert body["bottom_depth_m"] <= 160
            assert 0.2 <= body["density_contrast"] <= 1.0


def test_manifest_match_requires_identical_rows_and_order(tmp_path) -> None:
    fields = ("sample_id", "relative_path")
    left = tmp_path/"left.csv"; right = tmp_path/"right.csv"
    for path in (left, right):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerow({"sample_id":"sample_1","relative_path":"samples/sample_1.npz"})
    assert manifests_match(left, right)
    right.write_text("sample_id,relative_path\nsample_2,samples/sample_2.npz\n", encoding="utf-8")
    assert not manifests_match(left, right)
