"""Regression tests for canonical single-plane dataset geometry."""

from __future__ import annotations

import numpy as np

from cnn_inversion_3d.single_plane_review import SinglePlaneReviewConfig
from dataset_generation.generate_single_plane_dataset import _sample_body


def test_canonical_geometry_is_the_single_source_of_truth() -> None:
    """Verify approved model, receiver, range, and margin definitions."""

    config = SinglePlaneReviewConfig()
    assert config.dataset_geometry_version == "canonical_single_plane_v1"
    assert config.density_shape == (24, 64, 64)
    assert config.gravity_shape == (81, 81)
    assert config.cnn_gravity_shape == (81, 81, 1)
    assert config.density_x_edges_m == (-5.0, 635.0)
    assert config.density_y_edges_m == (-5.0, 635.0)
    assert config.density_z_edges_m == (0.0, 240.0)
    assert config.observation_x_m[[0, -1]].tolist() == [-85.0, 715.0]
    assert config.observation_y_m[[0, -1]].tolist() == [-85.0, 715.0]
    assert config.horizontal_margin_cells == 8
    assert config.horizontal_margin_m == 80.0


def test_sampler_cannot_reproduce_old_edge_touching_body() -> None:
    """Ensure the old sample_000114 x_end=64 geometry is impossible."""

    config = SinglePlaneReviewConfig()
    rng = np.random.default_rng(20260727)
    for _ in range(10_000):
        body = _sample_body(rng, config)
        assert int(body["x_start"]) >= 8
        assert int(body["x_end"]) <= 56
        assert int(body["y_start"]) >= 8
        assert int(body["y_end"]) <= 56
        assert float(body["bottom_depth_m"]) <= 160.0

