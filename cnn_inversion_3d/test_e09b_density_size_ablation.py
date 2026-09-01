"""Configuration checks for the controlled E09B density/size ablation."""

from cnn_inversion_3d.e09b_density_size_ablation import EXPERIMENTS, NEW_LABELS


def test_required_runs_form_the_requested_controlled_design() -> None:
    assert NEW_LABELS == ("E09B-5", "E09B-6", "E09B-7", "E09B-8")
    assert EXPERIMENTS["E09B-5"] == {
        "slug": "E09B_5_depth2p5", "depth": 2.5, "amplitude": 0.0, "small": False
    }
    for label in ("E09B-2", "E09B-6", "E09B-7", "E09B-8"):
        assert EXPERIMENTS[label]["depth"] == 2.0
    assert EXPERIMENTS["E09B-2"]["amplitude"] == 0.0
    assert EXPERIMENTS["E09B-2"]["small"] is False
    assert EXPERIMENTS["E09B-6"]["amplitude"] == 1.0
    assert EXPERIMENTS["E09B-6"]["small"] is False
    assert EXPERIMENTS["E09B-7"]["amplitude"] == 0.0
    assert EXPERIMENTS["E09B-7"]["small"] is True
    assert EXPERIMENTS["E09B-8"]["amplitude"] == 1.0
    assert EXPERIMENTS["E09B-8"]["small"] is True
