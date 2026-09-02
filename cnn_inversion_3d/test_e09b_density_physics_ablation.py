"""Control-table tests for E09B-9/10/11."""
from cnn_inversion_3d.e09b_density_physics_ablation import ALL, RUNS


def test_exact_two_by_two_ablation_configuration():
    assert ALL["E09B-6'"]["body"]==0 and ALL["E09B-6'"]["gravity"]==0
    assert RUNS["E09B-9"]["body"]==1 and RUNS["E09B-9"]["gravity"]==0
    assert RUNS["E09B-10"]["body"]==0 and RUNS["E09B-10"]["gravity"]==.001
    assert RUNS["E09B-11"]["body"]==1 and RUNS["E09B-11"]["gravity"]==.001
    assert tuple(RUNS)==("E09B-9","E09B-10","E09B-11")
