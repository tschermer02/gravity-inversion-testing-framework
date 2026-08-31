from cnn_inversion_3d.e09b_lambda_sweep import RUNS

def test_sweep_has_exact_controlled_lambda_configurations():
    assert [(v["lambda_depth"],v["lambda_sensitivity"]) for v in RUNS.values()] == [
        (1.5,1.0),(2.0,1.0),(3.0,1.0),(2.0,0.5)
    ]
    assert len({v["slug"] for v in RUNS.values()}) == 4
