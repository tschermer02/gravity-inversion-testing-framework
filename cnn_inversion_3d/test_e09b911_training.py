"""Focused tests for the controlled E09B-9/10/11 loss ablation."""
import numpy as np
import tensorflow as tf

from cnn_inversion_3d.dataset import DENSITY_SHAPE, SINGLE_PLANE_GRAVITY_SHAPE
from cnn_inversion_3d.differentiable_gravity import DifferentiableSinglePlaneGz
from cnn_inversion_3d.e09b_training import build_e09b_sensitivity_weights
from cnn_inversion_3d.e09b911_training import (
    E09B911LossConfig, E09B911TrainingModel, body_density_mse_per_sample,
)
from cnn_inversion_3d.model import ModelConfig, build_asymmetric_2d_unet_model
from cnn_inversion_3d.train import run_e09b911_preflight


def batch():
    truth=np.zeros((1,*DENSITY_SHAPE),np.float32);truth[:,2:6,25:31,27:35,:]=0.6
    return tf.zeros((1,*SINGLE_PLANE_GRAVITY_SHAPE)),tf.constant(truth)


def wrapper(body=0.0,gravity=0.0):
    _,weights=build_e09b_sensitivity_weights()
    inversion=build_asymmetric_2d_unet_model(ModelConfig(base_filters=1))
    config=E09B911LossConfig(lambda_depth=2,lambda_amplitude=1,
        lambda_body_density=body,lambda_gravity=gravity)
    return E09B911TrainingModel(inversion,weights,DifferentiableSinglePlaneGz(),
                                gravity_scale=.23,loss_config=config)


class FailingForwardOperator(tf.keras.layers.Layer):
    """Sentinel proving that a disabled physics term is never evaluated."""

    def call(self, prediction):
        raise AssertionError("gravity forward operator must not be called")


def test_body_density_loss_is_true_support_voxel_mse():
    _,truth=batch();prediction=truth*.5
    np.testing.assert_allclose(body_density_mse_per_sample(truth,prediction),[.09],rtol=1e-5)


def test_total_adds_only_configured_terms_and_operator_is_fixed():
    gravity,truth=batch();model=wrapper(body=1,gravity=.001)
    terms=model.compute_loss_terms(gravity,truth,training=False)
    expected=terms[1]+2*terms[4]+terms[5]+terms[6]+terms[7]+terms[9]
    np.testing.assert_allclose(terms[-1],expected,rtol=1e-6)
    assert not model.forward_operator.trainable
    assert len(model.forward_operator.trainable_variables)==0
    assert model.inversion_model.count_params()==3076


def test_zero_gravity_weight_skips_forward_operator_and_gravity_trackers():
    gravity,truth=batch();_,weights=build_e09b_sensitivity_weights()
    inversion=build_asymmetric_2d_unet_model(ModelConfig(base_filters=1))
    model=E09B911TrainingModel(
        inversion,weights,FailingForwardOperator(),gravity_scale=.23,
        loss_config=E09B911LossConfig(
            lambda_depth=2,lambda_amplitude=1,
            lambda_body_density=1,lambda_gravity=0,
        ),
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),jit_compile=False)
    terms=model.compute_loss_terms(gravity,truth,training=False)
    assert float(terms[8].numpy())==0.0
    assert float(terms[9].numpy())==0.0
    logs=model.train_step((gravity,truth))
    assert "body_density_loss" in logs
    assert "gravity_loss" not in logs


def test_preflight_reports_finite_nonzero_component_gradients():
    gravity,truth=batch();model=wrapper(body=1,gravity=.001)
    diagnostics=run_e09b911_preflight(model,gravity,truth)
    assert diagnostics["forward_operator_trainable_variables"]==0
    assert diagnostics["body_density_gradient_norm"]>0
    assert diagnostics["gravity_gradient_norm"]>0
    assert diagnostics["output_shape"]==[1,24,64,64,1]


def test_combined_wrapper_runs_optimizer_step_and_tracks_new_terms():
    gravity,truth=batch();model=wrapper(body=1,gravity=.001)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    before=[value.numpy().copy() for value in model.inversion_model.trainable_variables]
    logs=model.train_step((gravity,truth))
    after=[value.numpy() for value in model.inversion_model.trainable_variables]
    assert any(np.any(old!=new) for old,new in zip(before,after))
    for key in ("body_density_loss","gravity_loss","weighted_gravity_loss",
                "gravity_rmse","gravity_correlation"):
        assert key in logs and np.isfinite(float(logs[key].numpy()))
