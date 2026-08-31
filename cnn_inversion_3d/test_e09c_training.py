import numpy as np
import tensorflow as tf
from cnn_inversion_3d.dataset import DENSITY_SHAPE,SINGLE_PLANE_GRAVITY_SHAPE
from cnn_inversion_3d.e09b_training import build_e09b_sensitivity_weights
from cnn_inversion_3d.e09c_training import E09CLossConfig,E09CTrainingModel,vertical_extent_losses
from cnn_inversion_3d.model import ModelConfig,build_asymmetric_2d_unet_model
from cnn_inversion_3d.train import build_inversion_model_for_architecture

def body(z0=3,nz=3):
    x=np.zeros((1,*DENSITY_SHAPE),np.float32); x[:,z0:z0+nz,28:34,30:36,:]=.6; return tf.constant(x)
def test_extent_perfect_near_zero_and_differentiable():
    truth=body(); pred=tf.Variable(truth)
    with tf.GradientTape() as tape: losses=vertical_extent_losses(truth,pred)
    assert float(losses[-1]) < 1e-10 and tape.gradient(losses[-1],pred) is not None
def test_deeper_and_thicker_increase_expected_losses():
    truth=body(); deeper=vertical_extent_losses(truth,body(6,3)); thicker=vertical_extent_losses(truth,body(3,6))
    assert float(deeper[0]+deeper[1]) > 0 and float(thicker[2]) > 0
def test_selector_preserves_e09_architecture():
    a=build_asymmetric_2d_unet_model(ModelConfig(base_filters=1)); c=build_inversion_model_for_architecture("single_plane_asymmetric_2d_unet_extent_loss",ModelConfig(base_filters=1))
    assert a.to_json()==c.to_json() and c.output_shape==(None,*DENSITY_SHAPE)
def test_e09c_total_adds_extent_and_gradients_reach_model():
    _,w=build_e09b_sensitivity_weights(); inv=build_asymmetric_2d_unet_model(ModelConfig(base_filters=1)); wrapper=E09CTrainingModel(inv,w,loss_config=E09CLossConfig())
    g=tf.zeros((1,*SINGLE_PLANE_GRAVITY_SHAPE)); truth=body()
    with tf.GradientTape() as tape: terms=wrapper.compute_loss_terms(g,truth,training=True)
    grads=tape.gradient(terms[-1],inv.trainable_variables)
    np.testing.assert_allclose(terms[-1],terms[1]+terms[4]+terms[5]+terms[9],rtol=1e-6)
    assert any(x is not None for x in grads) and np.isfinite(float(terms[-1]))
