"""Vertical-extent supervision layered only onto the E09B objective."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import tensorflow as tf
from cnn_inversion_3d.e09a_training import density_depth_profile
from cnn_inversion_3d.e09b_training import E09BLossConfig, E09BTrainingModel
from cnn_inversion_3d.single_plane_review import SinglePlaneReviewConfig

@dataclass(frozen=True)
class E09CLossConfig:
    lambda_density: float = 1.0; lambda_depth: float = 1.0; alpha_center: float = 1.0
    lambda_sensitivity: float = 1.0; sensitivity_gamma: float = 0.5
    sensitivity_weight_min: float = 0.5; sensitivity_weight_max: float = 5.0
    lambda_extent: float = 1.0; top_quantile: float = 0.05
    bottom_quantile: float = 0.95; boundary_sharpness: float = 100.0
    epsilon: float = 1e-8; body_fraction: float = 0.5
    def validate(self) -> None:
        for name in ("lambda_density","lambda_depth","alpha_center","lambda_sensitivity","sensitivity_gamma","lambda_extent"):
            if getattr(self,name) < 0: raise ValueError(f"{name} must not be negative.")
        if not 0 < self.top_quantile < self.bottom_quantile < 1: raise ValueError("Require 0 < top_quantile < bottom_quantile < 1.")
        if self.boundary_sharpness <= 0 or self.epsilon <= 0: raise ValueError("Sharpness and epsilon must be positive.")
        E09BLossConfig(lambda_density=self.lambda_density,lambda_depth=self.lambda_depth,alpha_center=self.alpha_center,
            lambda_sensitivity=self.lambda_sensitivity,sensitivity_gamma=self.sensitivity_gamma,
            sensitivity_weight_min=self.sensitivity_weight_min,sensitivity_weight_max=self.sensitivity_weight_max,
            epsilon=self.epsilon,body_fraction=self.body_fraction).validate()

def soft_vertical_boundaries(density: tf.Tensor, *, top_quantile=.05, bottom_quantile=.95,
    sharpness=100., epsilon=1e-8, geometry: SinglePlaneReviewConfig|None=None) -> tuple[tf.Tensor,tf.Tensor,tf.Tensor]:
    """Return differentiable top, bottom, thickness in meters via soft CDF quantiles."""
    cfg=geometry or SinglePlaneReviewConfig(); profile=density_depth_profile(density)
    p=profile/(tf.reduce_sum(profile,axis=1,keepdims=True)+tf.cast(epsilon,profile.dtype)); cdf=tf.cumsum(p,axis=1)
    def locate(q: float, edges: tf.Tensor) -> tf.Tensor:
        weights=tf.nn.softmax(-tf.cast(sharpness,cdf.dtype)*tf.square(cdf-tf.cast(q,cdf.dtype)),axis=1)
        return tf.reduce_sum(weights*edges[None,:],axis=1)
    centers=tf.cast(cfg.density_z_min_center_m+tf.range(cfg.nz,dtype=tf.float32)*cfg.dz_m,density.dtype)
    top=locate(top_quantile,centers-cfg.dz_m/2); bottom=locate(bottom_quantile,centers+cfg.dz_m/2)
    return top,bottom,bottom-top

def vertical_extent_losses(truth: tf.Tensor,prediction: tf.Tensor,*,top_quantile=.05,bottom_quantile=.95,
    sharpness=100.,epsilon=1e-8) -> tuple[tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor]:
    kwargs=dict(top_quantile=top_quantile,bottom_quantile=bottom_quantile,sharpness=sharpness,epsilon=epsilon)
    tt,tb,th=soft_vertical_boundaries(truth,**kwargs); pt,pb,ph=soft_vertical_boundaries(prediction,**kwargs)
    span=tf.cast(240.,prediction.dtype)
    top=tf.reduce_mean(tf.square((pt-tt)/span)); bottom=tf.reduce_mean(tf.square((pb-tb)/span)); thick=tf.reduce_mean(tf.square((ph-th)/span))
    return top,bottom,thick,(top+bottom+thick)/3

class E09CTrainingModel(E09BTrainingModel):
    def __init__(self,inversion_model:tf.keras.Model,sensitivity_weights:np.ndarray,*,loss_config:E09CLossConfig):
        loss_config.validate(); super().__init__(inversion_model,sensitivity_weights,loss_config=E09BLossConfig(
            lambda_density=loss_config.lambda_density,lambda_depth=loss_config.lambda_depth,alpha_center=loss_config.alpha_center,
            lambda_sensitivity=loss_config.lambda_sensitivity,sensitivity_gamma=loss_config.sensitivity_gamma,
            sensitivity_weight_min=loss_config.sensitivity_weight_min,sensitivity_weight_max=loss_config.sensitivity_weight_max,
            epsilon=loss_config.epsilon,body_fraction=loss_config.body_fraction))
        self._name="e09c_vertical_extent_wrapper"; self.loss_config=loss_config
        for name in ("extent_loss","top_boundary_loss","bottom_boundary_loss","thickness_loss"):
            self.trackers[name]=tf.keras.metrics.Mean(name=name)
    def compute_loss_terms(self,gravity,truth,*,training):
        base=super().compute_loss_terms(gravity,truth,training=training); cfg=self.loss_config
        top,bottom,thick,extent=vertical_extent_losses(truth,base[0],top_quantile=cfg.top_quantile,bottom_quantile=cfg.bottom_quantile,sharpness=cfg.boundary_sharpness,epsilon=cfg.epsilon)
        total=base[7]+cfg.lambda_extent*extent
        return (*base[:6],top,bottom,thick,extent,total)
    def _update(self,truth,terms):
        super()._update(truth,(*terms[:6],tf.constant(0.0,dtype=terms[-1].dtype),terms[-1]))
        for name,value in zip(("top_boundary_loss","bottom_boundary_loss","thickness_loss","extent_loss"),terms[6:10]): self.trackers[name].update_state(value)
