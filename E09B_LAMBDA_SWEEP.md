# E09B Lambda Sweep

This controlled experiment changes only `lambda_depth` and
`lambda_sensitivity` in the existing E09B objective. It runs all four training
jobs first, then all predictions, then all analyses, and finally compares them
with the existing E09B baseline.

```bash
env -u LD_LIBRARY_PATH TF_FORCE_GPU_ALLOW_GROWTH=true \
python -m cnn_inversion_3d.e09b_lambda_sweep
```

Outputs use `training_outputs/E09B_1...E09B_4`, matching prediction folders,
and `analysis_outputs/E09B_lambda_sweep/`. To resume, use `--resume`; completed
stages are skipped only after their saved configuration is verified. Use
`--overwrite` only to intentionally replace all four sweep outputs. The
original `E09B_integrated_sensitivity` baseline is never trained or overwritten.
