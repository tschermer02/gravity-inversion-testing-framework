# E09B Density–Physics Ablation

This controlled 2x2 series keeps E09B-6-prime's dataset, unchanged
190,592-parameter E09 architecture, preprocessing, optimizer, learning rate,
batch size, seed, and existing loss weights fixed.

| Experiment | Body-density loss | Gravity loss |
|---|---:|---:|
| E09B-6' | 0 | 0 |
| E09B-9 | 1 | 0 |
| E09B-10 | 0 | 0.001 |
| E09B-11 | 1 | 0.001 |

The gravity term uses the existing fixed `DifferentiableSinglePlaneGz` and
global normalized gravity MSE. It adds no trainable parameters.

Run the complete train-then-predict-then-analyze workflow on CHPC:

```bash
python -m cnn_inversion_3d.e09b_density_physics_ablation --resume --overwrite
```
