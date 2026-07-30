# CNN 3D Gravity Inversion

The CNN workflow separates density-space reconstruction quality from
gravity-space forward consistency:

```text
Known density model
        ↓
Validated FWD3D gravity model
        ↓
Synthetic 3D gravity anomaly
        ↓
CNN preprocessing and training or inference
        ↓
Recovered density model
        ↓
Validated FWD3D gravity model
        ↓
Recovered gravity anomaly
        ↓
True gravity vs recovered gravity vs residual
```

The CNN gravity residual convention is:

```text
gravity_residual = recovered_gravity - true_gravity
```

The optional consistency stage reconstructs its model grid, receiver
elevations, coordinate convention, array ordering, channel, and units from
the dataset metadata. It calls the same translated MATLAB FWD3D solver used
for dataset generation. It does not implement a second physical solver.

## Running gravity-consistency analysis

First create density predictions with `cnn_inversion_3d.predict`. Then run:

```powershell
python -m cnn_inversion_3d.analyze_predictions `
    --dataset datasets/fwd3d_matlab_edge_rectangular_baseline `
    --predictions prediction_outputs/E01_edge_aligned_transpose_bf8_balanced `
    --metrics prediction_metrics.json `
    --evaluate-gravity-consistency `
    --save-gravity-volumes `
    --gravity-comparison-receivers 0 3 7 `
    --overwrite
```

Without `--evaluate-gravity-consistency`, analysis retains its previous
density-only behavior. Existing consistency results are cached and skipped;
use `--overwrite` to recompute them.

Per-sample outputs are written under:

```text
prediction_outputs/<experiment>/gravity_consistency/<sample>/
```

Each sample receives a metrics JSON and gravity comparison figure. With
`--save-gravity-volumes`, it also receives:

- `true_gravity.npy`
- `recovered_gravity.npy`
- `gravity_residual.npy`

The prediction directory receives `gravity_consistency_metrics.csv`.
Gravity metrics are also added to `combined_test_metrics.csv`, and aggregate
statistics are stored in the `gravity_consistency` section of
`test_analysis.json`.

## Scientific interpretation

Density-space and gravity-space agreement answer different questions.
Density metrics measure similarity to the known synthetic subsurface.
Gravity-consistency metrics measure whether the recovered density reproduces
the observed data under the validated forward model.

Gravity inversion is non-unique. A recovered density can differ from the
known density while producing similar gravity. Therefore, a low gravity
residual does not prove that the recovered density is geologically correct.
A high gravity residual does show that the predicted density fails to
reproduce the input gravity under the stated physical configuration.

Synthetic-model experiments 01 through 06 retain their existing workflows,
interfaces, plots, metrics, and outputs. The CNN consistency stage is
separate and opt-in.
