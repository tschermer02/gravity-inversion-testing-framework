# E09B Density-Amplitude and Body-Size Ablation

This controlled follow-up uses completed E09B-2 (`lambda_depth=2.0`) as the
baseline for the 2x2 amplitude/small-body experiment. E09B-5 separately tests
only `lambda_depth=2.5` between E09B-2 and E09B-3.

The workflow runs all four training stages, then all predictions, then all
gravity-consistency analyses, and finally builds the combined comparison.
Valid completed outputs are verified and reused with `--resume`.

```bash
python -m cnn_inversion_3d.e09b_density_size_ablation --resume --overwrite
```

Training body-volume weights are derived only from `train_manifest.csv`.
Held-out small/medium/large groups are created once from the common test-set
ground truth and are used only for evaluation.
