# E09B Density and Body-Size Ablation

E09B-2 is the controlled baseline. The common 100-sample test set is ranked once
by true volume into exactly 25 small, 50 medium, and 25 large bodies (quantile
values: q25=193.5, q75=604 cells). Test quantiles are never used in training.

## Direct answers

1. Depth refinement: E09B-5 Dice minus E09B-2 = -0.03211; versus E09B-3 = -0.02120.
2. Best overlap/depth compromise by the four-objective rank used here: E09B-6.
3. E09B-6 density MAE change from E09B-2: -0.0112374 g/cm^3.
4. E09B-6 density-error/gravity-RMSE correlation: -0.014; association does not establish causation.
5. E09B-6 Dice change from E09B-2: +0.04583.
6. E09B-2 small-minus-large Dice: -0.41228.
7. E09B-7 small-body Dice change from E09B-2: -0.00382.
8. E09B-7 small-body top-depth MAE change: +0.000 m.
9. E09B-7 small-body density MAE change: +0.00304995 g/cm^3.
10. E09B-7 small-body gravity RMSE change: +0.00163228 mGal.
11. Medium/large Dice changes for E09B-7: -0.01230 / -0.00763.
12. E09B-8 small-body Dice minus E09B-6: -0.03154.
13. E09B-8 density MAE minus E09B-7: -0.0169348 g/cm^3.
14. Combined intervention classification: **conflicting** (0/4 objectives outperform both single interventions).

## Objective leaders

- Best IoU/Dice: E09B-6
- Best combined vertical-depth errors: E09B-6
- Best density-amplitude recovery: E09B-6
- Best gravity consistency by RMSE: E09B-6
- Best small-body Dice: E09B-6
- Best four-objective rank compromise: E09B-6

Density bias and its under/overprediction direction are reported overall and by
body-size group. Correlations are descriptive associations, not proof of causation.

## Run everything

```bash
python -m cnn_inversion_3d.e09b_density_size_ablation --resume --overwrite
```
