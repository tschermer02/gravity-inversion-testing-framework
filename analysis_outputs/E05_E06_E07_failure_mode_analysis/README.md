# E05 / E06 / E07 Failure-Mode Analysis

## What was analyzed

The same 100 canonical held-out samples were matched by sample ID. Saved prediction metrics, saved density arrays, and saved gravity-consistency CSVs were analyzed at density threshold 0.1. Density sums are reported only as mass proxies.

No model was loaded, no dataset was regenerated, and no prediction or forward-gravity calculation was run.

## E05 -> E06 findings

Mean MSE changed -60.0%, mean Dice +75.4%, and mean gravity RMSE -43.9%. Lower MSE occurred for 100.0% of samples and higher Dice for 100.0%.

| Metric | Percent change |
|---|---:|
| Mean MSE | -60.0% |
| Mean MAE | -56.6% |
| Mean IoU | +91.6% |
| Mean Dice | +75.4% |
| Median volume-ratio error | -53.6% |
| Median mass-ratio error | -61.7% |
| Mean absolute top-depth error | -24.2% |
| Mean absolute bottom-depth error | -71.2% |
| Mean absolute thickness error | -61.6% |
| Mean saved gravity RMSE | -43.9% |

| Paired sample criterion | Improved |
|---|---:|
| Lower MSE | 100.0% |
| Higher Dice | 100.0% |
| Volume ratio closer to 1 | 100.0% |
| Mass ratio closer to 1 | 100.0% |
| Better top depth | 59.0% |
| Better bottom depth | 100.0% |
| Better thickness | 100.0% |

## E06 -> E07 findings

Mean MSE changed -16.7%, mean Dice +12.2%, and mean gravity RMSE -12.7%. Lower MSE occurred for 74.0% of samples and higher Dice for 75.0%.

| Metric | Percent change |
|---|---:|
| Mean MSE | -16.7% |
| Mean MAE | -12.1% |
| Mean IoU | +14.4% |
| Mean Dice | +12.2% |
| Median volume-ratio error | -15.8% |
| Median mass-ratio error | -8.4% |
| Mean absolute top-depth error | -7.2% |
| Mean absolute bottom-depth error | +21.3% |
| Mean absolute thickness error | +9.8% |
| Mean saved gravity RMSE | -12.7% |

| Paired sample criterion | Improved |
|---|---:|
| Lower MSE | 74.0% |
| Higher Dice | 75.0% |
| Volume ratio closer to 1 | 76.0% |
| Mass ratio closer to 1 | 75.0% |
| Better top depth | 35.0% |
| Better bottom depth | 6.0% |
| Better thickness | 21.0% |

## E05 -> E07 paired result

Mean MSE changed -66.7%, mean Dice +96.8%, and mean gravity RMSE -51.0%. Lower MSE occurred for 100.0% of samples and higher Dice for 100.0%.

| Metric | Percent change |
|---|---:|
| Mean MSE | -66.7% |
| Mean MAE | -61.9% |
| Mean IoU | +119.3% |
| Mean Dice | +96.8% |
| Median volume-ratio error | -60.9% |
| Median mass-ratio error | -64.9% |
| Mean absolute top-depth error | -29.7% |
| Mean absolute bottom-depth error | -65.0% |
| Mean absolute thickness error | -57.8% |
| Mean saved gravity RMSE | -51.0% |

| Paired sample criterion | Improved |
|---|---:|
| Lower MSE | 100.0% |
| Higher Dice | 100.0% |
| Volume ratio closer to 1 | 100.0% |
| Mass ratio closer to 1 | 100.0% |
| Better top depth | 67.0% |
| Better bottom depth | 100.0% |
| Better thickness | 100.0% |

## Current E07 failure mode

Excess predicted volume is largest by normalized diagnostic score (4.647; median |volume ratio - 1|).

The ranking uses dimensionless, interpretable error indicators so unlike units are not compared directly. Gravity uses the already-saved relative-L2 result; RMSE and correlation are also preserved in the CSV outputs.

## Recommended target for E08

Test one explicit occupied-volume/sparsity regularizer while holding E07 fixed.
