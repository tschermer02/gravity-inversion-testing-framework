# E09B-6-prime

E09B-6-prime is a one-variable follow-up to E09B-6. It keeps the unchanged
190,592-parameter E09 model and the exact E09B-6 objective, optimizer, seed
convention, and training settings. Only training/validation dataset size and
coverage change.

The generated dataset contains 10,000 balanced training samples, 1,000 balanced
validation samples, and byte-for-byte copies of the original 100 held-out test
sample files. Size strata are defined without test labels from all valid source
dimension combinations: small `32–288`, medium `289–560`, and large `561–2048`
cells. Each size group is crossed with low/medium/high strata spanning the
unchanged `0.2–1.0 g/cm³` density range.

Run the complete sequential workflow on CHPC:

```bash
python -m cnn_inversion_3d.e09b6prime_workflow --resume --overwrite
```
