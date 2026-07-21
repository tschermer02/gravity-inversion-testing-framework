# Gravity Inversion Testing Framework

A modular Python framework for evaluating a published pretrained CNN for 3D gravity inversion.

The framework generates synthetic density models, calculates their gravity responses, applies the pretrained CNN, and compares the recovered density model with the known true model. The goal is to evaluate the pretrained network, not retrain it.

## Workflow

Each experiment follows the same general process:

1. Generate a synthetic density model.
2. Forward model its gravity response.
3. Apply the pretrained CNN inversion.
4. Compare the recovered and true density models.
5. Forward model the recovered density model.
6. Compare the original and reconstructed gravity fields.
7. Save metrics, models, and figures.

## Current Experiments

- **Single compact body**  
  Tests isolated rectangular density bodies at different locations and depths.

- **Multiple bodies at different depths**  
  Tests whether the CNN can separate overlapping gravity signals from multiple sources.

- **Dipping bodies**  
  Tests elongated and inclined structures with more realistic geological orientations.

- **Salt-dome-like structures**  
  Tests curved geometries and both positive and negative density contrasts.

- **Basement relief**  
  Tests broad, laterally continuous basement interfaces and uplifts.

- **Noise robustness**  
  Tests reconstruction quality under increasing levels of Gaussian noise.

## Repository Structure

```text
gravity-inversion-testing-framework/
├── cnn_inversion/
├── environments/
├── evaluation/
│   ├── metrics.py
│   ├── plotting.py
│   ├── project_paths.py
│   └── visualization.py
├── forward_modeling/
├── modified_code/
├── notes/
├── original_repository/
├── results/
├── synthetic_models/
│   ├── 01_single_compact_body/
│   ├── 02_multiple_depths/
│   ├── 03_dipping_body/
│   ├── 04_salt_dome/
│   ├── 05_basement_relief/
│   ├── 06_noise_tests/
│   └── common/
│       ├── bodies.py
│       ├── cnn_pipeline.py
│       ├── experiment_runner.py
│       ├── forward_pipeline.py
│       ├── generators.py
│       ├── grid.py
│       ├── model_io.py
│       ├── noise.py
│       ├── output.py
│       ├── paths.py
│       └── validation.py
├── .gitignore
└── test_env.py
