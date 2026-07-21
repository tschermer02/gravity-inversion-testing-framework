from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from cnn_inversion.cnn_predictor import CNNGravityInverter
from evaluation.plotting import plot_gravity_comparison, plot_noise_robustness_summary
from forward_modeling.forward_model import GravityForwardModel
from synthetic_models.common.bodies import (
    RectangularBodySpec,
    build_density_model,
)
from synthetic_models.common.cnn_pipeline import (
    run_cnn_inversion,
)
from synthetic_models.common.generators import (
    generate_single_body,
)
from synthetic_models.common.grid import GridSpec
from synthetic_models.common.model_io import (
    save_metrics_csv,
)
from synthetic_models.common.noise import (
    GaussianNoiseSpec,
    add_gaussian_noise,
    summarize_noise,
)
from synthetic_models.common.paths import (
    build_experiment_paths,
)

MetricValue = str | float | int
CaseMetrics = dict[str, MetricValue]

def define_noise_specifications(
) -> tuple[GaussianNoiseSpec, ...]:
    """Define the controlled noise-amplitude sweep."""

    return (
        GaussianNoiseSpec(
            name="noise_00_percent",
            noise_fraction=0.00,
            seed=42,
        ),
        GaussianNoiseSpec(
            name="noise_01_percent",
            noise_fraction=0.01,
            seed=42,
        ),
        GaussianNoiseSpec(
            name="noise_02_percent",
            noise_fraction=0.02,
            seed=42,
        ),
        GaussianNoiseSpec(
            name="noise_05_percent",
            noise_fraction=0.05,
            seed=42,
        ),
        GaussianNoiseSpec(
            name="noise_10_percent",
            noise_fraction=0.10,
            seed=42,
        ),
    )

def define_test_body(
    *,
    grid: GridSpec,
) -> RectangularBodySpec:
    """Define the common true model for every noise level."""

    return generate_single_body(
        name="noise_test_compact_body",
        x_start=27,
        y_start=27,
        z_start=9,
        x_width=10,
        y_width=10,
        z_thickness=5,
        density_contrast=0.5,
        grid=grid,
    )

def validate_gravity_array(
    *,
    gravity: np.ndarray,
    grid: GridSpec,
    label: str,
) -> np.ndarray:
    """Validate and normalize a gravity array representation."""

    expected_shape = (
        grid.ny,
        grid.nx,
    )

    normalized = np.ascontiguousarray(
        gravity,
        dtype=np.float32,
    )

    if normalized.shape != expected_shape:
        raise ValueError(
            f"{label}: expected shape {expected_shape}, "
            f"but received {normalized.shape}."
        )

    if not np.all(np.isfinite(normalized)):
        raise ValueError(
            f"{label}: contains NaN or infinite values."
        )

    return normalized

def main() -> None:
    """
    Test CNN reconstruction robustness under added Gaussian noise.

    Every noise level uses:

    - The same true density model.
    - The same clean gravity response.
    - The same normalized random-noise pattern.
    - A different noise amplitude.
    """

    grid = GridSpec()

    paths = build_experiment_paths(
        experiment_directory=(
            Path(__file__).resolve().parent
        ),
    )

    base_body = define_test_body(
        grid=grid,
    )

    forward_model = GravityForwardModel(
        grid=grid,
        cppforward_path=paths.cppforward,
    )

    cnn_inverter = CNNGravityInverter(
        grid=grid,
        inv_model_path=paths.inv_model,
        weights_path=paths.cnn_weights,
    )

    true_model = build_density_model(
        grid=grid,
        case=base_body,
    )

    clean_gravity = forward_model.calculate(
        model=true_model,
    )

    clean_gravity = validate_gravity_array(
        gravity=clean_gravity,
        grid=grid,
        label="clean_gravity",
    )

    paths.forward_responses.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_gravity_path = (
        paths.forward_responses
        / "noise_test_clean_gravity.npy"
    )

    np.save(
        clean_gravity_path,
        clean_gravity,
    )

    noisy_gravity_maps: dict[str, np.ndarray] = {
        "clean": clean_gravity,
    }

    all_metrics: list[CaseMetrics] = []

    noise_specifications = (
        define_noise_specifications()
    )

    for case_number, specification in enumerate(
        noise_specifications,
        start=1,
    ):
        print(
            f"\nNoise case {case_number} of "
            f"{len(noise_specifications)}: "
            f"{specification.name}"
        )

        specification.validate()

        noisy_gravity, noise = add_gaussian_noise(
            gravity=clean_gravity,
            specification=specification,
        )

        noisy_gravity = validate_gravity_array(
            gravity=noisy_gravity,
            grid=grid,
            label=(
                f"{specification.name} noisy gravity"
            ),
        )

        noise = validate_gravity_array(
            gravity=noise,
            grid=grid,
            label=(
                f"{specification.name} noise"
            ),
        )

        noise_summary = summarize_noise(
            clean_gravity=clean_gravity,
            noisy_gravity=noisy_gravity,
            noise=noise,
            specification=specification,
        )

        # Give every noise realization a unique case name so that
        # recovered models, gravity arrays, and figures are not
        # overwritten.
        noisy_case = replace(
            base_body,
            name=(
                f"{base_body.name}_"
                f"{specification.name}"
            ),
        )

        noise_path = (
            paths.forward_responses
            / f"{noisy_case.name}_noise.npy"
        )

        noisy_gravity_path = (
            paths.forward_responses
            / f"{noisy_case.name}_noisy_gravity.npy"
        )

        np.save(
            noise_path,
            noise,
        )

        np.save(
            noisy_gravity_path,
            noisy_gravity,
        )

        cnn_metrics = run_cnn_inversion(
            body=noisy_case,
            true_model=true_model,
            gravity_anomaly=noisy_gravity,
            clean_reference_gravity=clean_gravity,
            noise_percent=float(
                noise_summary["noise_percent"]
            ),
            grid=grid,
            forward_model=forward_model,
            cnn_inverter=cnn_inverter,
            paths=paths,
        )

        combined_metrics: CaseMetrics = {
            "case_name": noisy_case.name,
            **noise_summary,
            **cnn_metrics,
        }

        all_metrics.append(
            combined_metrics
        )

        noisy_gravity_maps[
            f"{noise_summary['noise_percent']:.0f}% noise"
        ] = noisy_gravity

    metrics_path = (
        paths.metrics
        / "noise_cnn_robustness_metrics.csv"
    )

    save_metrics_csv(
        case_metrics=all_metrics,
        output_path=metrics_path,
    )

    robustness_figure_path = (
        paths.figures
        / "cnn_noise_robustness_summary.png"
    )
    

    plot_noise_robustness_summary(
        metrics_csv=metrics_path,
        output_path=robustness_figure_path,
    )

    plot_gravity_comparison(
        anomalies=noisy_gravity_maps,
        grid=grid,
        output_path=(
            paths.figures
            / "noise_input_gravity_comparison.png"
        ),
    )

    print("\nNoise robustness experiment complete.")
    print(
        f"  Noise cases completed: "
        f"{len(all_metrics)}"
    )
    print(
        f"  Clean gravity: "
        f"{clean_gravity_path}"
    )
    print(
        f"  Combined metrics: "
        f"{metrics_path}"
    )


if __name__ == "__main__":
    main()