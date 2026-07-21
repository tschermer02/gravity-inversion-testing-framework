from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ExperimentPaths:
    """Paths used by one synthetic-model experiment."""

    experiment_directory: Path
    true_models: Path
    forward_responses: Path
    figures: Path
    metrics: Path
    cnn_outputs: Path

    cppforward: Path
    inv_model: Path
    cnn_weights: Path

def build_experiment_paths(
    experiment_directory: Path,
) -> ExperimentPaths:
    """Construct input and output paths for one experiment folder."""

    experiment_directory = experiment_directory.resolve()

    project_root = Path(__file__).resolve().parents[2]

    return ExperimentPaths(
        experiment_directory=experiment_directory,

        true_models=experiment_directory / "true_models",
        forward_responses=experiment_directory / "forward_responses",
        figures=experiment_directory / "figures",
        metrics=experiment_directory / "metrics",
        cnn_outputs=experiment_directory / "cnn_outputs",

        cppforward=(
            project_root
            / "modified_code"
            / "CNN_gravity_inversion"
            / "cppforward.py"
        ),

        inv_model=(
            project_root
            / "modified_code"
            / "CNN_gravity_inversion"
            / "inv_model.py"
        ),

        cnn_weights=(
            project_root
            / "modified_code"
            / "test_data"
            / "3Dmodel_weight3by3.h5"
        ),
    )