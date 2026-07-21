from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_REPOSITORY = (
    PROJECT_ROOT
    / "original_repository"
    / "CNN_gravity_inversion_com-geo"
)

WORKING_REPOSITORY = (
    PROJECT_ROOT
    / "modified_code"
    / "CNN_gravity_inversion_working"
)

REPRODUCED_EXAMPLE = PROJECT_ROOT / "reproduced_example"
SYNTHETIC_MODELS = PROJECT_ROOT / "synthetic_models"
FORWARD_MODELING = PROJECT_ROOT / "forward_modeling"
EVALUATION = PROJECT_ROOT / "evaluation"
RESULTS = PROJECT_ROOT / "results"
NOTES = PROJECT_ROOT / "notes"


def print_paths() -> None:
    print("Project root:", PROJECT_ROOT)
    print("Original repository:", ORIGINAL_REPOSITORY)
    print("Working repository:", WORKING_REPOSITORY)
    print("Results folder:", RESULTS)


if __name__ == "__main__":
    print_paths()