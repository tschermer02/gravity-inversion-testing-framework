from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


NOTES_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = NOTES_ROOT.parent

SYNTHETIC_ROOT = PROJECT_ROOT / "synthetic_models"
REPRODUCED_EXAMPLE_ROOT = NOTES_ROOT / "reproduced_example"
RESULTS_ROOT = NOTES_ROOT / "results"

OUTPUT_PATH = NOTES_ROOT / "report.md"

EXPERIMENTS = (
    {
        "title": "Single Compact Density Body",
        "folder": "01_single_compact_body",
        "description": (
            "This baseline experiment tested whether the pretrained CNN could "
            "recover the location and broad geometry of a simple isolated "
            "density anomaly."
        ),
        "figure_patterns": (
            "*true_recovered_comparison*.png",
            "*workflow*.png",
            "*comparison*.png",
            "*.png",
        ),
        "noise": False,
    },
    {
        "title": "Multiple Bodies at Different Depths",
        "folder": "02_multiple_depths",
        "description": (
            "This experiment tested whether the CNN could distinguish several "
            "density bodies and how reconstruction quality changed with depth."
        ),
        "figure_patterns": (
            "*true_recovered_comparison*.png",
            "*workflow*.png",
            "*comparison*.png",
            "*.png",
        ),
        "noise": False,
    },
    {
        "title": "Dipping Body",
        "folder": "03_dipping_body",
        "description": (
            "This experiment tested recovery of an elongated inclined body "
            "whose geometry differed from the compact training-style examples."
        ),
        "figure_patterns": (
            "*true_recovered_comparison*.png",
            "*workflow*.png",
            "*comparison*.png",
            "*.png",
        ),
        "noise": False,
    },
    {
        "title": "Salt-Dome-Like Structure",
        "folder": "04_salt_dome",
        "description": (
            "This experiment evaluated recovery of a rounded geological "
            "structure with curved boundaries."
        ),
        "figure_patterns": (
            "*true_recovered_comparison*.png",
            "*workflow*.png",
            "*comparison*.png",
            "*.png",
        ),
        "noise": False,
    },
    {
        "title": "Basement Relief",
        "folder": "05_basement_relief",
        "description": (
            "This experiment represented a laterally continuous geological "
            "interface and tested recovery of large-scale regional structure."
        ),
        "figure_patterns": (
            "*true_recovered_comparison*.png",
            "*workflow*.png",
            "*comparison*.png",
            "*.png",
        ),
        "noise": False,
    },
    {
        "title": "Noise Robustness",
        "folder": "06_noise_tests",
        "description": (
            "This experiment evaluated sensitivity to additive Gaussian noise "
            "at 0%, 1%, 2%, 5%, and 10% of the clean gravity signal scale."
        ),
        "figure_patterns": (
            "cnn_noise_robustness_summary.png",
            "*noise*robustness*summary*.png",
            "*summary*.png",
            "*.png",
        ),
        "noise": True,
    },
)

PREFERRED_METRICS = (
    "noise_percent",
    "model_correlation",
    "model_relative_l2",
    "model_iou",
    "model_dice",
    "center_error_cells",
    "volume_ratio",
    "clean_cnn_gravity_correlation",
    "clean_cnn_gravity_relative_l2",
)


def relative_path(path: Path) -> str:
    """Return a portable path relative to notes/report.md."""
    resolved_path = path.resolve()

    try:
        notes_relative = resolved_path.relative_to(
            NOTES_ROOT.resolve()
        )
        return notes_relative.as_posix()
    except ValueError:
        project_relative = resolved_path.relative_to(
            PROJECT_ROOT.resolve()
        )
        return (Path("..") / project_relative).as_posix()


def find_first_file(
    directories: Iterable[Path],
    patterns: Iterable[str],
) -> Path | None:
    """Find the first file using pattern priority."""
    for pattern in patterns:
        for directory in directories:
            if not directory.exists():
                continue

            matches = sorted(
                path
                for path in directory.rglob(pattern)
                if path.is_file()
            )

            if matches:
                return matches[0]

    return None


def find_metrics_csv(experiment_root: Path) -> Path | None:
    """Find the most likely metrics CSV in an experiment."""
    return find_first_file(
        directories=(
            experiment_root / "metrics",
            experiment_root,
        ),
        patterns=(
            "*metrics.csv",
            "*.csv",
        ),
    )


def readable_name(column: str) -> str:
    """Convert an internal metric name to a readable label."""
    replacements = {
        "model": "Density",
        "cnn": "CNN",
        "l2": "L2",
        "iou": "IoU",
        "rmse": "RMSE",
    }

    return " ".join(
        replacements.get(word.lower(), word.capitalize())
        for word in column.split("_")
    )


def format_value(raw_value: str) -> str:
    """Format a value for a concise Markdown table."""
    try:
        value = float(raw_value)
    except ValueError:
        return raw_value

    if value == float("inf"):
        return "∞"

    if value == float("-inf"):
        return "-∞"

    if abs(value) >= 1000:
        return f"{value:.3e}"

    return f"{value:.3f}"


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read all rows from a CSV."""
    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def build_metrics_table(
    csv_path: Path | None,
    *,
    noise_experiment: bool,
) -> str:
    """Create a compact Markdown table from one metrics CSV."""
    if csv_path is None:
        return "*Metrics CSV not found.*"

    rows = read_rows(csv_path)

    if not rows:
        return "*Metrics CSV contains no rows.*"

    if noise_experiment:
        selected_rows = rows
    else:
        selected_rows = rows[: min(5, len(rows))]

    available_columns = set(rows[0])

    identity_columns = [
        column
        for column in (
            "case_name",
            "body_name",
            "component_name",
        )
        if column in available_columns
    ]

    metric_columns = [
        column
        for column in PREFERRED_METRICS
        if column in available_columns
    ]

    columns = identity_columns[:1] + metric_columns[:6]

    if not columns:
        return (
            f"*Metrics found at `{relative_path(csv_path)}`, but no preferred "
            "summary columns were detected.*"
        )

    lines = [
        "| " + " | ".join(readable_name(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in selected_rows:
        lines.append(
            "| "
            + " | ".join(
                format_value(row.get(column, ""))
                for column in columns
            )
            + " |"
        )

    lines.extend(
        (
            "",
            f"*Full metrics: `{relative_path(csv_path)}`*",
        )
    )

    return "\n".join(lines)


def image_markdown(
    figure_path: Path | None,
    alt_text: str,
) -> str:
    """Create Markdown for an image or a visible missing-file note."""
    if figure_path is None:
        return f"*Representative figure not found for {alt_text}.*"

    return f"![{alt_text}]({relative_path(figure_path)})"


def find_reproduced_example_figure() -> Path | None:
    """Find a representative figure for the reproduced example."""
    return find_first_file(
        directories=(
            REPRODUCED_EXAMPLE_ROOT / "figures",
            REPRODUCED_EXAMPLE_ROOT,
            RESULTS_ROOT,
        ),
        patterns=(
            "*true_recovered_comparison*.png",
            "*gravity_fit_comparison*.png",
            "*workflow*.png",
            "*comparison*.png",
            "*.png",
        ),
    )


def build_report() -> str:
    """Build the complete Markdown progress report."""
    sections: list[str] = [
        "# Summer Progress Report",
        "## Development and Validation of a CNN-Based Gravity Inversion Testing Framework",
        "",
        "**Taylor Schermer**  ",
        "Department of Geology & Geophysics  ",
        "University of Utah  ",
        "Summer 2026",
        "",
        "# 1. Objective",
        "",
        (
            "The objective of this work was to reproduce the published "
            "CNN-based gravity inversion workflow and develop a reusable "
            "Python framework for independently evaluating the pretrained "
            "model on new synthetic geological models. The framework builds "
            "known density models, calculates their synthetic gravity "
            "responses, applies CNN inversion, compares recovered and true "
            "models, and checks whether the recovered models reproduce the "
            "input gravity data."
        ),
        "",
        "# 2. Framework Overview",
        "",
        (
            "Each experiment follows the same automated workflow: true-model "
            "generation, gravity forward modeling, CNN inversion, density-model "
            "comparison, recovered-gravity validation, figure generation, and "
            "metrics export. Common modules are reused across experiments so "
            "that new test cases can be added without duplicating the full "
            "pipeline."
        ),
        "",
        "# 3. Experimental Results",
        "",
        "## 3.1 Published Example Reproduction",
        "",
        (
            "The example supplied by the original authors was reproduced first "
            "to verify the network architecture, pretrained weights, input "
            "formatting, forward-modeling setup, and output density "
            "representation before independent synthetic testing."
        ),
        "",
        image_markdown(
            find_reproduced_example_figure(),
            "Published example reproduction",
        ),
        "",
    ]

    for index, experiment in enumerate(EXPERIMENTS, start=2):
        experiment_root = SYNTHETIC_ROOT / str(experiment["folder"])

        figure_path = find_first_file(
            directories=(
                experiment_root / "figures",
                experiment_root,
            ),
            patterns=experiment["figure_patterns"],
        )

        metrics_path = find_metrics_csv(experiment_root)

        sections.extend(
            (
                f"## 3.{index} {experiment['title']}",
                "",
                str(experiment["description"]),
                "",
                image_markdown(
                    figure_path,
                    str(experiment["title"]),
                ),
                "",
                "**Key metrics**",
                "",
                build_metrics_table(
                    metrics_path,
                    noise_experiment=bool(experiment["noise"]),
                ),
                "",
            )
        )

    sections.extend(
        (
            "# 4. Overall Findings",
            "",
            (
                "The published workflow was successfully reproduced, and a "
                "reusable framework was developed for systematic testing of "
                "the pretrained CNN on independently generated synthetic "
                "geological models."
            ),
            "",
            "- The CNN generally recovered the primary location and broad geometry of simple and moderately complex density structures.",
            "- Recovered density models were commonly smoother than the true models, especially near sharp boundaries.",
            "- Deeper, elongated, and irregular structures were more difficult to recover than compact shallow bodies.",
            "- Recovered models often reproduced the gravity data more accurately than they reproduced the true voxel-wise density distribution.",
            "- Reconstruction quality degraded progressively as Gaussian noise increased.",
            "- The framework now provides a consistent basis for testing additional synthetic models and alternative inversion approaches.",
            "",
            "# 5. Current Framework Capabilities",
            "",
            "- Synthetic density-model generation",
            "- Forward gravity modeling",
            "- CNN inversion using the published pretrained model",
            "- True-versus-recovered density comparison",
            "- Recovered-gravity validation",
            "- Automated quantitative metrics",
            "- Automated figure generation",
            "- Controlled Gaussian-noise experiments",
            "- Reusable experiment directory and output management",
            "",
            "# 6. Next Steps",
            "",
            (
                "The next phase will expand the distribution of synthetic "
                "models, evaluate more challenging geometries and physical "
                "properties, and investigate newer deep-learning methods that "
                "may improve reconstruction accuracy and generalization."
            ),
            "",
        )
    )

    return "\n".join(sections)


def main() -> None:
    """Generate notes/report.md from the current project outputs."""
    if not SYNTHETIC_ROOT.is_dir():
        raise FileNotFoundError(
            f"Synthetic-model directory not found: {SYNTHETIC_ROOT}"
        )

    report = build_report()

    OUTPUT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print(f"Generated report: {OUTPUT_PATH}")
    print("")
    print("Run this command from the project root:")
    print("  python -m notes.report")
    print("")
    print("Optional Word conversion:")
    print("  pandoc notes/report.md -o notes/report.docx")


if __name__ == "__main__":
    main()