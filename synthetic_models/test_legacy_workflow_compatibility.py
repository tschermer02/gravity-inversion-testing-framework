from __future__ import annotations

import importlib
import inspect

import pytest

from synthetic_models.common.experiment_runner import (
    run_experiment,
)
from synthetic_models.common.generators import (
    generate_single_body,
)
from synthetic_models.common.paths import (
    build_experiment_paths,
)


EXPERIMENT_MODULES = (
    "synthetic_models.01_single_compact_body.run",
    "synthetic_models.02_multiple_depths.run",
    "synthetic_models.03_dipping_body.run",
    "synthetic_models.04_salt_dome.run",
    "synthetic_models.05_basement_relief.run",
    "synthetic_models.06_noise_tests.run",
)


@pytest.mark.parametrize(
    "module_name",
    EXPERIMENT_MODULES,
)
def test_legacy_experiment_imports(
    module_name: str,
) -> None:
    """Verify that synthetic experiments 01 through 06 still import."""

    module = importlib.import_module(
        module_name
    )

    assert callable(
        module.main
    )


def test_legacy_shared_interfaces_do_not_require_new_arguments() -> None:
    """Verify established synthetic workflow call signatures."""

    assert "evaluate_gravity_consistency" not in inspect.signature(
        run_experiment
    ).parameters
    assert "evaluate_gravity_consistency" not in inspect.signature(
        generate_single_body
    ).parameters
    assert "evaluate_gravity_consistency" not in inspect.signature(
        build_experiment_paths
    ).parameters
