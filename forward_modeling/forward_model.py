from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

from synthetic_models.common.grid import GridSpec

def load_cppforward_module(cppforward_path: Path) -> ModuleType:
    """
    Load the repository's cppforward.py directly from its file path.

    This avoids package-import problems caused by the repository folder
    structure while still using the original Python/DLL implementation.
    """

    cppforward_path = cppforward_path.resolve()

    if not cppforward_path.exists():
        raise FileNotFoundError(
            f"Could not find cppforward.py:\n{cppforward_path}"
        )

    module_spec = importlib.util.spec_from_file_location(
        "repository_cppforward",
        cppforward_path,
    )

    if module_spec is None or module_spec.loader is None:
        raise ImportError(
            f"Could not create an import specification for:\n"
            f"{cppforward_path}"
        )

    cppforward = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cppforward)

    return cppforward

class GravityForwardModel:
    """
    Experiment-facing interface to the repository's DLL-backed
    gravity forward model.
    """

    def __init__(
        self,
        grid: GridSpec,
        cppforward_path: Path,
    ) -> None:
        self.grid = grid

        self._cppforward = load_cppforward_module(
            cppforward_path=cppforward_path,
        )

        print("Building gravity kernel...")

        self._kernel = self._cppforward.gravity_forward_Va(
            grid.nx,
            grid.ny,
            grid.nz,
            grid.dx,
            grid.dy,
            grid.dz,
        )

        print("Gravity kernel is ready.")

    def calculate(self, model: np.ndarray) -> np.ndarray:
        """
        Calculate the surface gravity anomaly for one density model.

        Expected density-model order:
            model[z, y, x]

        Returned gravity-data order:
            anomaly[y, x]
        """

        expected_shape = (
            self.grid.nz,
            self.grid.ny,
            self.grid.nx,
        )

        if model.shape != expected_shape:
            raise ValueError(
                f"Expected density model shape {expected_shape}, "
                f"but received {model.shape}."
            )

        if not np.issubdtype(model.dtype, np.floating):
            raise TypeError(
                "Density model must contain floating-point values. "
                f"Received dtype {model.dtype}."
            )

        if not np.all(np.isfinite(model)):
            raise ValueError(
                "Density model contains NaN or infinite values."
            )

        model_for_cpp = np.ascontiguousarray(
            model,
            dtype=np.float32,
        )

        anomaly = self._cppforward.gravity_forward(
            model_for_cpp,
            self._kernel,
            self.grid.nx,
            self.grid.ny,
            self.grid.nz,
        )

        anomaly = np.asarray(
            anomaly,
            dtype=np.float32,
        )

        expected_anomaly_shape = (
            self.grid.ny,
            self.grid.nx,
        )

        if anomaly.shape != expected_anomaly_shape:
            raise ValueError(
                f"Expected gravity anomaly shape "
                f"{expected_anomaly_shape}, "
                f"but received {anomaly.shape}."
            )

        if not np.all(np.isfinite(anomaly)):
            raise ValueError(
                "Calculated gravity anomaly contains "
                "NaN or infinite values."
            )

        return anomaly