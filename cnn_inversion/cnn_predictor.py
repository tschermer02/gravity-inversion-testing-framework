from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import tensorflow as tf

from synthetic_models.common.grid import GridSpec

def _load_module_from_path(
    module_name: str,
    module_path: Path,
) -> ModuleType:
    """Load a Python module directly from a source-file path."""

    module_path = module_path.resolve()

    if not module_path.is_file():
        raise FileNotFoundError(
            f"Python module file not found: {module_path}"
        )

    module_spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )

    if module_spec is None or module_spec.loader is None:
        raise ImportError(
            f"Could not create an import specification for "
            f"{module_path}."
        )

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module

def prepare_cnn_inputs(
    gravity_anomaly: np.ndarray,
    grid: GridSpec,
) -> tuple[np.ndarray, np.ndarray]:

    expected_shape = (
        grid.ny,
        grid.nx,
    )

    anomaly = np.asarray(
        gravity_anomaly,
        dtype=np.float32,
    )

    if anomaly.shape != expected_shape:
        raise ValueError(
            f"Expected gravity anomaly shape {expected_shape}, "
            f"but received {anomaly.shape}."
        )

    if not np.all(np.isfinite(anomaly)):
        raise ValueError(
            "Gravity anomaly contains NaN or infinite values."
        )

    if np.isclose(grid.dz, 0.0):
        raise ValueError(
            "Grid dz must be nonzero for CNN preprocessing."
        )

    # Preserve the preprocessing used by the authors.
    scaled_anomaly = anomaly / np.float32(grid.dz)

    anomaly_input = np.ascontiguousarray(
        scaled_anomaly[np.newaxis, np.newaxis, :, :],
        dtype=np.float32,
    )

    spacing_input = np.ascontiguousarray(
        np.array(
            [[
                grid.dx / grid.dz,
                grid.dy / grid.dz,
            ]],
            dtype=np.float32,
        )
    )

    return anomaly_input, spacing_input

class CNNGravityInverter:
    """Wrapper around the authors' pretrained gravity-inversion CNN."""

    def __init__(
        self,
        grid: GridSpec,
        inv_model_path: Path,
        weights_path: Path,
    ) -> None:
        self.grid = grid
        self.inv_model_path = inv_model_path.resolve()
        self.weights_path = weights_path.resolve()

        if not self.inv_model_path.is_file():
            raise FileNotFoundError(
                "CNN architecture file not found: "
                f"{self.inv_model_path}"
            )

        if not self.weights_path.is_file():
            raise FileNotFoundError(
                "CNN weights file not found: "
                f"{self.weights_path}"
            )

        inv_model_module = _load_module_from_path(
            module_name="repository_inv_model",
            module_path=self.inv_model_path,
        )

        architecture_function = getattr(
            inv_model_module,
            "inv_model7_3by3",
            None,
        )

        if architecture_function is None:
            raise AttributeError(
                f"{self.inv_model_path} does not define "
                "inv_model7_3by3()."
            )

        strategy = tf.distribute.get_strategy()

        with strategy.scope():
            self._model = architecture_function()
            self._model.load_weights(
                str(self.weights_path)
            )

        self._validate_model_interface()

        print("CNN architecture loaded: inv_model7_3by3")
        print(f"CNN weights loaded from: {self.weights_path}")

    def _validate_model_interface(self) -> None:
        """Check that the loaded network has the expected inputs and output."""

        if len(self._model.inputs) != 2:
            raise ValueError(
                "The CNN must have two inputs: gravity anomaly and "
                "grid-spacing ratios."
            )

        expected_output_shape = (
            None,
            self.grid.nz,
            self.grid.ny,
            self.grid.nx,
        )

        actual_output_shape = tuple(
            self._model.output_shape
        )

        if actual_output_shape != expected_output_shape:
            raise ValueError(
                "Unexpected CNN output shape. "
                f"Expected {expected_output_shape}, "
                f"but the model reports {actual_output_shape}."
            )

    def predict(
        self,
        gravity_anomaly: np.ndarray,
    ) -> np.ndarray:

        anomaly_input, spacing_input = prepare_cnn_inputs(
            gravity_anomaly=gravity_anomaly,
            grid=self.grid,
        )

        prediction = self._model.predict(
            [
                anomaly_input,
                spacing_input,
            ],
            batch_size=1,
            verbose=0,
        )

        prediction = np.asarray(
            prediction,
            dtype=np.float32,
        )

        expected_prediction_shape = (
            1,
            self.grid.nz,
            self.grid.ny,
            self.grid.nx,
        )

        if prediction.shape != expected_prediction_shape:
            raise ValueError(
                "Unexpected CNN prediction shape. "
                f"Expected {expected_prediction_shape}, "
                f"but received {prediction.shape}."
            )

        if not np.all(np.isfinite(prediction)):
            raise ValueError(
                "CNN prediction contains NaN or infinite values."
            )

        recovered_model = np.ascontiguousarray(
            prediction[0],
            dtype=np.float32,
        )

        return recovered_model