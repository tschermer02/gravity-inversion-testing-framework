from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forward_modeling.matlab_fwd3d.receivers import ReceiverGrid
from dataset_generation.matlab_grid import MatlabCompatibleGridSpec


@dataclass(frozen=True)
class BodySamplingConfig:
    """
    Parameter ranges for one randomly generated rectangular body.

    All body dimensions and positions are expressed in model-cell indices.
    End indices follow normal Python slicing and are therefore exclusive.

    The baseline dataset intentionally contains:

    - one body per model
    - positive density contrast
    - no observational noise
    - no regularization
    """

    minimum_width_x: int = 4
    maximum_width_x: int = 16

    minimum_width_y: int = 4
    maximum_width_y: int = 16

    minimum_thickness_z: int = 2
    maximum_thickness_z: int = 8

    minimum_top_depth_index: int = 2
    maximum_top_depth_index: int = 16

    minimum_density_contrast: float = 0.2
    maximum_density_contrast: float = 1.0

    horizontal_margin_cells: int = 2

    def validate(
        self,
        grid: MatlabCompatibleGridSpec,
    ) -> None:
        """
        Validate body-sampling ranges against the repository grid.
        """

        integer_ranges = {
            "width_x": (
                self.minimum_width_x,
                self.maximum_width_x,
            ),
            "width_y": (
                self.minimum_width_y,
                self.maximum_width_y,
            ),
            "thickness_z": (
                self.minimum_thickness_z,
                self.maximum_thickness_z,
            ),
            "top_depth_index": (
                self.minimum_top_depth_index,
                self.maximum_top_depth_index,
            ),
        }

        for name, (
            minimum,
            maximum,
        ) in integer_ranges.items():
            if minimum < 1:
                raise ValueError(
                    f"{name} minimum must be at least one."
                )

            if maximum < minimum:
                raise ValueError(
                    f"{name} maximum must be greater than or "
                    "equal to its minimum."
                )

        if self.horizontal_margin_cells < 0:
            raise ValueError(
                "horizontal_margin_cells must not be negative."
            )

        if (
            self.minimum_density_contrast
            <= 0.0
        ):
            raise ValueError(
                "minimum_density_contrast must be positive."
            )

        if (
            self.maximum_density_contrast
            < self.minimum_density_contrast
        ):
            raise ValueError(
                "maximum_density_contrast must be greater than or "
                "equal to minimum_density_contrast."
            )

        available_x = (
            grid.nx
            - 2 * self.horizontal_margin_cells
        )

        available_y = (
            grid.ny
            - 2 * self.horizontal_margin_cells
        )

        if self.maximum_width_x > available_x:
            raise ValueError(
                "maximum_width_x does not fit inside the grid "
                "with the requested horizontal margin."
            )

        if self.maximum_width_y > available_y:
            raise ValueError(
                "maximum_width_y does not fit inside the grid "
                "with the requested horizontal margin."
            )

        deepest_possible_end = (
            self.maximum_top_depth_index
            + self.maximum_thickness_z
        )

        if deepest_possible_end > grid.nz:
            raise ValueError(
                "The deepest possible body exceeds the model grid: "
                f"{deepest_possible_end} > {grid.nz}."
            )


@dataclass(frozen=True)
class ReceiverSamplingConfig:
    """
    Configuration for the three-dimensional gravity receiver volume.

    Z is positive downward in the translated FWD3D implementation.
    Negative receiver Z values therefore represent elevations above the
    model surface.
    """

    number_of_levels: int = 8
    first_level_z: float = -10.0
    level_spacing: float = 10.0

    def validate(self) -> None:
        """Validate receiver-volume parameters."""

        if self.number_of_levels < 1:
            raise ValueError(
                "number_of_levels must be at least one."
            )

        if self.level_spacing <= 0.0:
            raise ValueError(
                "level_spacing must be greater than zero."
            )

        if self.first_level_z >= 0.0:
            raise ValueError(
                "first_level_z must be negative for above-surface "
                "receivers under the positive-downward convention."
            )

    def build_receiver_grid(
        self,
        grid: MatlabCompatibleGridSpec,
    ) -> ReceiverGrid:
        """
        Build a receiver volume aligned with the repository X-Y grid.

        Returns
        -------
        ReceiverGrid
            Receiver grid whose volume order is
            ``(receiver_z, receiver_y, receiver_x)``.
        """

        self.validate()

        receiver_z = (
            self.first_level_z
            - self.level_spacing
            * np.arange(
                self.number_of_levels,
                dtype=np.float64,
            )
        )

        receiver_x = np.linspace(
            grid.x_min,
            grid.x_max,
            grid.nx,
            dtype=np.float64,
        )

        receiver_y = np.linspace(
            grid.y_min,
            grid.y_max,
            grid.ny,
            dtype=np.float64,
        )

        return ReceiverGrid(
            x=receiver_x,
            y=receiver_y,
            z=receiver_z,
        )


@dataclass(frozen=True)
class DatasetGenerationConfig:
    """
    Complete configuration for baseline dataset generation.

    Parameters
    ----------
    number_of_samples
        Number of gravity-density pairs to generate.
    random_seed
        Seed used by NumPy's random-number generator.
    output_directory
        Directory containing samples, manifest, and metadata.
    overwrite
        Whether an existing dataset directory may be replaced.
    compressed
        Whether individual NumPy sample files use compressed NPZ storage.
    receiver_chunk_size
        Number of receiver points processed together by the forward solver.
    """

    number_of_samples: int = 1000
    random_seed: int = 20260727

    output_directory: Path = Path(
        "datasets/fwd3d_rectangular_baseline"
    )

    overwrite: bool = False
    compressed: bool = True

    receiver_chunk_size: int = 128

    body: BodySamplingConfig = BodySamplingConfig()
    receivers: ReceiverSamplingConfig = (
        ReceiverSamplingConfig()
    )

    def validate(
        self,
        grid: MatlabCompatibleGridSpec,
    ) -> None:
        """Validate the complete generation configuration."""

        if self.number_of_samples < 1:
            raise ValueError(
                "number_of_samples must be at least one."
            )

        if self.receiver_chunk_size < 1:
            raise ValueError(
                "receiver_chunk_size must be at least one."
            )

        self.body.validate(grid)
        self.receivers.validate()

    def resolved_output_directory(
        self,
        repository_root: Path,
    ) -> Path:
        """
        Return an absolute output directory.

        Relative paths are interpreted from the repository root.
        """

        output_directory = self.output_directory

        if not output_directory.is_absolute():
            output_directory = (
                repository_root
                / output_directory
            )

        return output_directory.resolve()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the configuration into JSON-serializable values.
        """

        result = asdict(self)

        result["output_directory"] = str(
            self.output_directory
        )

        return result