from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dataset_generation.config import (
    BodySamplingConfig,
)
from synthetic_models.common.bodies import (
    RectangularBodySpec,
)
from dataset_generation.matlab_grid import MatlabCompatibleGridSpec


@dataclass(frozen=True)
class SampledBody:
    """
    One sampled rectangular body and its derived geometry metadata.

    Index conventions
    -----------------
    ``RectangularBodySpec`` uses normal Python slicing:

    - start indices are inclusive
    - end indices are exclusive

    For example, ``z_start=5`` and ``z_end=10`` occupy vertical cell
    indices 5 through 9.

    Center indices describe the geometric center of the occupied cell
    centers. They may therefore be fractional for bodies containing an
    even number of cells.
    """

    specification: RectangularBodySpec

    width_x: int
    width_y: int
    thickness_z: int

    top_depth_index: int
    bottom_depth_index: int

    center_x_index: float
    center_y_index: float
    center_z_index: float

    top_depth_m: float
    bottom_depth_m: float
    center_depth_m: float

    width_x_m: float
    width_y_m: float
    thickness_z_m: float

    center_x_m: float
    center_y_m: float

    def to_manifest_row(
        self,
        *,
        sample_index: int,
        relative_path: str,
        gravity_minimum: float,
        gravity_maximum: float,
        gravity_mean: float,
        gravity_standard_deviation: float,
        nonzero_density_cells: int,
    ) -> dict[str, int | float | str]:
        """
        Return one CSV-compatible manifest row.

        Parameters
        ----------
        sample_index
            Dataset-wide sample number.
        relative_path
            Sample path relative to the dataset directory.
        gravity_minimum
            Minimum gravity value in mGal.
        gravity_maximum
            Maximum gravity value in mGal.
        gravity_mean
            Mean gravity value in mGal.
        gravity_standard_deviation
            Gravity standard deviation in mGal.
        nonzero_density_cells
            Number of occupied density cells.

        Returns
        -------
        dict
            Manifest values for the sampled body.
        """

        body = self.specification

        return {
            "sample_index": sample_index,
            "relative_path": relative_path,
            "body_name": body.name,
            "x_start": body.x_start,
            "x_end": body.x_end,
            "y_start": body.y_start,
            "y_end": body.y_end,
            "z_start": body.z_start,
            "z_end": body.z_end,
            "width_x": self.width_x,
            "width_y": self.width_y,
            "thickness_z": self.thickness_z,
            "top_depth_index": self.top_depth_index,
            "bottom_depth_index": self.bottom_depth_index,
            "center_x_index": self.center_x_index,
            "center_y_index": self.center_y_index,
            "center_z_index": self.center_z_index,
            "top_depth_m": self.top_depth_m,
            "bottom_depth_m": self.bottom_depth_m,
            "center_depth_m": self.center_depth_m,
            "width_x_m": self.width_x_m,
            "width_y_m": self.width_y_m,
            "thickness_z_m": self.thickness_z_m,
            "center_x_m": self.center_x_m,
            "center_y_m": self.center_y_m,
            "density_contrast": body.density_contrast,
            "nonzero_density_cells": nonzero_density_cells,
            "gravity_minimum_mgal": gravity_minimum,
            "gravity_maximum_mgal": gravity_maximum,
            "gravity_mean_mgal": gravity_mean,
            "gravity_std_mgal": gravity_standard_deviation,
        }

    
class RectangularBodySampler:
    """
    Randomly sample one positive rectangular anomalous-density body.

    The sampler uses only discrete model-cell coordinates so every
    generated body can be represented exactly by ``RectangularBodySpec``.
    """

    def __init__(
        self,
        *,
        grid: MatlabCompatibleGridSpec,
        config: BodySamplingConfig,
        random_generator: np.random.Generator,
    ) -> None:
        config.validate(grid)

        self.grid = grid
        self.config = config
        self.random_generator = (
            random_generator
        )

    def sample(
        self,
        sample_index: int,
    ) -> SampledBody:
        """
        Sample one valid body.

        Parameters
        ----------
        sample_index
            Dataset index used to generate a deterministic body name.

        Returns
        -------
        SampledBody
            Sampled body specification and derived metadata.
        """

        width_x = self._sample_integer_inclusive(
            self.config.minimum_width_x,
            self.config.maximum_width_x,
        )

        width_y = self._sample_integer_inclusive(
            self.config.minimum_width_y,
            self.config.maximum_width_y,
        )

        thickness_z = (
            self._sample_integer_inclusive(
                self.config.minimum_thickness_z,
                self.config.maximum_thickness_z,
            )
        )

        maximum_top_for_thickness = min(
            self.config.maximum_top_depth_index,
            self.grid.nz - thickness_z,
        )

        if (
            maximum_top_for_thickness
            < self.config.minimum_top_depth_index
        ):
            raise RuntimeError(
                "No valid top-depth index exists for the sampled "
                f"thickness {thickness_z}."
            )

        z_start = self._sample_integer_inclusive(
            self.config.minimum_top_depth_index,
            maximum_top_for_thickness,
        )

        z_end = z_start + thickness_z

        x_minimum_start = (
            self.config.horizontal_margin_cells
        )

        x_maximum_start = (
            self.grid.nx
            - self.config.horizontal_margin_cells
            - width_x
        )

        y_minimum_start = (
            self.config.horizontal_margin_cells
        )

        y_maximum_start = (
            self.grid.ny
            - self.config.horizontal_margin_cells
            - width_y
        )

        x_start = self._sample_integer_inclusive(
            x_minimum_start,
            x_maximum_start,
        )

        y_start = self._sample_integer_inclusive(
            y_minimum_start,
            y_maximum_start,
        )

        x_end = x_start + width_x
        y_end = y_start + width_y

        density_contrast = float(
            self.random_generator.uniform(
                self.config.minimum_density_contrast,
                self.config.maximum_density_contrast,
            )
        )

        specification = RectangularBodySpec(
            name=(
                f"rectangular_body_"
                f"{sample_index:06d}"
            ),
            x_start=x_start,
            x_end=x_end,
            y_start=y_start,
            y_end=y_end,
            z_start=z_start,
            z_end=z_end,
            density_contrast=(
                density_contrast
            ),
        )

        specification.validate(
            self.grid
        )

        return SampledBody(
            specification=specification,
            width_x=width_x,
            width_y=width_y,
            thickness_z=thickness_z,
            top_depth_index=z_start,
            bottom_depth_index=z_end - 1,
            center_x_index=(
                x_start + x_end - 1
            )
            / 2.0,
            center_y_index=(
                y_start + y_end - 1
            )
            / 2.0,
            center_z_index=(
                z_start + z_end - 1
            )
            / 2.0,
            top_depth_m=self.grid.z_edge_from_index(
                z_start
            ),
            bottom_depth_m=self.grid.z_edge_from_index(
                z_end
            ),
            center_depth_m=(
                self.grid.z_edge_from_index(z_start)
                + self.grid.z_edge_from_index(z_end)
            )
            / 2.0,
            width_x_m=width_x * self.grid.dx,
            width_y_m=width_y * self.grid.dy,
            thickness_z_m=thickness_z * self.grid.dz,
            center_x_m=(
                self.grid.x_min
                + (
                    x_start + x_end - 1
                )
                / 2.0
                * self.grid.dx
            ),
            center_y_m=(
                self.grid.y_min
                + (
                    y_start + y_end - 1
                )
                / 2.0
                * self.grid.dy
            ),
        )

    def _sample_integer_inclusive(
        self,
        minimum: int,
        maximum: int,
    ) -> int:
        """
        Sample an integer including both endpoints.
        """

        if maximum < minimum:
            raise ValueError(
                "maximum must be greater than or equal to minimum."
            )

        return int(
            self.random_generator.integers(
                minimum,
                maximum + 1,
            )
        )