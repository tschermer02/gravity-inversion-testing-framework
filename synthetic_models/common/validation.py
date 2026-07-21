from __future__ import annotations

import numpy as np

from synthetic_models.common.bodies import (
    CaseSpec,
    DippingBodySpec,
    MultiBodyCaseSpec,
    RectangularBodySpec,
    SaltDomeSpec,
    BasementReliefSpec
)
from synthetic_models.common.grid import GridSpec

def validate_case_names(
    cases: list[CaseSpec],
) -> None:
    """Validate that every synthetic case has a unique, nonempty name.

    Parameters
    ----------
    cases
        Synthetic case specifications to validate.

    Raises
    ------
    ValueError
        If a case name is empty or duplicated.
    """
    case_names = [case.name for case in cases]

    if any(not name.strip() for name in case_names):
        raise ValueError(
            "Every synthetic case must have a nonempty name."
        )

    if len(case_names) != len(set(case_names)):
        raise ValueError(
            "Every synthetic case must have a unique name."
        )

def validate_density_model(
    model: np.ndarray,
    body: CaseSpec,
    grid: GridSpec,
    *,
    model_label: str,
    check_rectangular_body: bool = False,
) -> int | None:
    """Validate a generated density model.

    The function always validates the model's array type, shape, dtype,
    memory layout, and numerical values.

    When ``check_rectangular_body`` is enabled, the function also validates
    the occupied-cell count and density values where possible. Exact
    occupied-cell counts are checked for axis-aligned rectangular bodies.
    Dipping bodies are voxelized after rotation, so their exact occupied-cell
    count depends on orientation and cell-center alignment; for those bodies,
    the function only requires at least one occupied cell.

    Parameters
    ----------
    model
        Density model in ``(z, y, x)`` array order.
    body
        Case specification used to create the density model.
    grid
        Grid specification defining the expected model dimensions.
    model_label
        Human-readable model label used in validation error messages.
    check_rectangular_body
        Whether to perform geometry-specific occupied-cell and density checks.
        The name is retained for compatibility with the existing pipeline.

    Returns
    -------
    int or None
        Number of nonzero model cells when geometry-specific validation is
        requested; otherwise, ``None``.

    Raises
    ------
    TypeError
        If the model is not a NumPy array, has the wrong dtype, or the case
        specification is unsupported.
    ValueError
        If the model has the wrong shape, is not C-contiguous, contains
        invalid values, has an unexpected occupied-cell count, contains no
        occupied cells, or uses an incorrect density contrast.
    """
    if not isinstance(model, np.ndarray):
        raise TypeError(
            f"{body.name}: {model_label} model must be a NumPy array, "
            f"but received {type(model).__name__}."
        )

    expected_shape = (
        grid.nz,
        grid.ny,
        grid.nx,
    )

    if model.shape != expected_shape:
        raise ValueError(
            f"{body.name}: expected {model_label} model shape "
            f"{expected_shape}, but received {model.shape}."
        )

    if model.dtype != np.float32:
        raise TypeError(
            f"{body.name}: expected {model_label} model dtype "
            f"float32, but received {model.dtype}."
        )

    if not model.flags["C_CONTIGUOUS"]:
        raise ValueError(
            f"{body.name}: {model_label} model must be C-contiguous."
        )

    if not np.all(np.isfinite(model)):
        raise ValueError(
            f"{body.name}: {model_label} model contains "
            "NaN or infinite values."
        )

    if not check_rectangular_body:
        return None

    actual_nonzero_cells = int(np.count_nonzero(model))

    if actual_nonzero_cells == 0:
        raise ValueError(
            f"{body.name}: {model_label} model contains no nonzero cells."
        )

    expected_nonzero_cells = _get_expected_nonzero_cell_count(
        body=body,
    )

    if (
        expected_nonzero_cells is not None
        and actual_nonzero_cells != expected_nonzero_cells
    ):
        raise ValueError(
            f"{body.name}: expected {expected_nonzero_cells} "
            f"nonzero cells, but found {actual_nonzero_cells}."
        )

    _validate_occupied_density_values(
        model=model,
        body=body,
        model_label=model_label,
    )

    return actual_nonzero_cells

def _get_expected_nonzero_cell_count(
    *,
    body: CaseSpec,
) -> int | None:
    """Return an exact occupied-cell count when one is available.
    """
    if isinstance(body, RectangularBodySpec):
        return (
            (body.z_end - body.z_start)
            * (body.y_end - body.y_start)
            * (body.x_end - body.x_start)
        )

    if isinstance(body, MultiBodyCaseSpec):
        if body.allow_overlap:
            # Overlapping bodies can share cells, so summing their individual
            # volumes does not give an exact unique occupied-cell count.
            return None

        return body.expected_nonzero_cells

    if isinstance(
        body,
        (
            DippingBodySpec,
            SaltDomeSpec,
            BasementReliefSpec,
        ),
    ):
        return None

    raise TypeError(
        f"Unsupported case specification: {type(body).__name__}."
    )

def _validate_occupied_density_values(
    *,
    model: np.ndarray,
    body: CaseSpec,
    model_label: str,
) -> None:
    """Validate the density values assigned to occupied model cells.

    Parameters
    ----------
    model
        Density model to inspect.
    body
        Case specification containing the expected density contrast.
    model_label
        Human-readable model label used in error messages.

    Raises
    ------
    ValueError
        If an occupied cell does not equal the expected density contrast.
    """
    expected_density = np.float32(body.density_contrast)
    occupied_values = model[model != 0.0]

    if not np.all(occupied_values == expected_density):
        raise ValueError(
            f"{body.name}: {model_label} model contains nonzero density "
            f"values that do not equal the specified contrast "
            f"{body.density_contrast}."
        )