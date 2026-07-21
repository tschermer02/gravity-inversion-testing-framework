import ctypes
from ctypes import c_float
from pathlib import Path
import numpy as np

# =============================================================================
# Load the compiled Windows DLL
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
WORKING_REPO_DIR = SCRIPT_DIR.parent

lib_path = (
    WORKING_REPO_DIR
    / "forward_modeling_lib"
    / "libpyforward.dll"
)

print("Forward DLL:", lib_path)

if not lib_path.exists():
    raise FileNotFoundError(f"Could not find DLL:\n{lib_path}")

lib = ctypes.CDLL(str(lib_path))

# =============================================================================
# Tell ctypes the function signatures
# =============================================================================

lib.forward_va.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
]
lib.forward_va.restype = None

lib.AplusS.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.AplusS.restype = None


# =============================================================================
# Helper function
# =============================================================================

def trans_np_as_c_ptr(array, length):
    """
    Convert a NumPy array into a C float pointer that can be passed to the DLL.
    """

    array = np.reshape(array, length)
    array = np.asarray(array, dtype=np.float32)

    c_array = (ctypes.c_float * len(array))(*array)

    return c_array


# =============================================================================
# Build the gravity kernel (Va)
# =============================================================================

def gravity_forward_Va(Nx, Ny, Nz, dx, dy, dz):
    """
    Precompute the gravity-response kernel.

    Returns
    -------
    c_Va : ctypes float array
        Kernel used by the forward model.
    """

    vanum = (2 * Nx - 1) * (2 * Ny - 1) * Nz

    Va = np.zeros(vanum, dtype=np.float32)

    c_Va = trans_np_as_c_ptr(Va, vanum)

    measure_h = 0.5

    lib.forward_va(
        c_Va,
        Nx,
        Ny,
        Nz,
        c_float(dx),
        c_float(dy),
        c_float(dz),
        c_float(measure_h),
    )

    return c_Va


# =============================================================================
# Compute gravity anomaly from a density model
# =============================================================================

def gravity_forward(model, c_Va, Nx, Ny, Nz):
    """
    Compute the gravity anomaly produced by a 3D density model.
    """

    c_model = trans_np_as_c_ptr(model, Nx * Ny * Nz)

    anomal = np.zeros(Nx * Ny, dtype=np.float32)

    c_anomal = trans_np_as_c_ptr(anomal, Nx * Ny)

    lib.AplusS(
        c_Va,
        c_model,
        c_anomal,
        Nx,
        Ny,
        Nz,
    )

    anomal = np.array(c_anomal)

    # Return as a 64 x 64 gravity map
    anomal = np.reshape(anomal, (Ny, Nx))

    return anomal