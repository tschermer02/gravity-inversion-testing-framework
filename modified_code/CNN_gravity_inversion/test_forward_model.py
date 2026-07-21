import numpy as np
import cppforward


# Small test-grid dimensions.
Nx = 4
Ny = 4
Nz = 3

# Cell dimensions.
dx = 10.0
dy = 10.0
dz = 1.0

# Density-model ordering is:
# (depth, y, x) = (Nz, Ny, Nx).
model = np.zeros(
    (Nz, Ny, Nx),
    dtype=np.float32,
)

# Place one unit-density cell near the center of the model.
model[1, 2, 2] = 1.0

print("Building gravity kernel...")

kernel = cppforward.gravity_forward_Va(
    Nx,
    Ny,
    Nz,
    dx,
    dy,
    dz,
)

print("Calculating gravity anomaly...")

anomaly = cppforward.gravity_forward(
    model,
    kernel,
    Nx,
    Ny,
    Nz,
)

# gravity_forward() already returns shape (Ny, Nx).
print("\nResults")
print("Model shape:", model.shape)
print("Anomaly shape:", anomaly.shape)
print("Anomaly dtype:", anomaly.dtype)
print("Minimum:", anomaly.min())
print("Maximum:", anomaly.max())
print("Contains NaN:", np.isnan(anomaly).any())
print("\nGravity anomaly:")
print(anomaly)