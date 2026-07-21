from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import cppforward
import DataReader as dr


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TEST_DATA_DIR = REPO_ROOT / "test_data"

model_file = TEST_DATA_DIR / "models_dataset_t.txt"
anomaly_file = TEST_DATA_DIR / "anomal_dataset_t.txt"
spacing_file = TEST_DATA_DIR / "dxdydz_dataset_t.txt"

figure_dir = REPO_ROOT / "reproduced_example" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Grid dimensions
# ---------------------------------------------------------------------------

Nx = 64
Ny = 64
Nz = 24

# Choose one example from the test dataset.
sample_index = 0


# ---------------------------------------------------------------------------
# Load the authors' test data
# ---------------------------------------------------------------------------

models, anomalies = dr.load_3Ddata(
    model_file,
    anomaly_file,
    Nx,
    Ny,
    Nz,
)

dxdy, dz_values = dr.load_3Ddata_dxdydz(
    spacing_file
)

models = np.asarray(models, dtype=np.float32)
anomalies = np.asarray(anomalies, dtype=np.float32)
dxdy = np.asarray(dxdy, dtype=np.float32)
dz_values = np.asarray(dz_values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Recover dx, dy, and dz
# ---------------------------------------------------------------------------

# DataReader returns:
# dxdy[:, 0] = dx / dz
# dxdy[:, 1] = dy / dz
# dz_values[:, 0] = dz

dz = float(dz_values[sample_index, 0])
dx = float(dxdy[sample_index, 0] * dz)
dy = float(dxdy[sample_index, 1] * dz)

print("dx:", dx)
print("dy:", dy)
print("dz:", dz)


# ---------------------------------------------------------------------------
# Select one model and corresponding supplied anomaly
# ---------------------------------------------------------------------------

true_model = models[sample_index]

# Stored anomaly shape is (sample, channel, x, y).
provided_anomaly = anomalies[sample_index, 0]


# ---------------------------------------------------------------------------
# Calculate the anomaly with the Windows DLL
# ---------------------------------------------------------------------------

print("Building forward kernel...")

kernel = cppforward.gravity_forward_Va(
    Nx,
    Ny,
    Nz,
    dx,
    dy,
    dz,
)

print("Calculating forward response...")

calculated_anomaly = cppforward.gravity_forward(
    true_model,
    kernel,
    Nx,
    Ny,
    Nz,
)


# ---------------------------------------------------------------------------
# Compare the two anomalies
# ---------------------------------------------------------------------------

residual = calculated_anomaly - provided_anomaly

mae = np.mean(np.abs(residual))
rmse = np.sqrt(np.mean(residual ** 2))

relative_error = (
    np.linalg.norm(residual)
    / np.linalg.norm(provided_anomaly)
)

correlation = np.corrcoef(
    provided_anomaly.reshape(-1),
    calculated_anomaly.reshape(-1),
)[0, 1]

print("\nForward-model comparison")
print("Provided anomaly shape:", provided_anomaly.shape)
print("Calculated anomaly shape:", calculated_anomaly.shape)
print("Provided min/max:", provided_anomaly.min(), provided_anomaly.max())
print("Calculated min/max:", calculated_anomaly.min(), calculated_anomaly.max())
print("MAE:", mae)
print("RMSE:", rmse)
print("Relative L2 error:", relative_error)
print("Correlation:", correlation)
print("Contains NaN:", np.isnan(calculated_anomaly).any())


# ---------------------------------------------------------------------------
# Save comparison plots
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

image_1 = axes[0].pcolor(provided_anomaly)
axes[0].set_title("Authors' anomaly")
fig.colorbar(image_1, ax=axes[0])

image_2 = axes[1].pcolor(calculated_anomaly)
axes[1].set_title("Calculated anomaly")
fig.colorbar(image_2, ax=axes[1])

image_3 = axes[2].pcolor(residual)
axes[2].set_title("Residual")
fig.colorbar(image_3, ax=axes[2])

plt.tight_layout()

output_path = figure_dir / "forward_model_validation.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved comparison figure:", output_path)