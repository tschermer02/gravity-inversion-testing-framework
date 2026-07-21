from pathlib import Path
'''
Train and evaluate the 3D gravity-inversion CNN.

This script:
1. Loads training and test gravity/density datasets.
2. Loads grid-spacing information.
3. Builds the inv_model7_3by3() network.
4. Optionally resumes from saved weights.
5. Trains and evaluates the model.
6. Saves weights, training history, plots, and predictions.

Expected array shapes
---------------------
Gravity anomaly:
    (samples, 1, 64, 64)

Grid ratios:
    (samples, 2), containing [dx/dz, dy/dz]

Density model:
    (samples, 24, 64, 64)
'''

# Set TensorFlow environment variables before importing TensorFlow.
# This reduces informational logging. Change "2" to "0" for full logs.
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf

import DataReader as dr
import inv_model
import inv_plot as inv_p


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# Directory containing this script.
SCRIPT_DIR = Path(__file__).resolve().parent

# Parent directory containing CNN_gravity_inversion/ and test_data/.
REPO_ROOT = SCRIPT_DIR.parent

# Dataset and output directories.
TEST_DATA_DIR = REPO_ROOT / "test_data"
RESULTS_DIR = REPO_ROOT / "reproduced_example"
FIGURE_DIR = RESULTS_DIR / "figures"
HISTORY_DIR = RESULTS_DIR / "history"

# Create output directories if they do not already exist.
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Training data.
input_model = TEST_DATA_DIR / "models_dataset.txt"
input_anomal = TEST_DATA_DIR / "anomal_dataset.txt"
input_dxdydz = TEST_DATA_DIR / "dxdydz_dataset.txt"

# Test data.
input_model_test = TEST_DATA_DIR / "models_dataset_t.txt"
input_anomal_test = TEST_DATA_DIR / "anomal_dataset_t.txt"
input_dxdydz_test = TEST_DATA_DIR / "dxdydz_dataset_t.txt"

# Saved model and diagnostic outputs.
model_weight_save_path = TEST_DATA_DIR / "3Dmodel_weight3by3_test.weights.h5"
model_plot_path = FIGURE_DIR / "model_architecture.png"
history_save_path = HISTORY_DIR / "history_3by3.pkl"

# Output figures.
true_model_figure_path = FIGURE_DIR / "true_density_model.png"
predicted_model_figure_path = FIGURE_DIR / "predicted_density_model.png"
history_figure_path = FIGURE_DIR / "training_history.png"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

Nx = 64
Ny = 64
Nz = 24

# Start with a manageable value while validating the script.
# Increase this only after confirming that the entire training pipeline works.
EPOCHS = 100
BATCH_SIZE = 32

# Set to True to continue training from an existing weight file.
RESUME_TRAINING = True

# Index of the test example to visualize.
TEST_SAMPLE_INDEX = 80


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def require_file(path):
    """Raise a clear error when a required input file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def print_array_summary(name, array):
    """Print useful diagnostics for a NumPy array."""
    print(f"\n{name}")
    print("  shape:", array.shape)
    print("  dtype:", array.dtype)
    print("  min:", np.min(array))
    print("  max:", np.max(array))
    print("  mean:", np.mean(array))
    print("  contains NaN:", np.isnan(array).any())


# ---------------------------------------------------------------------------
# Validate required files
# ---------------------------------------------------------------------------

for required_path in [
    input_model,
    input_anomal,
    input_dxdydz,
    input_model_test,
    input_anomal_test,
    input_dxdydz_test,
]:
    require_file(required_path)

print("Training model file:", input_model)
print("Training anomaly file:", input_anomal)
print("Training spacing file:", input_dxdydz)
print("Test model file:", input_model_test)
print("Test anomaly file:", input_anomal_test)
print("Test spacing file:", input_dxdydz_test)
print("Weight file:", model_weight_save_path)


# ---------------------------------------------------------------------------
# Select the TensorFlow execution strategy
# ---------------------------------------------------------------------------

# get_strategy() works on CPU, one GPU, or a default local setup.
# MirroredStrategy is mainly useful when multiple GPUs are available.
strategy = tf.distribute.get_strategy()

print("\nNumber of devices:", strategy.num_replicas_in_sync)
print("Physical GPUs:", tf.config.list_physical_devices("GPU"))


# ---------------------------------------------------------------------------
# Load training and test datasets
# ---------------------------------------------------------------------------

# Load density models and gravity anomalies.
d_model, d_anomal = dr.load_3Ddata(
    input_model,
    input_anomal,
    Nx,
    Ny,
    Nz,
)

# Load [dx, dy, dz], then convert them to [dx/dz, dy/dz].
dxdy, dz = dr.load_3Ddata_dxdydz(input_dxdydz)

# Load test data.
d_model_t, d_anomal_t = dr.load_3Ddata(
    input_model_test,
    input_anomal_test,
    Nx,
    Ny,
    Nz,
)

dxdy_t, dz_t = dr.load_3Ddata_dxdydz(input_dxdydz_test)

# TensorFlow models normally use float32.
d_model = np.asarray(d_model, dtype=np.float32)
d_anomal = np.asarray(d_anomal, dtype=np.float32)
dxdy = np.asarray(dxdy, dtype=np.float32)

d_model_t = np.asarray(d_model_t, dtype=np.float32)
d_anomal_t = np.asarray(d_anomal_t, dtype=np.float32)
dxdy_t = np.asarray(dxdy_t, dtype=np.float32)

print_array_summary("Training density models", d_model)
print_array_summary("Training gravity anomalies", d_anomal)
print_array_summary("Training grid ratios", dxdy)

print_array_summary("Test density models", d_model_t)
print_array_summary("Test gravity anomalies", d_anomal_t)
print_array_summary("Test grid ratios", dxdy_t)

# Check that all training inputs contain the same number of samples.
if not (len(d_model) == len(d_anomal) == len(dxdy)):
    raise ValueError(
        "Training sample counts do not match: "
        f"models={len(d_model)}, anomalies={len(d_anomal)}, dxdy={len(dxdy)}"
    )

# Check that all test inputs contain the same number of samples.
if not (len(d_model_t) == len(d_anomal_t) == len(dxdy_t)):
    raise ValueError(
        "Test sample counts do not match: "
        f"models={len(d_model_t)}, anomalies={len(d_anomal_t)}, dxdy={len(dxdy_t)}"
    )


# ---------------------------------------------------------------------------
# Build, compile, and optionally restore the CNN
# ---------------------------------------------------------------------------

with strategy.scope():
    # Build the final 3 x 3 convolution architecture selected by the authors.
    model = inv_model.inv_model7_3by3()

    # Print the architecture once.
    model.summary()

    # Adam updates the trainable weights.
    # MSLE compares predicted and true density values on a logarithmic scale.
    # MAPE is retained from the original script as an additional metric.
    model.compile(
        optimizer="adam",
        loss="mean_squared_logarithmic_error",
        metrics=["mape"],
    )

    # Resume from saved weights only when requested and when the file exists.
    if RESUME_TRAINING and model_weight_save_path.exists():
        print("\nLoading existing weights:", model_weight_save_path)
        model.load_weights(model_weight_save_path)
    else:
        print("\nStarting with newly initialized weights.")


# ---------------------------------------------------------------------------
# Train the model
# ---------------------------------------------------------------------------

print("\nStarting training")

# Inputs are passed in the same order used when the Keras model was created:
# inputs=[anomal_input, dxdy_input]
history = model.fit(
    [d_anomal, dxdy],
    d_model,
    validation_data=([d_anomal_t, dxdy_t], d_model_t),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    verbose=1,
)

# Save the final trained weights.
model.save_weights(model_weight_save_path)
print("Saved weights:", model_weight_save_path)


# ---------------------------------------------------------------------------
# Evaluate the test dataset
# ---------------------------------------------------------------------------

print("\nEvaluating test data")

test_results = model.evaluate(
    [d_anomal_t, dxdy_t],
    d_model_t,
    verbose=2,
    return_dict=True,
)

print("Test results:")

if isinstance(test_results, dict):
    for metric_name, metric_value in test_results.items():
        print(f"  {metric_name}: {metric_value}")
else:
    print(test_results)


# ---------------------------------------------------------------------------
# Save training history
# ---------------------------------------------------------------------------

with open(history_save_path, "wb") as file_pi:
    pickle.dump(history.history, file_pi)

print("Saved history:", history_save_path)


# ---------------------------------------------------------------------------
# Predict the full test dataset
# ---------------------------------------------------------------------------

print("\nStarting prediction")

d_model_p = model.predict(
    [d_anomal_t, dxdy_t],
    batch_size=BATCH_SIZE,
    verbose=1,
)

print_array_summary("Predicted density models", d_model_p)

print("\nModel comparison metrics")

# Mean Absolute Error
mae = np.mean(np.abs(d_model_p - d_model_t))

# Root Mean Squared Error
rmse = np.sqrt(np.mean((d_model_p - d_model_t) ** 2))

# Relative L2 error
relative_error = (
    np.linalg.norm(d_model_p - d_model_t)
    / np.linalg.norm(d_model_t)
)

# Correlation coefficient
correlation = np.corrcoef(
    d_model_t.reshape(-1),
    d_model_p.reshape(-1),
)[0, 1]

print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"Relative L2 Error: {relative_error:.6f}")
print(f"Correlation: {correlation:.6f}")

# ---------------------------------------------------------------------------
# Plot one test example
# ---------------------------------------------------------------------------

# Ensure the requested test index exists.
if TEST_SAMPLE_INDEX >= len(d_model_t):
    print(
        f"\nRequested sample {TEST_SAMPLE_INDEX} does not exist. "
        f"Using sample {len(d_model_t) - 1} instead."
    )
    TEST_SAMPLE_INDEX = len(d_model_t) - 1

inv_p.plot_model_64(
    d_model_t[TEST_SAMPLE_INDEX],
    title=f"True density model — sample {TEST_SAMPLE_INDEX}",
    save_path=true_model_figure_path,
)

inv_p.plot_model_64(
    d_model_p[TEST_SAMPLE_INDEX],
    title=f"Predicted density model — sample {TEST_SAMPLE_INDEX}",
    save_path=predicted_model_figure_path,
)

inv_p.history_plot(
    history,
    save_path=history_figure_path,
)


# ---------------------------------------------------------------------------
# Save an image of the network architecture
# ---------------------------------------------------------------------------

try:
    tf.keras.utils.plot_model(
        model,
        to_file=str(model_plot_path),
        show_shapes=True,
        show_layer_names=True,
    )
    print("Saved model architecture:", model_plot_path)
except Exception as error:
    # plot_model requires pydot and Graphviz.
    print("\nCould not create the model architecture image.")
    print("Reason:", error)
    print("The trained model and weights were still saved successfully.")


print("\nTraining script completed.")


