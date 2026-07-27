from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from pathlib import Path
import os

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.plotting import (
    plot_published_example_density_comparison,
)
from synthetic_models.common.grid import GridSpec


# DataReader.py is a local file in this repository.
# It contains functions for loading and reshaping the gravity anomaly data.
import DataReader as dr

# inv_model.py is a local file containing the CNN architecture.
import inv_model as inv_model

# inv_plot.py is a local file containing plotting functions.
import inv_plot as inv_p

# Datatransfer.py is a local file that contains preprocessing functions.
import Datatransfer as ds

# ---------------------------------------------------------------------------
# DEVICE CONFIGURATION
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TEST_DATA_DIR = REPO_ROOT / "test_data"

input_pre_anomal = TEST_DATA_DIR / "anomal_one.txt"
model_weight_save_path = TEST_DATA_DIR / "3Dmodel_weight3by3.h5"
model_save_path = TEST_DATA_DIR / "3Dpredict_model_one.txt"
true_model_path = TEST_DATA_DIR / "model_one.txt"

FIGURE_DIR = REPO_ROOT / "reproduced_example" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

anomaly_figure_path = FIGURE_DIR / "input_gravity_anomaly.png"
model_figure_path = FIGURE_DIR / "predicted_density_model.png"

test_model_path = TEST_DATA_DIR / "models_dataset_t.txt"
test_anomaly_path = TEST_DATA_DIR / "anomal_dataset_t.txt"
test_spacing_path = TEST_DATA_DIR / "dxdydz_dataset_t.txt"

sample_index = 80

# Select the visible GPU ("0" = first GPU, "2" = third GPU).
# Remove this line for CPU use. Ideally set it before importing TensorFlow.
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Create a TensorFlow distribution strategy.
strategy = tf.distribute.get_strategy()

# Print the number of devices participating in the distribution strategy.
print("Number of device :%d " % strategy.num_replicas_in_sync)


# ---------------------------------------------------------------------------
# INPUT AND OUTPUT FILE PATHS
# ---------------------------------------------------------------------------

# These are hard-coded paths from the original author's Linux computer.
# They must be changed before this script can run on your Windows computer.

# Path to the observed gravity anomaly that will be inverted.
input_pre_anomal = TEST_DATA_DIR / "anomal_one.txt"

# Path to the pretrained CNN weight file.
model_weight_save_path = TEST_DATA_DIR / "3Dmodel_weight3by3.h5"

# Path where the predicted 3D density model will be saved as a text file.
model_save_path = TEST_DATA_DIR / "3Dpredict_model_one.txt"

print("Gravity file:", input_pre_anomal)
print("Output file:", model_save_path)

if not input_pre_anomal.exists():
    raise FileNotFoundError(input_pre_anomal)

if not model_weight_save_path.exists():
    raise FileNotFoundError(model_weight_save_path)

if not true_model_path.exists():
    raise FileNotFoundError(true_model_path)
# ---------------------------------------------------------------------------
# ORIGINAL DATA GRID DIMENSIONS
# ---------------------------------------------------------------------------

# Number of points or cells in the x-direction in the original gravity data.
Nx_ori = 64

# Number of points or cells in the y-direction in the original gravity data.
Ny_ori = 64

# Number of cells in the z-direction in the original 3D model.
Nz_ori = 24


# ---------------------------------------------------------------------------
# PHYSICAL MODEL COORDINATES
# ---------------------------------------------------------------------------

# Beginning x-coordinate of the physical model domain.
begin_x = 0

# Ending x-coordinate of the physical model domain.
end_x = 630

# Beginning y-coordinate of the physical model domain.
begin_y = 0

# Ending y-coordinate of the physical model domain.
end_y = 630

# Maximum depth or ending z-coordinate.
# The script assumes that the beginning z-coordinate is zero.
end_z = 23


# ---------------------------------------------------------------------------
# CNN GRID DIMENSIONS
# ---------------------------------------------------------------------------

# Number of cells or samples in x expected by the CNN.
Nx = 64

# Number of cells or samples in y expected by the CNN.
Ny = 64

# Number of depth cells in the CNN's predicted density model.
Nz = 24


# ---------------------------------------------------------------------------
# LOAD THE INPUT GRAVITY ANOMALY
# ---------------------------------------------------------------------------

# Load and reshape the gravity anomaly to the CNN's required grid size.
# Likely output shape: (batch, channel, x, y) = (1, 1, 64, 64).

anomal = dr.load_pre_anomal(
    input_pre_anomal,
    Nx_ori,
    Ny_ori,
    Nx,
    Ny,
)

print("\nLoaded anomaly:")
print("Shape:", anomal.shape)
print("Data type:", anomal.dtype)
print("Minimum:", np.min(anomal))
print("Maximum:", np.max(anomal))
print("Mean:", np.mean(anomal))
print("Contains NaN:", np.isnan(anomal).any())
print("True model file:", true_model_path)

# ---------------------------------------------------------------------------
# CALCULATE GRID SPACING
# ---------------------------------------------------------------------------

# Calculate grid spacing in x, y, and z.
# Here: dx = 10, dy = 10, dz = 1.

[dx, dy, dz] = [
    (end_x - begin_x) / (Nx - 1),
    (end_y - begin_y) / (Ny - 1),
    end_z / (Nz - 1),
]


# ---------------------------------------------------------------------------
# CREATE THE GRID-SPACING INPUT FOR THE CNN
# ---------------------------------------------------------------------------

# Calculate horizontal-to-vertical grid-spacing ratios.
# This creates a one-dimensional NumPy array.
dxdy = np.array([dx / dz, dy / dz])

# Resize the array
dxdy = np.resize(dxdy, (1, 2))


# ---------------------------------------------------------------------------
# PREPROCESS THE GRAVITY ANOMALY
# ---------------------------------------------------------------------------

# Transform or scale the gravity anomaly using the vertical grid spacing dz.
print("\nBefore dz scaling:")
print("Minimum:", np.min(anomal))
print("Maximum:", np.max(anomal))

anomal = ds.anomal_tran_deltz(anomal, dz)

print("\nAfter dz scaling:")
print("Minimum:", np.min(anomal))
print("Maximum:", np.max(anomal))

all_true_models, all_anomalies = dr.load_3Ddata(
    test_model_path,
    test_anomaly_path,
    Nx,
    Ny,
    Nz,
)

all_dxdy, all_dz = dr.load_3Ddata_dxdydz(
    test_spacing_path
)

true_model = np.asarray(
    all_true_models[sample_index],
    dtype=np.float32,
)

anomal = np.asarray(
    all_anomalies[sample_index:sample_index + 1],
    dtype=np.float32,
)

dxdy = np.asarray(
    all_dxdy[sample_index:sample_index + 1],
    dtype=np.float32,
)

# ---------------------------------------------------------------------------
# BUILD THE CNN AND RUN THE INVERSION
# ---------------------------------------------------------------------------

# Enter the TensorFlow distribution-strategy context.
#
# Any model variables created inside this block are placed and managed
# according to the selected MirroredStrategy.
with strategy.scope():

    # Build the CNN architecture.
    model2 = inv_model.inv_model7_3by3()

    # Print a summary of the network architecture.
    model2.summary()

    # Load the learned CNN weights from the .h5 file.
    model2.load_weights(model_weight_save_path)

    # Print a status message before running inference.
    print("starting predict\n")

    # Run the CNN inversion.
    # Convert NumPy arrays to TensorFlow float32 tensors.
    anomal_tensor = tf.convert_to_tensor(anomal, dtype=tf.float32)
    dxdy_tensor = tf.convert_to_tensor(dxdy, dtype=tf.float32)

    # Inputs must follow the same order used when the model was created:
    d_model_p = model2.predict(
        [anomal_tensor, dxdy_tensor]
    )

    print("\nPrediction results:")
    print("Shape:", d_model_p.shape)
    print("Data type:", d_model_p.dtype)
    print("Minimum:", np.min(d_model_p))
    print("Maximum:", np.max(d_model_p))
    print("Mean:", np.mean(d_model_p))
    print("Contains NaN:", np.isnan(d_model_p).any())

    # Print a message indicating that the results will now be plotted.
    print("plot result\n")


grid = GridSpec()

comparison_figure_path = (
    FIGURE_DIR
    / "published_example_true_recovered_comparison.png"
)

plot_published_example_density_comparison(
    true_model=true_model,
    recovered_model=d_model_p[0],
    grid=grid,
    output_path=comparison_figure_path,
    case_name="Published Example Reproduction",
)

# ---------------------------------------------------------------------------
# PLOT THE INPUT GRAVITY ANOMALY
# ---------------------------------------------------------------------------

# Extract one two-dimensional gravity map from the anomaly tensor.
inv_p.plot_anomal(
    anomal[0, 0, :, :],
    title="Input Gravity Anomaly",
    save_path=anomaly_figure_path,
)


# ---------------------------------------------------------------------------
# PLOT THE PREDICTED 3D DENSITY MODEL
# ---------------------------------------------------------------------------

# Extract the first predicted model from the batch.
inv_p.plot_model_64(
    d_model_p[0, :, :, :],
    title="CNN-Predicted Density Model",
    save_path=model_figure_path,
)


# ---------------------------------------------------------------------------
# FLATTEN THE PREDICTED MODEL
# ---------------------------------------------------------------------------

# Convert the predicted 3D density model into a single column of values.
d_model_p = d_model_p.reshape(-1, 1)
print("Flattened output shape:", d_model_p.shape)


# ---------------------------------------------------------------------------
# SAVE THE PREDICTED DENSITY MODEL
# ---------------------------------------------------------------------------

# Save the flattened predicted density model as a text file.
np.savetxt(
    model_save_path,
    d_model_p,
    fmt="%.07f",
    newline="\n",
)