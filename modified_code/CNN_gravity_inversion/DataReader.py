from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

# Pillow's Image class is used to resize the gravity anomaly map.
from PIL import Image

def load_data(f_model, f_anomal, Nx, Ny):
    """
    Load a model dataset and its corresponding anomaly dataset.

    Parameters
    ----------
    f_model : str or Path
        Path to the model-data text file.
    f_anomal : str or Path
        Path to the anomaly-data text file.
    Nx : int
        Number of grid cells or samples in the x-direction.
    Ny : int
        Number of grid cells or samples in the y-direction.

    Returns
    -------
    model : numpy.ndarray
        Model array reshaped to:
        (number_of_samples, Nx, Ny, 1)
    anomal : numpy.ndarray
        Loaded anomaly data.
    """

    # Load model values from a space-delimited text file.
    model = np.loadtxt(f_model, delimiter=" ")

    # Reshape the model into a collection of 2D grids with one channel.
    #
    # Expected shape:
    # (number of samples, Nx, Ny, 1)
    model = np.resize(model, (model.shape[0], Nx, Ny, 1))

    # Load the corresponding anomaly data from a space-delimited text file.
    anomal = np.loadtxt(f_anomal, delimiter=" ")

    # Print the complete arrays for debugging.
    # This may produce very large terminal output for large datasets.
    print("model\n", model)
    print("anomal\n", anomal)

    # Return both datasets.
    return model, anomal


def load_3Ddata(f_model, f_anomal, Nx, Ny, Nz):
    """
    Load a 3D density-model dataset and corresponding gravity anomalies.

    Parameters
    ----------
    f_model : str or Path
        Path to the 3D model-data text file.
    f_anomal : str or Path
        Path to the anomaly-data text file.
    Nx, Ny, Nz : int
        Dimensions of the 3D model grid.

    Returns
    -------
    model : numpy.ndarray
        Model array with shape:
        (number_of_samples, Nz, Nx, Ny)
    anomal : numpy.ndarray
        Gravity anomaly array with shape:
        (number_of_samples, 1, Nx, Ny)
    """

    # Load the density-model dataset.
    model = np.loadtxt(f_model, delimiter=" ")

    # Print the original row and column counts of the loaded model file.
    print(
        "shape of f_model:\n",
        model.shape[0],
        model.shape[1],
    )

    # Reshape each model sample into a 3D volume.
    # Shape convention:
    # (batch, depth, x, y)
    model = np.resize(
        model,
        (model.shape[0], Nz, Nx, Ny),
    )

    # Load the corresponding gravity-anomaly dataset.
    anomal = np.loadtxt(f_anomal, delimiter=" ")

    # Reshape each anomaly into a single-channel 2D grid.
    # Shape convention:
    # (batch, channel, x, y)
    anomal = np.resize(
        anomal,
        (anomal.shape[0], 1, Nx, Ny),
    )

    return model, anomal


def load_3Ddata_dxdydz(f_dxdydz):
    """
    Load grid spacings and convert dx and dy into ratios relative to dz.

    The input file is expected to contain three values per sample:

        dx, dy, dz

    Returns
    -------
    dxdy : numpy.ndarray
        Two-column array containing:
        [dx/dz, dy/dz]
    dz : numpy.ndarray
        One-column array containing the original dz values.
    """

    # Load grid-spacing values from the text file.
    dxdydz = np.loadtxt(f_dxdydz, delimiter=" ")

    # Print the original loaded data.
    # "chushi" likely means "initial" or "original."
    print("chushi:\n", dxdydz)

    # Ensure the data have three columns:
    # dx, dy, and dz.
    dxdydz = np.resize(
        dxdydz,
        (dxdydz.shape[0], 3),
    )

    # Extract the third column as dz.
    # Shape:
    # (number_of_samples, 1)
    dz = dxdydz[:, 2:3]

    # Extract the first two columns as dx and dy.
    # Shape:
    # (number_of_samples, 2)
    dxdy = dxdydz[:, 0:2]

    # Normalize horizontal grid spacing by vertical grid spacing.
    # Result:
    # [dx/dz, dy/dz]
    dxdy = np.true_divide(dxdy, dz)

    # Print the normalized spacing ratios.
    print("dxdy: ", dxdy)

    return dxdy, dz


def load_3Ddata_dxdydz_dz(f_dxdydz):
    """
    Load dx, dy, and dz values without normalizing them.

    Returns
    -------
    dxdydz : numpy.ndarray
        Array with shape:
        (number_of_samples, 3)
    """

    # Load the grid-spacing data.
    dxdydz = np.loadtxt(f_dxdydz, delimiter=" ")

    # Print the original loaded values.
    print("chushi:\n", dxdydz)

    # Ensure that each sample has exactly three values:
    # dx, dy, and dz.
    dxdydz = np.resize(
        dxdydz,
        (dxdydz.shape[0], 3),
    )

    return dxdydz


def load_pre_anomal(f_anomal, Nx, Ny, final_Nx, final_Ny):
    """
    Load one gravity anomaly, resize it, and add batch/channel dimensions.

    Parameters
    ----------
    f_anomal : str or Path
        Path to the anomaly text file.
    Nx, Ny : int
        Original anomaly dimensions.
    final_Nx, final_Ny : int
        Dimensions required by the CNN.

    Returns
    -------
    anomal : numpy.ndarray
        Resized anomaly with shape:
        (1, 1, final_Nx, final_Ny)
    """

    # Load the gravity-anomaly values from the text file.
    anomal = np.loadtxt(f_anomal)

    # Reshape the values into the original 2D anomaly grid.
    anomal = np.resize(anomal, (Nx, Ny))

    # Convert the NumPy array into a Pillow Image so it can be resized.
    im = Image.fromarray(anomal)

    # Resize the anomaly map to the dimensions expected by the CNN.
    #
    # Image.ANTIALIAS was used in older Pillow versions for high-quality
    # downsampling. In newer Pillow versions it has been replaced by:
    #
    # Image.Resampling.LANCZOS
    im_res = im.resize(
        (final_Nx, final_Ny),
        Image.Resampling.LANCZOS,
    )  

    # Convert the resized image back into a NumPy array.
    anomal = np.array(im_res)

    # Add a batch dimension and a channel dimension.
    #
    # Final shape:
    # (batch, channel, x, y)
    # (1, 1, final_Nx, final_Ny)
    anomal = np.resize(
        anomal,
        (1, 1, final_Nx, final_Ny),
    )

    return anomal


def load_pre_model(f_model, Nx, Ny, Nz):
    """
    Load one previously known or true 3D density model.

    Parameters
    ----------
    f_model : str or Path
        Path to the model text file.
    Nx, Ny, Nz : int
        Dimensions of the 3D density grid.

    Returns
    -------
    model : numpy.ndarray
        Model with shape:
        (1, Nz, Nx, Ny)
    """

    # Print a status message before reading the model.
    print("loading...pre_model")

    # Load density values from a space-delimited text file.
    model = np.loadtxt(f_model, delimiter=" ")

    # Print a status message before reshaping.
    print("resizing...pre_model")

    # Reshape the values into one 3D model.
    # Final shape:
    # (batch, depth, x, y)
    # (1, Nz, Nx, Ny)
    model = np.resize(
        model,
        (1, Nz, Nx, Ny),
    )

    return model

