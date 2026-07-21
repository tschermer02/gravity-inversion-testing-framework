from __future__ import absolute_import, division, print_function, unicode_literals

from pathlib import Path

import matplotlib.pyplot as plt


def plot_model(d_model_t, title="none", save_path=None):
    """Plot selected depth slices from a smaller 3D density model."""

    # Start a new figure so previous plots do not overlap this one.
    plt.figure(figsize=(15, 12))

    slice_t_1 = d_model_t[1, :, :].T
    slice_t_2 = d_model_t[2, :, :].T
    slice_t_3 = d_model_t[3, :, :].T
    slice_t_4 = d_model_t[4, :, :].T
    slice_t_5 = d_model_t[5, :, :].T

    slice_t_6 = d_model_t[6, :, :].T
    slice_t_7 = d_model_t[7, :, :].T
    slice_t_8 = d_model_t[8, :, :].T
    slice_t_9 = d_model_t[9, :, :].T

    # Use one common color scale for all slices.
    vmin_t = d_model_t.min()
    vmax_t = d_model_t.max()

    slices = [
        slice_t_1,
        slice_t_2,
        slice_t_3,
        slice_t_4,
        slice_t_5,
        slice_t_6,
        slice_t_7,
        slice_t_8,
        slice_t_9,
    ]

    for i, slice_t in enumerate(slices, start=1):
        plt.subplot(5, 5, i)
        plt.pcolor(slice_t, vmin=vmin_t, vmax=vmax_t)
        plt.colorbar()
        plt.title(f"Depth slice {i}")

    # Add one title for the entire figure.
    plt.suptitle(title, fontsize="large", fontweight="bold")

    # Prevent labels and colorbars from overlapping.
    plt.tight_layout()

    # Save the figure when a path is provided.
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print("Saved model figure:", save_path)

    # Close the figure instead of calling plt.show().
    plt.close()


def plot_model_64(d_model_t, title="none", save_path=None):
    """Plot up to 25 horizontal slices from a 3D density model."""

    # Determine the model dimensions.
    Nz, Nx, Ny = d_model_t.shape

    # Limit the figure to 25 slices because the layout is 5 x 5.
    if Nz > 25:
        Nz = 25

    # Use the same color scale for every depth slice.
    vmin_t = d_model_t.min()
    vmax_t = d_model_t.max()

    # Start a new figure large enough for 25 subplots.
    plt.figure(figsize=(18, 15))

    for i in range(1, Nz + 1):
        # Extract one depth slice and transpose it for plotting.
        slice_t = d_model_t[i - 1, :, :].T

        plt.subplot(5, 5, i)

        plt.pcolor(
            slice_t,
            vmin=vmin_t,
            vmax=vmax_t,
        )

        plt.colorbar()

        # Label slices using one-based depth numbering.
        plt.title(f"Depth slice {i}")

    # Add one overall title above all subplots.
    plt.suptitle(title, fontsize="large", fontweight="bold")

    # Improve spacing between the subplots.
    plt.tight_layout()

    # Save the figure if a path was supplied.
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print("Saved 3D model figure:", save_path)

    # Close the figure to free memory and avoid the non-interactive warning.
    plt.close()


def plot_anomal(anomal, title="none", save_path=None):
    """Plot a 2D gravity anomaly map."""

    # Start a new figure.
    plt.figure(figsize=(8, 6))

    plt.title(
        title,
        fontsize="large",
        fontweight="bold",
    )

    plt.pcolor(anomal)
    plt.colorbar(label="Gravity anomaly")

    plt.xlabel("X grid index")
    plt.ylabel("Y grid index")

    plt.tight_layout()

    # Save the anomaly figure if a path was supplied.
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print("Saved anomaly figure:", save_path)

    plt.close()


def history_plot(history, save_path=None):
    """Plot and optionally save the training and validation loss."""

    plt.figure(figsize=(8, 6))

    # Plot training loss.
    plt.plot(history.history["loss"], label="Training loss")

    # Plot validation loss when it exists.
    if "val_loss" in history.history:
        plt.plot(
            history.history["val_loss"],
            label="Validation loss",
        )

    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()

    # Save the figure when an output path is provided.
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print("Saved training-history figure:", save_path)

    plt.close()