from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synthetic_models.common.bodies import RectangularBodySpec
from synthetic_models.common.grid import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter

def _box_faces(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> list[list[tuple[float, float, float]]]:
    """Return the six polygon faces of a rectangular box."""

    vertices = [
        (x_min, y_min, z_min),
        (x_max, y_min, z_min),
        (x_max, y_max, z_min),
        (x_min, y_max, z_min),
        (x_min, y_min, z_max),
        (x_max, y_min, z_max),
        (x_max, y_max, z_max),
        (x_min, y_max, z_max),
    ]

    return [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[3], vertices[0], vertices[4], vertices[7]],
    ]

def plot_density_model_3d(
    
    model: np.ndarray,
    grid: GridSpec,
    body: RectangularBodySpec,
    output_path: Path,
    make_animation: bool = True,
) -> None:
    """
    Plot the full model domain as a transparent box and the density body
    as a solid rectangular prism.

    A static PNG is always saved. When make_animation is True, a rotating
    GIF is also saved using the same filename stem.
    """

    expected_shape = (grid.nz, grid.ny, grid.nx)

    if model.shape != expected_shape:
        raise ValueError(
            f"Expected model shape {expected_shape}, "
            f"but received {model.shape}."
        )

    if not np.any(model != 0.0):
        raise ValueError("The density model contains no nonzero cells.")

    # Convert the body's index ranges into physical coordinates.
    #
    # The ending indices are exclusive in the NumPy slice, so they represent
    # the outer boundary immediately after the final occupied cell.
    body_x_min = grid.x_min + body.x_start * grid.dx
    body_x_max = grid.x_min + body.x_end * grid.dx

    body_y_min = grid.y_min + body.y_start * grid.dy
    body_y_max = grid.y_min + body.y_end * grid.dy

    body_z_min = grid.z_min + body.z_start * grid.dz
    body_z_max = grid.z_min + body.z_end * grid.dz

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")

    # Draw the full inversion domain as a nearly transparent box.
    domain_faces = _box_faces(
        x_min=grid.x_min,
        x_max=grid.x_max,
        y_min=grid.y_min,
        y_max=grid.y_max,
        z_min=grid.z_min,
        z_max=grid.z_max,
    )

    domain_box = Poly3DCollection(
        domain_faces,
        facecolors="lightgray",
        edgecolors="gray",
        linewidths=0.8,
        alpha=0.05,
    )

    axis.add_collection3d(domain_box)

    # Draw the anomalous density body as a solid rectangular prism.
    body_faces = _box_faces(
        x_min=body_x_min,
        x_max=body_x_max,
        y_min=body_y_min,
        y_max=body_y_max,
        z_min=body_z_min,
        z_max=body_z_max,
    )

    body_box = Poly3DCollection(
        body_faces,
        facecolors="tab:orange",
        edgecolors="black",
        linewidths=1.5,
        alpha=0.9,
    )

    axis.add_collection3d(body_box)

    # Mark the physical center of the body.
    body_x_center = 0.5 * (body_x_min + body_x_max)
    body_y_center = 0.5 * (body_y_min + body_y_max)
    body_z_center = 0.5 * (body_z_min + body_z_max)

    axis.plot(
        [body_x_center],
        [body_y_center],
        [body_z_center],
        marker="o",
        markersize=7,
        color="red",
        linestyle="None",
        label="Body center",
    )

    # Set the full physical model limits.
    axis.set_xlim(grid.x_min, grid.x_max)
    axis.set_ylim(grid.y_min, grid.y_max)

    # Reverse the z-axis so that increasing depth is displayed downward.
    axis.set_zlim(grid.z_max, grid.z_min)

    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("Depth z")

    vertical_exaggeration = 8.0

    axis.set_title(
        f"{body.name}\n"
        f"Density contrast = {body.density_contrast}, "
        f"vertical exaggeration = {vertical_exaggeration:g}×"
    )

    axis.set_box_aspect(
        (
            grid.x_max - grid.x_min,
            grid.y_max - grid.y_min,
            vertical_exaggeration
            * (grid.z_max - grid.z_min),
        )
    )

    axis.legend()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save a static view first.
    axis.view_init(
        elev=24,
        azim=-55,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    # Optionally create a rotating GIF.
    if make_animation:

        def update_view(frame: int) -> tuple:
            axis.view_init(
                elev=24,
                azim=frame,
            )

            return (axis,)

        animation = FuncAnimation(
            figure,
            update_view,
            frames=range(0, 360, 5),
            interval=20,
            blit=False,
        )

        animation_path = output_path.with_suffix(".gif")

        animation.save(
            animation_path,
            writer=PillowWriter(fps=5),
            dpi=150,
        )

    plt.close(figure)
