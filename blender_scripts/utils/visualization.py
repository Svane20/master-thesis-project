import numpy as np
from numpy.typing import NDArray

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from pydelatin import Delatin

from typing import Tuple, Dict

from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


def visualize_terrain(height_map: NDArray[np.float32], segmentation_map: NDArray[np.uint8]) -> None:
    """
    Visualize the terrain and segmentation map in 2D.

    Args:
        height_map (NDArray[np.float32]): The normalized height map (0-1).
        segmentation_map (NDArray[np.uint8]): A 3-channel segmentation map (RGB).
    """
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

    # Terrain visualization
    axs[0].imshow(height_map, cmap="terrain")
    axs[0].set_title("Terrain")
    axs[0].axis("off")
    axs[0].contour(height_map, colors='black', linewidths=0.5)

    # Segmentation map visualization
    axs[1].imshow(segmentation_map)
    axs[1].set_title("Segmentation Map")
    axs[1].axis("off")
    axs[1].contour(height_map, colors='black', linewidths=0.5)

    plt.tight_layout()
    plt.show()


def visualize_terrain_mesh(mesh: Delatin) -> None:
    """
    Visualize a 3D terrain mesh generated by Delatin.

    Args:
        mesh (Delatin): The Delatin mesh containing vertices and faces.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    vertices = mesh.vertices
    faces = mesh.triangles

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    # Create a list of triangles for the mesh
    mesh_faces = [vertices[face] for face in faces]
    poly3d_collection = Poly3DCollection(mesh_faces, facecolor='cyan', linewidths=1, edgecolor='r', alpha=0.5)
    ax.add_collection3d(poly3d_collection)

    ax.set_xlim([np.min(x), np.max(x)])
    ax.set_ylim([np.min(y), np.max(y)])
    ax.set_zlim([np.min(z), np.max(z)])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def visualize_terrain_sub_meshes_2d(sub_meshes: Dict[str, Tuple[NDArray[np.float32], NDArray[np.int32]]]) -> None:
    """
    Visualize sub-meshes in 2D using scatter plots.

    Args:
        sub_meshes (Dict[str, Tuple[NDArray[np.float32], NDArray[np.int32]]]): A dictionary of sub-meshes categorized as 'grass', 'texture', and 'beds'.
    """
    fig, ax = plt.subplots()

    colors = {
        "grass": 'green',
        "texture": 'gray',
        "beds": 'brown'
    }

    for mesh_name, (vertices, faces) in sub_meshes.items():
        if len(vertices) == 0:
            continue
        ax.scatter(vertices[:, 0], vertices[:, 1], c=colors[mesh_name], label=mesh_name, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Top-Down View of Terrain Sub-Meshes')
    plt.legend()
    plt.show()


def visualize_terrain_sub_meshes_3d(sub_meshes: Dict[str, Tuple[NDArray[np.float32], NDArray[np.int32]]]) -> None:
    """
    Visualize sub-meshes in 3D using scatter plots.

    Args:
        sub_meshes (Dict[str, Tuple[NDArray[np.float32], NDArray[np.int32]]]): A dictionary of sub-meshes categorized as 'grass', 'texture', and 'beds'.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = {
        "grass": 'green',
        "texture": 'gray',
        "beds": 'brown'
    }

    for mesh_name, (vertices, faces) in sub_meshes.items():
        if len(vertices) == 0:
            continue
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=colors[mesh_name], label=mesh_name)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()
