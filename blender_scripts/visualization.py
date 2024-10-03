import numpy as np

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from pydelatin import Delatin

from typing import Tuple, Dict


def visualize_terrain(
        height_map: np.ndarray,
        segmentation_map: np.ndarray,
        grass: np.ndarray,
        texture: np.ndarray,
        beds: np.ndarray
) -> None:
    """
    Visualize the terrain and segmentation map with generated masks.

    Args:
        height_map: The normalized height map (0-1).
        segmentation_map: A 3-channel segmentation map (RGB).
        grass: Binary mask for grass areas.
        texture: Binary mask for texture areas.
        beds: Binary mask for bed areas.
    """
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))

    # Terrain visualization
    axs[0].imshow(height_map, cmap="terrain")
    axs[0].set_title("Terrain")
    axs[0].axis("off")

    # Add contour lines over the terrain
    axs[0].contour(height_map, colors='black', linewidths=0.5)

    # Segmentation map
    axs[1].imshow(segmentation_map)
    axs[1].set_title("Segmentation Map")
    axs[1].axis("off")

    # Add contour lines over the segmentation map
    axs[1].contour(height_map, colors='black', linewidths=0.5)

    # Grass mask
    axs[2].imshow(grass, cmap='gray')
    axs[2].set_title('Grass Mask')
    axs[2].axis('off')

    # Texture mask
    axs[3].imshow(texture, cmap='gray')
    axs[3].set_title('Texture Mask')
    axs[3].axis('off')

    # Beds mask
    axs[4].imshow(beds, cmap='gray')
    axs[4].set_title('Beds Mask')
    axs[4].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_terrain_mesh(mesh: Delatin) -> None:
    """
    Visualize Delatin mesh.

    Args:
        mesh: The Delatin mesh.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    vertices = mesh.vertices
    faces = mesh.triangles

    # Extract the x, y, z coordinates of the vertices
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    # Create a list of triangles, where each triangle is represented by 3 vertices
    mesh_faces = [vertices[face] for face in faces]

    # Create a Poly3DCollection for the triangles (mesh faces)
    poly3d_collection = Poly3DCollection(mesh_faces, facecolor='cyan', linewidths=1, edgecolor='r', alpha=0.5)
    ax.add_collection3d(poly3d_collection)

    # Set the limits for the axes
    ax.set_xlim([np.min(x), np.max(x)])
    ax.set_ylim([np.min(y), np.max(y)])
    ax.set_zlim([np.min(z), np.max(z)])

    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display the mesh
    plt.show()


def visualize_terrain_sub_meshes_2d(sub_meshes: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """
    Visualize sub-meshes in 2D using matplotlib.

    Args:
        sub_meshes: A dictionary of sub-meshes categorized as 'grass', 'texture', and 'beds'.
    """
    fig, ax = plt.subplots()

    # Define colors for each sub-mesh
    colors = {
        "grass": 'green',
        "texture": 'gray',
        "beds": 'brown'
    }

    # Plot each sub-mesh as a 2D scatter plot (X, Y coordinates only)
    for mesh_name, (vertices, faces) in sub_meshes.items():
        if len(vertices) == 0:
            continue  # Skip empty sub-meshes

        # Plot X and Y coordinates, ignoring Z
        ax.scatter(vertices[:, 0], vertices[:, 1], c=colors[mesh_name], label=mesh_name, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Top-Down View of Terrain Sub-Meshes')
    plt.legend()
    plt.show()


def visualize_terrain_sub_meshes_3d(sub_meshes: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    """
    Visualize sub-meshes in 3D

    Args:
        sub_meshes: A dictionary of sub-meshes categorized as 'grass', 'texture', and 'beds'.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define colors for each sub-mesh
    colors = {
        "grass": 'green',
        "texture": 'gray',
        "beds": 'brown'
    }

    # Plot each sub-mesh with its associated color
    for mesh_name, (vertices, faces) in sub_meshes.items():
        if len(vertices) == 0:
            continue  # Skip empty sub-meshes

        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=colors[mesh_name], label=mesh_name)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()
