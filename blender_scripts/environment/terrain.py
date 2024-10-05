from mathutils import Vector
from mathutils.noise import fractal

from PIL import Image
from pydelatin import Delatin

import numpy as np
from typing import Tuple

from configuration.consts import Constants
from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


def create_delatin_mesh_from_terrain(terrain: np.ndarray, seed: int = None) -> Delatin:
    """
    Create meshes using Delatin algorithm.

    Args:
        terrain: The terrain to create meshes from.
        seed: Random seed for reproducibility.

    Returns:
        vertices: The vertices of the mesh.
        faces: The faces of the mesh.
    """
    logger.info("Creating Delatin mesh...")

    if seed is not None:
        np.random.seed(seed)

    width, height = terrain.shape

    delatin = Delatin(terrain * np.random.uniform(low=1, high=2.5), width=width, height=height)

    logger.info(f"Delatin mesh created: {len(delatin.vertices)} vertices, {len(delatin.triangles)} faces.")

    return delatin


def create_terrain_segmentation(
        world_size: int = int(Constants.Default.WORLD_SIZE),
        num_octaves: Tuple[int, int] = (3, 4),
        H: Tuple[float, float] = (0.4, 0.5),
        lacunarity: Tuple[float, float] = (1.1, 1.2),
        image_size: int = Constants.Default.IMAGE_SIZE,
        band: int = 48,
        noise_basis: str = Constants.Default.NOISE_BASIS,
        seed: int = None
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generates a fractal height map and a segmentation map for the terrain.

    Args:
        world_size: Size of the terrain in Blender units.
        num_octaves: Number of octaves for fractal noise.
        H: Controls roughness of the fractal noise.
        lacunarity: Frequency multiplier for successive noise layers.
        image_size: Size of the resulting image (height map resolution).
        band: Threshold band around the midpoint to classify different areas.
        noise_basis: Type of noise basis (e.g., "PERLIN_ORIGINAL").
        seed: Random seed for reproducibility.

    Return:
        height_map: The normalized height map (0-1).
        seg_map: A 3-channel segmentation map (RGB).
        masks: Tuple of binary masks (grass, texture, beds).
    """

    normalized_height_map = _generate_fractal_heightmap(
        world_size=world_size,
        num_octaves=num_octaves,
        H=H,
        lacunarity=lacunarity,
        image_size=image_size,
        noise_basis=noise_basis,
        seed=seed
    )

    # Step 2: Generate segmentation map
    return _create_segmentation_masks(
        height_map=normalized_height_map,
        band=band
    )


def _generate_fractal_heightmap(
        world_size: float = Constants.Default.WORLD_SIZE,
        num_octaves: Tuple[int, int] = (3, 4),
        H: Tuple[float, float] = (0.4, 0.5),
        lacunarity: Tuple[float, float] = (1.1, 1.2),
        image_size: int = Constants.Default.IMAGE_SIZE,
        noise_basis: str = Constants.Default.NOISE_BASIS,
        seed: int = None
) -> np.ndarray:
    """
    Generates a normalized height map based on fractal noise.

    Args:
        world_size: Size of the terrain in Blender units.
        num_octaves: Number of octaves for fractal noise.
        H: Controls roughness of the fractal noise.
        lacunarity: Frequency multiplier for successive noise layers.
        image_size: Size of the resulting image (height map resolution).
        noise_basis: Type of noise basis (e.g., "PERLIN_ORIGINAL").
        seed: Random seed for reproducibility.

    Returns:
        A height map (2D array) normalized to [0, 255] representing terrain heights.
    """
    logger.info("Generating fractal height map...")

    if seed is not None:
        np.random.seed(seed)

    # Generate a grid of points
    grid = np.linspace(-world_size / 2, world_size / 2, 1000, endpoint=True)

    # Randomize fractal noise parameters
    restart = np.random.randint(world_size // 3, world_size // 2)
    num_octaves = int(np.random.randint(*num_octaves))
    H = np.random.uniform(*H)
    lacunarity = np.random.uniform(*lacunarity)
    offset = np.random.randint(low=0.0, high=world_size // 2)

    # Generate the height map using fractal noise
    height_map = []
    for x in grid:
        depths = []

        for y in grid:
            z = fractal(
                Vector((x / restart + offset, y / restart + offset, 0)),
                H,
                lacunarity,
                num_octaves,
                noise_basis=noise_basis,
            )
            depths.append(z)
        height_map.append(depths)

    # Resize the height map to the desired image size
    height_map = np.array(height_map)
    height_map = Image.fromarray(height_map).resize((image_size,) * 2, Image.Resampling.BILINEAR)
    height_map = np.array(height_map)

    # Normalize the height map to [0, 255]
    height_map_min = np.min(height_map)
    height_map_max = np.max(height_map)
    normalized_height_map = (height_map - height_map_min) / (height_map_max - height_map_min) * 255

    logger.info(f"Fractal height map generated: {normalized_height_map.shape}.")

    return normalized_height_map


def _create_segmentation_masks(
        height_map: np.ndarray,
        band: int = 48,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Creates a segmentation map from the height map based on thresholds.

    Args:
        height_map: The normalized height map (0-1).
        band: Threshold band around the midpoint to classify different areas.

    Returns:
        height_map: The scaled height map (0-1).
        seg_map: A 3-channel segmentation map (RGB).
        masks: Tuple of binary masks (grass, texture, beds).
    """
    logger.info("Creating segmentation map...")

    # Threshold values
    lower_band = 128 - band
    upper_band = 128 + band

    # Create binary masks for grass, not grass, and beds as uint8 arrays
    grass = np.asarray((height_map > lower_band) & (height_map < upper_band)).astype(np.uint8) * 255
    texture = np.asarray(height_map >= upper_band).astype(np.uint8) * 255
    beds = np.asarray(height_map <= lower_band).astype(np.uint8) * 255

    # Create segmentation map with 3 channels (R: texture, G: grass, B: beds) by stacking the channels
    seg_map = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype=np.uint8)
    seg_map[..., 0] = texture
    seg_map[..., 1] = grass
    seg_map[..., 2] = beds

    # Normalize the scaled height map to [0, 1]
    height_map /= 255

    logger.info(f"Segmentation map created: {seg_map.shape}.")

    return height_map, seg_map, (grass, texture, beds)
