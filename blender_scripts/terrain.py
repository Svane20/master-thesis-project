import time

from mathutils import Vector
from mathutils.noise import fractal

import numpy as np
from typing import Tuple
import logging

from numpy import ndarray

from consts import Constants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_segmentation(
        world_size: int = int(Constants.Default.WORLD_SIZE),
        num_octaves: Tuple[int, int] = (3, 4),
        H: Tuple[float, float] = (0.4, 0.5),
        lacunarity: Tuple[float, float] = (1.1, 1.2),
        image_size: int = Constants.Default.IMAGE_SIZE,
        band: int = 48,
        noise_basis: str = Constants.Default.NOISE_BASIS
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generates a height map and a segmentation map for the terrain.

    Args:
        world_size: Size of the terrain in Blender units.
        num_octaves: Number of octaves for fractal noise.
        H: Controls roughness of the fractal noise.
        lacunarity: Frequency multiplier for successive noise layers.
        image_size: Size of the resulting image (height map resolution).
        band: Threshold band around the midpoint to classify different areas.
        noise_basis: Type of noise basis (e.g., "PERLIN_ORIGINAL").

    Returns:
        height_map: The normalized height map (0-1).
        seg_map: A 3-channel segmentation map (RGB).
        masks: Tuple of binary masks (grass, non-grass, beds).
    """

    timeit = time.time()

    # Step 1: Generate normalized height map
    height_map, seg_map, (grass, not_grass, beds) = _generate_height_map(
        world_size=world_size,
        image_size=image_size,
        num_octaves=num_octaves,
        H=H,
        lacunarity=lacunarity,
        noise_basis=noise_basis
    )

    logger.info(f"New Method time taken: {time.time() - timeit:.2f} seconds")

    return height_map, seg_map, (grass, not_grass, beds)


def _generate_height_map(
        world_size: int = int(Constants.Default.WORLD_SIZE),
        num_octaves: Tuple[int, int] = (3, 4),
        H: Tuple[float, float] = (0.4, 0.5),
        lacunarity: Tuple[float, float] = (1.1, 1.2),
        image_size: int = Constants.Default.IMAGE_SIZE,
        band: int = 48,
        noise_basis: str = Constants.Default.NOISE_BASIS
) -> tuple[ndarray | ndarray, ndarray | ndarray, tuple[ndarray | ndarray, ndarray | ndarray, ndarray | ndarray]]:
    """
    Generates a normalized height map based on fractal noise.

    Args:
        world_size: Size of the terrain in Blender units.
        num_octaves: Number of octaves for fractal noise.
        H: Controls roughness of the fractal noise.
        lacunarity: Frequency multiplier for successive noise layers.
        image_size: Size of the resulting image (height map resolution).
        noise_basis: Type of noise basis (e.g., "PERLIN_ORIGINAL").

    Returns:
        A height map (2D array) normalized to [0, 1] representing terrain heights.
    """
    np.random.seed(42)

    # Generate a grid of points
    grid = np.linspace(-world_size / 2, world_size / 2, image_size, endpoint=True)

    # Randomize fractal noise parameters
    restart = np.random.randint(low=world_size // 3, high=world_size // 2)
    num_octaves = int(np.random.randint(*num_octaves))
    H = np.random.uniform(*H)
    lacunarity = np.random.uniform(*lacunarity)
    offset = np.random.randint(low=0, high=world_size // 2)

    # Vectorized fractal noise calculation
    d_map = []
    for x in grid:
        d_row = []
        for y in grid:
            z = fractal(
                Vector((x / restart + offset, y / restart + offset, 0)),
                H,
                lacunarity,
                num_octaves,
                noise_basis=noise_basis,
            )
            d_row.append(z)
        d_map.append(d_row)

    # Convert the result to a numpy array
    height_map = np.array(d_map)

    # Normalize the height map to [0, 1]
    height_min = np.min(height_map)
    height_max = np.max(height_map)
    normalized_height_map = (height_map - height_min) / (height_max - height_min)

    return _generate_segmentation_map(normalized_height_map, band=band)


def _generate_segmentation_map(
        height_map: np.ndarray,
        band: int = 48,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Creates a segmentation map from the height map based on thresholds.

    Args:
        height_map: The normalized height map (0-1).
        band: Threshold band around the midpoint to classify different areas.

    Returns:
        scaled_height_map: The scaled height map (0-1).
        seg_map: A 3-channel segmentation map (RGB).
        masks: Tuple of binary masks (grass, non-grass, beds).
    """
    np.random.seed(42)

    # Scale the height map to [0, 255]
    scaled_height_map = height_map * 255

    # Precompute threshold values
    lower_band = 128 - band
    upper_band = 128 + band

    # Create binary masks for grass, not grass, and beds as uint8 arrays
    grass = ((scaled_height_map > lower_band) & (scaled_height_map < upper_band)).astype(np.uint8) * 255
    not_grass = (scaled_height_map >= upper_band).astype(np.uint8) * 255
    beds = (scaled_height_map <= lower_band).astype(np.uint8) * 255

    # Create segmentation map with 3 channels (R: not_grass, G: grass, B: beds) by stacking the channels
    seg_map = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype=np.uint8)
    seg_map[..., 0] = not_grass
    seg_map[..., 1] = grass
    seg_map[..., 2] = beds

    # Normalize the scaled height map to [0, 1]
    scaled_height_map /= 255

    return scaled_height_map, seg_map, (grass, not_grass, beds)
