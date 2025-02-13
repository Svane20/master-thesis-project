from mathutils import Vector
from mathutils.noise import fractal
from PIL import Image
from pydelatin import Delatin
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


def create_delatin_mesh_from_height_map(height_map: NDArray[np.float32], seed: int = None) -> Delatin:
    """
    Create meshes using the Delatin algorithm.

    Args:
        height_map (NDArray[np.float32]): The height map (terrain) to create meshes from.
        seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
        Delatin: The Delatin object with vertices and triangles representing the mesh.
    """
    logger.info("Creating Delatin mesh from terrain...")

    if seed is not None:
        np.random.seed(seed)
        logger.info(f"Seed set to {seed}")

    width, height = height_map.shape
    logger.info(f"Terrain shape: width={width}, height={height}")

    delatin = Delatin(height_map * np.random.uniform(low=1, high=2.5), width=width, height=height)

    logger.info(f"Delatin mesh created: {len(delatin.vertices)} vertices, {len(delatin.triangles)} faces.")

    return delatin


def create_terrain_segmentation(
        world_size: int,
        image_size: int,
        noise_basis: str,
        num_octaves: Tuple[int, int] = (3, 4),
        H: Tuple[float, float] = (0.4, 0.5),
        lacunarity: Tuple[float, float] = (1.1, 1.2),
        band: int = 48,
        seed: int = None
) -> Tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """
    Generates a fractal height map and a segmentation map for the terrain.

    Args:
        world_size (int): Size of the terrain in Blender units. Default is WorldDefaults.SIZE.
        num_octaves (Tuple[int, int]): Number of octaves for fractal noise. Default is (3, 4).
        H (Tuple[float, float]): Controls roughness of the fractal noise. Default is (0.4, 0.5).
        lacunarity (Tuple[float, float]): Frequency multiplier for successive noise layers. Default is (1.1, 1.2).
        image_size (int): Size of the resulting image (height map resolution). Default is ImageDefaults.SIZE.
        band (int): Threshold band around the midpoint to classify different areas. Default is 48.
        noise_basis (str): Type of noise basis (e.g., "PERLIN_ORIGINAL"). Default is "PERLIN_ORIGINAL".
        seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
        Tuple[NDArray[np.float32], NDArray[np.uint8]]: Normalized height map and segmentation map.
    """
    logger.info("Generating terrain segmentation...")

    normalized_height_map = _generate_fractal_heightmap(
        world_size=world_size,
        image_size=image_size,
        num_octaves=num_octaves,
        H=H,
        lacunarity=lacunarity,
        noise_basis=noise_basis,
        seed=seed
    )

    return _create_segmentation_map(
        height_map=normalized_height_map,
        band=band
    )


def _generate_fractal_heightmap(
        world_size: float,
        image_size: int,
        noise_basis: str,
        num_octaves: Tuple[int, int] = (3, 4),
        H: Tuple[float, float] = (0.4, 0.5),
        lacunarity: Tuple[float, float] = (1.1, 1.2),
        seed: int = None
) -> NDArray[np.float32]:
    """
    Generates a normalized height map based on fractal noise.

    Args:
        world_size (float): Size of the terrain in Blender units. Default is WorldDefaults.SIZE.
        num_octaves (Tuple[int, int]): Number of octaves for fractal noise. Default is (3, 4).
        H (Tuple[float, float]): Controls roughness of the fractal noise. Default is (0.4, 0.5).
        lacunarity (Tuple[float, float]): Frequency multiplier for successive noise layers. Default is (1.1, 1.2).
        image_size (int): Size of the resulting image (height map resolution). Default is ImageDefaults.SIZE.
        noise_basis (str): Type of noise basis (e.g., "PERLIN_ORIGINAL"). Default is "PERLIN_ORIGINAL".
        seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
        NDArray[np.float32]: A height map (2D array) normalized to [0, 255] representing terrain heights.
    """
    logger.info("Generating fractal height map...")

    if seed is not None:
        np.random.seed(seed)
        logger.info(f"Seed set to {seed}")

    # Generate a grid of points
    grid = np.linspace(-world_size / 2, world_size / 2, 1000, endpoint=True)

    # Randomize fractal noise parameters
    restart = np.random.randint(world_size // 3, world_size // 2)
    num_octaves = int(np.random.randint(*num_octaves))
    H = np.random.uniform(*H)
    lacunarity = np.random.uniform(*lacunarity)
    offset = np.random.randint(low=0.0, high=world_size // 2)

    logger.info(f"Fractal noise parameters: num_octaves={num_octaves}, H={H}, lacunarity={lacunarity}, offset={offset}")

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

    logger.info(f"Fractal height map generated with dimensions: {normalized_height_map.shape}")

    return normalized_height_map


def _create_segmentation_map(
        height_map: NDArray[np.float32],
        band: int = 48,
) -> Tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """
    Creates a segmentation map from the height map based on thresholds.

    Args:
        height_map (NDArray[np.float32]): The normalized height map (0-255).
        band (int): Threshold band around the midpoint to classify different areas. Default is 48.

    Returns:
        Tuple[NDArray[np.float32], NDArray[np.uint8]]: Normalized height map and RGB segmentation map.
    """
    logger.info("Creating segmentation map...")

    # Threshold values
    lower_band = 128 - band
    upper_band = 128 + band

    # Create binary masks for different segments
    grass = np.asarray((height_map > lower_band) & (height_map < upper_band)).astype(np.uint8) * 255
    texture = np.asarray(height_map >= upper_band).astype(np.uint8) * 255
    beds = np.asarray(height_map <= lower_band).astype(np.uint8) * 255

    # Create segmentation map with 3 channels (R: texture, G: grass, B: beds)
    segmentation_map = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype=np.uint8)
    segmentation_map[..., 0] = texture
    segmentation_map[..., 1] = grass
    segmentation_map[..., 2] = beds

    # Normalize the height map to [0, 1] for further use
    height_map /= 255

    logger.info(f"Segmentation map created with dimensions: {segmentation_map.shape}")

    return height_map, segmentation_map
