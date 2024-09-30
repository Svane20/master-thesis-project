import time

import bpy
from mathutils import Vector
from mathutils.noise import fractal

import logging
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image


from consts import Constants
from rendering.rendering import setup_rendering
from configs.configuration import Configuration, save_configuration, load_configuration
from terrain import generate_segmentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def generate_segmentation_map(
        world_size: float = 100.0,
        n_octaves: tuple[int, int] = (3, 4),
        H: tuple[float, float] = (0.4, 0.5),
        lacunarity: tuple[float, float] = (1.1, 1.2),
        img_size: int = 2048,
        band: int = 48,
        noise_basis: str = "PERLIN_ORIGINAL"
):
    timeit = time.time()

    grid = np.linspace(-world_size / 2, world_size / 2, 1000, endpoint=True)

    np.random.seed(42)

    restart = np.random.randint(world_size // 3, world_size // 2)
    n_octaves = int(np.random.randint(*n_octaves))
    H = np.random.uniform(*H)
    lacunarity = np.random.uniform(*lacunarity)
    offset = np.random.randint(0, world_size // 2)

    d_map = []
    for x in grid:
        d_row = []
        for y in grid:
            z = fractal(
                Vector((x / restart + offset, y / restart + offset, 0)),
                H,
                lacunarity,
                n_octaves,
                noise_basis=noise_basis,
            )
            d_row.append(z)
        d_map.append(d_row)
    d_map = np.array(d_map)

    d_map = Image.fromarray(d_map).resize((img_size,) * 2, Image.Resampling.BILINEAR)
    d_map = np.array(d_map)

    d_min = np.min(d_map)
    d_max = np.max(d_map)

    t_gen = (d_map - d_min) / (d_max - d_min) * 255

    grass = (t_gen < (128 + band)) * (t_gen > (128 - band))
    grass = grass.astype(np.uint8) * 255

    not_grass = t_gen >= (128 + band)
    not_grass = not_grass.astype(np.uint8) * 255

    beds = t_gen <= (128 - band)
    beds = beds.astype(np.uint8) * 255

    seg_map = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    seg_map[..., 0] = not_grass
    seg_map[..., 1] = grass
    seg_map[..., 2] = beds

    t_gen /= 255

    logger.info(f"Old method Time taken: {time.time() - timeit:.2f} seconds")

    return t_gen, seg_map, (grass, not_grass, beds)


def clear_cube() -> None:
    """Clear the cube object if it exists"""
    cube: bpy.types.Object = bpy.data.objects.get("Cube")
    if cube is not None:
        cube.select_set(True)
        bpy.ops.object.delete()


def setup() -> None:
    clear_cube()

    # Load settings
    config = load_configuration(path=Constants.Directory.CONFIG_PATH)
    if config is None:
        config = save_configuration(configuration=Configuration().model_dump(), path=Constants.Directory.CONFIG_PATH)

    configuration = Configuration(**config)
    terrain_config = configuration.terrain_configuration

    setup_rendering(
        render_configuration=configuration.render_configuration,
        camera_configuration=configuration.camera_configuration,
    )

    new = generate_segmentation(world_size=int(terrain_config.world_size), image_size=terrain_config.image_size)
    old = generate_segmentation_map(world_size=int(terrain_config.world_size), img_size=terrain_config.image_size)

    plt.hist(old[0].flatten(), bins=50, alpha=0.7, label='Old Method')
    plt.hist(new[0].flatten(), bins=50, alpha=0.5, label='New Method')
    plt.legend(loc='upper right')
    plt.title('Distribution of Height Map Values')
    plt.show()


def main() -> None:
    setup()


if __name__ == "__main__":
    main()
