import bpy

from pathlib import Path
from typing import List, Union, Tuple
import numpy as np

CURRENT_DIRECTORY: Path = Path(__file__).resolve().parent
ASSETS_DIRECTORY: Path = CURRENT_DIRECTORY / "assets"
BIOMES_DIRECTORY: Path = ASSETS_DIRECTORY / "biomes"


def get_all_biomes(directory: Path = BIOMES_DIRECTORY) -> List[str]:
    print(f"Searching for biomes in {directory}")

    paths = [str(f) for f in directory.rglob("*") if str(f).endswith(".biome")]

    return paths


def apply_biome(
        bpy_object: Union[str, bpy.types.Object],
        path: str,
        density: Tuple[float, float] = (20., 30.),
        label_index: int = 0,
) -> None:
    # Ensure the correct object is referenced
    if isinstance(bpy_object, str):
        bpy_object = bpy.data.objects.get(bpy_object)
        if bpy_object is None:
            raise ValueError(f"Object '{bpy_object}' not found in the scene.")

    # Set the object as the new emitter
    bpy.ops.scatter5.set_new_emitter(obj_name=bpy_object.name)

    # Access the active scene (safer than hardcoding 'Scene')
    scene = bpy.context.scene

    # Apply random density distribution
    scene.scatter5.operators.add_psy_density.f_distribution_density = np.random.uniform(*density)

    # Force view layer update
    bpy.context.view_layer.update()

    print(f"Applying biome {path} to {bpy_object.name}")

    # Track new objects added after scattering
    before_scattering = set(bpy.data.objects.keys())

    # Add the biome to the emitter
    bpy.ops.scatter5.add_biome(emitter_name=bpy_object.name, json_path=path)

    after_scattering = set(bpy.data.objects.keys())
    new_objects = after_scattering - before_scattering

    # Assign pass index to new scattered objects
    for object_name in new_objects:
        scattered_object = bpy.data.objects[object_name]
        scattered_object.pass_index = label_index
