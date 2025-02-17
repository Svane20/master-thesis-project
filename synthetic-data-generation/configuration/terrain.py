from pydantic import BaseModel
from typing import List


class TreesConfiguration(BaseModel):
    """
    Configuration for grass generation in the scene

    Attributes:
        directory (str): The directory where the tree files are located
        keywords (List[str]): The keywords get specific tree files
    """
    directory: str
    keywords: List[str] | None


class GrassConfiguration(BaseModel):
    """
    Configuration for grass generation in the scene

    Attributes:
        directory (str): The directory where the grass files are located
        keywords (List[str]): The keywords get specific grass files
    """
    directory: str
    keywords: List[str] | None


class NotGrassConfiguration(BaseModel):
    """
    Configuration for non-grass (e.g. flowers, mulch) generation in the scene

    Attributes:
        directory (str): The directory where the grass files are located
        keywords (List[str]): The keywords get specific grass files
    """
    directory: str
    keywords: List[str] | None


class TexturesConfiguration(BaseModel):
    """
    Configuration for texture generation in the scene

    Attributes:
        directories (List[str]): The directories where the texture files are located
    """
    directories: List[str]


class TerrainConfiguration(BaseModel):
    """
    Configuration for terrain generation in the scene.

    Attributes:
        world_size (float): The size of the terrain (world).
        image_size (int): The size of the image or terrain representation.
        noise_basis (int): The size of the noise basis.
        generate_trees (bool): Whether to generate trees.
        trees_configuration (TreesConfiguration): The configuration of the trees.
        grass_configuration (GrassConfiguration): The configuration of the grass.
        not_grass_configuration (NotGrassConfiguration): The configuration of the not grass.
    """
    world_size: float
    image_size: int
    noise_basis: str
    generate_trees: bool
    trees_configuration: TreesConfiguration
    grass_configuration: GrassConfiguration
    not_grass_configuration: NotGrassConfiguration
    textures_configuration: TexturesConfiguration
