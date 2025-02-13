from pydantic import BaseModel
from typing import List


class AddonConfiguration(BaseModel):
    """
    Configuration class for managing the plant library in Blender.
    """
    install: bool
    plugin_title: str
    plugin_path: str
    package_path: str
    library_paths: List[str]


class AddonsConfiguration(BaseModel):
    """
    Configuration class for installing and managing Blender add-ons.

    Attributes:

    """
    biome_reader_configuration: AddonConfiguration
