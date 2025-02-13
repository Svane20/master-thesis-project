from pydantic import BaseModel
from typing import List


class AddonConfiguration(BaseModel):
    """
    Configuration class for managing the plant library in Blender.
    """
    plugin_title: str | None
    plugin_path: str | None
    package_path: str | None
    library_paths: List[str] | None


class AddonsConfiguration(BaseModel):
    """
    Configuration class for installing and managing Blender add-ons.

    Attributes:

    """
    biome_reader_configuration: AddonConfiguration
