import bpy

from pathlib import Path
from typing import List
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path("C:\\Thesis")
PLUGIN_DIR = BASE_DIR / "plugins"
PACKAGES_DIR = BASE_DIR / "packages"

ASSETS_DIR = BASE_DIR / "assets"

PLANT_LIBRARY_DIR = ASSETS_DIR / "data" / "plant_library"
ASSET_PLANT_LIBRARY_LIBRARIES = [PLANT_LIBRARY_DIR]

VEGETATION_DIR = ASSETS_DIR / "data" / "vegetation"
ASSET_VEGETATION_LIBRARIES = [VEGETATION_DIR]


def install_addons() -> None:
    """Installs the required addons."""
    install_biome_reader()
    install_vegetation()

    bpy.ops.scatter5.reload_biome_library()


def install_biome_reader() -> None:
    """Installs the Biome-Reader addon."""
    plugin_title = "Biome-Reader"
    plugin_path = PLUGIN_DIR / "BiomeReaderPlugin.zip"
    install_addon(plugin_title, plugin_path)

    add_asset_library(plugin_title, ASSET_PLANT_LIBRARY_LIBRARIES)

    package_title = "plant_library.scatpack"
    package_path = PACKAGES_DIR / package_title
    install_package(package_title, package_path)


def install_vegetation() -> None:
    """Installs the Vegetation addon."""
    plugin_title = "Vegetation"
    plugin_path = PLUGIN_DIR / "Vegetation_V5.1_Addon.zip"
    install_addon(plugin_title, plugin_path)

    add_asset_library(plugin_title, ASSET_VEGETATION_LIBRARIES)

    package_title = "Vegetation_v5.1fix2_Geoscatter_Biomes_Pro.scatpack"
    package_path = PACKAGES_DIR / package_title
    install_package(package_title, package_path)


def install_addon(title: str, filepath: Path) -> None:
    """
    Installs the addon from the given filepath.

    Args:
        title: The title of the addon.
        filepath: The filepath of the addon.
    """
    logger.info(f"Installing {title} addon from {filepath}")

    bpy.ops.preferences.addon_install(filepath=filepath.as_posix(), overwrite=False)
    bpy.ops.preferences.addon_enable(module=title)
    bpy.ops.preferences.addon_refresh()


def install_package(title: str, filepath: Path) -> None:
    """
    Installs the package from the given filepath.

    Args:
        title: The title of the package.
        filepath: The filepath of the package.
    """
    logger.info(f"Installing {title} package from {filepath}")

    try:
        bpy.ops.scatter5.install_package(filepath=filepath.as_posix(), popup_menu=False)
        logger.info(f"Added {title}")
    except Exception as e:
        logger.error(f"Failed to add {title}: {e}")


def add_asset_library(title: str, directories: List[Path]) -> None:
    """
    Adds the asset library from the given directories.

    Args:
        title: The title of the asset library.
        directories: The directories of the asset library.
    """
    for directory in directories:
        dir_name = directory.name
        dir_path = directory.as_posix()

        if dir_name not in bpy.context.preferences.filepaths.asset_libraries.keys():
            logger.info(f"Installing {dir_name} package from {dir_path}")

            bpy.ops.preferences.asset_library_add(directory=dir_path)

            logger.info(f"Added asset library: {title}")
        else:
            logger.info(f"Asset library {dir_name} already exists")
