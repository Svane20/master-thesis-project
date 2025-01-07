import os
import bpy
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from custom_logging.custom_logger import setup_logger

# Load environment variables
load_dotenv()

# Set up logger
logger = setup_logger(__name__)

# Directories
BASE_DIR = Path(os.getenv("ADDONS_BASE_PATH"))
assert BASE_DIR is not None, "The ADDONS_BASE_PATH environment variable is not set."
PLUGIN_DIR = BASE_DIR / "plugins"
PACKAGES_DIR = BASE_DIR / "packages"
ASSETS_DIR = BASE_DIR / "assets"

PLANT_LIBRARY_DIR = ASSETS_DIR / "data" / "plant_library"
ASSET_PLANT_LIBRARY_LIBRARIES = [PLANT_LIBRARY_DIR]

VEGETATION_DIR = ASSETS_DIR / "data" / "vegetation"
ASSET_VEGETATION_LIBRARIES = [VEGETATION_DIR]


def install_addons() -> None:
    """
    Installs the required addons and reloads the biome library in Blender.
    """
    logger.info("Starting addon installation process.")

    _install_biome_reader()
    _install_vegetation()

    bpy.ops.scatter5.reload_biome_library()
    logger.info("Addon installation complete. Biome library reloaded.")


def _install_biome_reader() -> None:
    """
    Installs the Biome-Reader addon and its associated assets.
    """
    plugin_title = "Biome-Reader"
    plugin_path = PLUGIN_DIR / "BiomeReaderPlugin.zip"

    _install_addon(plugin_title, plugin_path)
    _add_asset_library(plugin_title, ASSET_PLANT_LIBRARY_LIBRARIES)

    package_title = "plant_library.scatpack"
    package_path = PACKAGES_DIR / package_title
    _install_package(package_title, package_path)


def _install_vegetation() -> None:
    """
    Installs the Vegetation addon and its associated assets.
    """
    plugin_title = "Vegetation"
    plugin_path = PLUGIN_DIR / "Vegetation_V5.1_Addon.zip"

    _install_addon(plugin_title, plugin_path)
    _add_asset_library(plugin_title, ASSET_VEGETATION_LIBRARIES)

    package_title = "Vegetation_v5.1fix2_Geoscatter_Biomes_Pro.scatpack"
    package_path = PACKAGES_DIR / package_title
    _install_package(package_title, package_path)


def _install_addon(title: str, filepath: Path) -> None:
    """
    Installs the addon from the given filepath.

    Args:
        title (str): The name of the addon.
        filepath (Path): The path to the addon file.
    """
    logger.info(f"Installing {title} addon from {filepath}")

    try:
        bpy.ops.preferences.addon_install(filepath=filepath.as_posix(), overwrite=False)
        bpy.ops.preferences.addon_enable(module=title)
        bpy.ops.preferences.addon_refresh()
        logger.info(f"{title} addon installed and enabled successfully.")
    except Exception as e:
        logger.error(f"Failed to install {title} addon: {e}")


def _install_package(title: str, filepath: Path) -> None:
    """
    Installs the package from the given filepath.

    Args:
        title (str): The name of the package.
        filepath (Path): The path to the package file.
    """
    logger.info(f"Installing {title} package from {filepath}")

    try:
        bpy.ops.scatter5.install_package(filepath=filepath.as_posix(), popup_menu=False)
        logger.info(f"{title} package installed successfully.")
    except Exception as e:
        logger.error(f"Failed to install {title} package: {e}")


def _add_asset_library(title: str, directories: List[Path]) -> None:
    """
    Adds asset libraries to Blender's preferences.

    Args:
        title (str): The name of the asset library.
        directories (List[Path]): The directories containing the asset libraries.
    """
    for directory in directories:
        dir_name = directory.name
        dir_path = directory.as_posix()

        if dir_name not in bpy.context.preferences.filepaths.asset_libraries.keys():
            logger.info(f"Adding asset library {dir_name} from {dir_path}")
            try:
                bpy.ops.preferences.asset_library_add(directory=dir_path)
                logger.info(f"Asset library {title} added successfully.")
            except Exception as e:
                logger.error(f"Failed to add asset library {dir_name}: {e}")
        else:
            logger.info(f"Asset library {dir_name} already exists.")
