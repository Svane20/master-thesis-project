import bpy
from pathlib import Path
from typing import List

from configuration.addons import AddonConfiguration
from custom_logging.custom_logger import setup_logger

# Set up logger
logger = setup_logger(__name__)


def install_addons(addons: List[AddonConfiguration]) -> None:
    """
    Installs the required addons and reloads the biome library in Blender.
    """
    logger.info("Starting addons installation process.")

    for addon in addons:
        if not addon.install:
            continue

        _install_addon(addon.plugin_title, addon.plugin_path)
        _add_asset_libraries(addon.library_paths)
        _install_package(addon.package_path)

    bpy.ops.scatter5.reload_biome_library()
    logger.info("Addons installation complete. Biome library reloaded.")


def _install_addon(title: str, filepath: str) -> None:
    """
    Installs the addon from the given filepath.

    Args:
        title (str): The name of the addon.
        filepath (Path): The path to the addon file.
    """
    logger.info(f"Installing {title} addon from {filepath}")

    try:
        bpy.ops.preferences.addon_install(filepath=filepath, overwrite=False)
        bpy.ops.preferences.addon_enable(module=title)
        bpy.ops.preferences.addon_refresh()
        logger.info(f"{title} addon installed and enabled successfully.")
    except Exception as e:
        logger.error(f"Failed to install {title} addon: {e}")


def _install_package(filepath: str) -> None:
    """
    Installs the package from the given filepath.

    Args:
        title (str): The name of the package.
        filepath (Path): The path to the package file.
    """
    path = Path(filepath)
    title = path.name
    logger.info(f"Installing {title} package from {filepath}")

    try:
        bpy.ops.scatter5.install_package(filepath=filepath, popup_menu=False)
        logger.info(f"{title} package installed successfully.")
    except Exception as e:
        logger.error(f"Failed to install {title} package: {e}")


def _add_asset_libraries(directories: List[str]) -> None:
    """
    Adds asset libraries to Blender's preferences.

    Args:
        directories (List[Path]): The directories containing the asset libraries.
    """
    for directory in directories:
        path = Path(directory)
        dir_name = path.name

        if dir_name not in bpy.context.preferences.filepaths.asset_libraries.keys():
            logger.info(f"Adding asset library {path.name} from {directory}")
            try:
                bpy.ops.preferences.asset_library_add(directory=directory)
                logger.info(f"Asset library {dir_name} added successfully.")
            except Exception as e:
                logger.error(f"Failed to add asset library {dir_name}: {e}")
        else:
            logger.info(f"Asset library {dir_name} already exists.")
