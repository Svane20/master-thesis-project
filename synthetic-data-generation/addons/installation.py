import bpy
from pathlib import Path
from typing import List
import logging

from configuration.addons import AddonConfiguration


def install_addons(addons: List[AddonConfiguration]) -> None:
    """
    Installs the required addons and reloads the biome library in Blender.
    """
    logging.info("Starting addons installation process.")

    for addon in addons:
        if addon.plugin_path:
            _install_addon(addon.plugin_title, addon.plugin_path)

        if addon.library_paths:
            _add_asset_libraries(addon.library_paths)

        if addon.package_path:
            _install_package(addon.package_path)

    bpy.ops.scatter5.reload_biome_library()
    logging.info("Addons installation complete. Biome library reloaded.")


def _install_addon(title: str, filepath: str) -> None:
    """
    Installs the addon from the given filepath.

    Args:
        title (str): The name of the addon.
        filepath (Path): The path to the addon file.
    """
    logging.debug(f"Installing {title} addon from {filepath}")

    try:
        bpy.ops.preferences.addon_install(filepath=filepath, overwrite=False)
        bpy.ops.preferences.addon_enable(module=title)
        bpy.ops.preferences.addon_refresh()
        logging.debug(f"{title} addon installed and enabled successfully.")
    except Exception as e:
        logging.error(f"Failed to install {title} addon: {e}")


def _install_package(filepath: str) -> None:
    """
    Installs the package from the given filepath.

    Args:
        title (str): The name of the package.
        filepath (Path): The path to the package file.
    """
    path = Path(filepath)
    title = path.name
    logging.debug(f"Installing {title} package from {filepath}")

    try:
        bpy.ops.scatter5.install_package(filepath=filepath, popup_menu=False)
        logging.debug(f"{title} package installed successfully.")
    except Exception as e:
        logging.error(f"Failed to install {title} package: {e}")


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
            logging.debug(f"Adding asset library {path.name} from {directory}")
            try:
                bpy.ops.preferences.asset_library_add(directory=directory)
                logging.debug(f"Asset library {dir_name} added successfully.")
            except Exception as e:
                logging.error(f"Failed to add asset library {dir_name}: {e}")
        else:
            logging.debug(f"Asset library {dir_name} already exists.")
