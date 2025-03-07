from typing import List
from pathlib import Path
import logging


def get_all_textures_by_directory(
        directory: str,
        include: List[str] | None = None,
        exclude: List[str] | None = None,
) -> List[str]:
    """
    Get all texture files in the specified directory.

    Args:
        directory (Path): The directory to search for texture files.
        include (List[str], optional): A list of keywords that must be present in the file path. Defaults to None.
        exclude (List[str], optional): A list of keywords to filter out from the file paths. Defaults to None.

    Returns:
        List[str]: A list of biome file paths.
    """
    paths = [str(f) for f in Path(directory).rglob("*") if str(f).endswith(".blend")]

    if include:
        paths = [path for path in paths if any(keyword in path for keyword in include)]

    if exclude:
        paths = [path for path in paths if not any(keyword in path for keyword in exclude)]

    logging.debug(f"Found {len(paths)} textures in {directory}")
    return paths
