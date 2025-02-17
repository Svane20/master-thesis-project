from typing import List
from pathlib import Path
import logging


def get_all_textures_by_directory(directory: str, keywords: List[str] | None = None) -> List[str]:
    """
    Get all texture files in the specified directory.

    Args:
        directory (Path): The directory to search for texture files.
        keywords (List[str], optional): A list of keywords to filter texture files. Defaults to None.

    Returns:
        List[str]: A list of biome file paths.
    """
    paths = [str(f) for f in Path(directory).rglob("*") if str(f).endswith(".blend")]

    if keywords:
        paths = [path for path in paths if any(keyword in path for keyword in keywords)]

    logging.info(f"Found {len(paths)} textures in {directory}")
    return paths
