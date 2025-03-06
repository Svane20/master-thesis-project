import json
from pathlib import Path
from enum import Enum
import logging


class MetadataKey(Enum):
    BIOMES = "biomes"
    TEXTURES = "textures"
    OBJECTS = "objects"
    HDRIS = "hdris"


# Global dictionary to store file paths.
METADATA = {
    MetadataKey.BIOMES.value: [],
    MetadataKey.TEXTURES.value: [],
    MetadataKey.OBJECTS.value: [],
    MetadataKey.HDRIS.value: [],
}


def add_entry(category: MetadataKey, filepath: str) -> None:
    """
    Append a filepath to the specified metadata category.

    Args:
        category (MetadataKey): The metadata category.
        filepath (str): The file path to append.
    """
    key = category.value
    logging.debug(f"Adding {filepath} to metadata category {key}")

    if key in METADATA:
        METADATA[key].append(filepath)
    else:
        METADATA[key] = [filepath]

    logging.debug(f"Metadata category {key} for {filepath} added.")


def save_metadata(directory: Path) -> None:
    """
    Save the METADATA dictionary as a JSON file in the given directory.

    Args:
        directory (Path): The directory to save the metadata file.
    """
    metadata_file = directory / "metadata.json"

    try:
        logging.info(f"Saving metadata to {metadata_file}")

        with open(metadata_file, "w") as f:
            json.dump(METADATA, f, indent=4)

        logging.info(f"Saved metadata to {metadata_file}")
    except Exception as e:
        logging.error(f"Failed to save metadata: {e}")
        raise
