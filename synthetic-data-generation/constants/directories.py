from pathlib import Path

BASE_DIRECTORY: Path = Path(__file__).resolve().parent.parent

# Configuration File
CONFIG_PATH: Path = BASE_DIRECTORY / "configuration.json"

# Output and Temporary Directories
OUTPUT_DIRECTORY: Path = BASE_DIRECTORY / "output"
TEMP_DIRECTORY: Path = BASE_DIRECTORY / "temp"
PLAYGROUND_DIRECTORY: Path = BASE_DIRECTORY / "playground"
BLENDER_FILES_DIRECTORY: Path = BASE_DIRECTORY / "blender_files"

# Asset Directories
ASSETS_DIRIRECTORY: Path = BASE_DIRECTORY / "assets"
BIOMES_DIRECTORY: Path = ASSETS_DIRIRECTORY / "biomes"

# HDRI Directories
HDRI_DIRECTORY: Path = ASSETS_DIRIRECTORY / "hdri"
HDRI_PURE_SKIES_DIRECTORY: Path = HDRI_DIRECTORY / "pure_skies"

# Model Directories
MODELS_DIRECTORY: Path = ASSETS_DIRIRECTORY / "models"
HOUSES_DIRECTORY: Path = MODELS_DIRECTORY / "houses"

# Texture Directories
ASSETS_TEXTURES_DIRECTORY: Path = ASSETS_DIRIRECTORY / "textures"
ASSETS_TEXTURES_GRASS_DIRECTORY: Path = ASSETS_TEXTURES_DIRECTORY / "grass"
