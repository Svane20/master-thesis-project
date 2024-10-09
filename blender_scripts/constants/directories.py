from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parent.parent
CONFIG_DIR: Path = BASE_DIR / "configuration"
CONFIG_PATH: Path = CONFIG_DIR / "config.json"
OUTPUT_DIR: Path = BASE_DIR / "output"
TEMP_DIR: Path = BASE_DIR / "temp"
PLAYGROUND_DIR: Path = BASE_DIR / "playground"
BLENDER_FILES_DIR: Path = BASE_DIR / "blender_files"
ASSETS_DIR: Path = BASE_DIR / "assets"
ASSETS_TEXTURES_DIR: Path = ASSETS_DIR / "textures"
ASSETS_TEXTURES_GRASS_DIR: Path = ASSETS_TEXTURES_DIR / "grass"
