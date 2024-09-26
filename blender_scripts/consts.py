from pathlib import Path


class Constants:
    class Default:
        WORLD_SIZE: float = 100.0

        IMAGE_SIZE: int = 2048
        IMAGE_WIDTH: int = 2048
        IMAGE_HEIGHT: int = 2048
        IMAGE_FORMAT: str = "png"

        NOISE_THRESHOLD: float = 0.01
        NOISE_BASIS: str = "PERLIN_ORIGINAL"

    class Directory:
        BASE_DIR: Path = Path(__file__).resolve().parent

        CONFIG_DIR: Path = BASE_DIR / "configs"
        CONFIG_PATH: Path = CONFIG_DIR / "config.json"

        OUTPUT_DIR: Path = BASE_DIR / "output"
        BLENDER_FILES_DIR: Path = BASE_DIR / "blender_files"

        ASSETS_DIR: Path = BASE_DIR / "assets"
        ASSETS_TEXTURES_DIR: Path = ASSETS_DIR / "textures"
        ASSETS_TEXTURES_GRASS_DIR: Path = ASSETS_TEXTURES_DIR / "grass"
