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
        BASE_DIR: Path = Path(__file__).resolve().parent.parent
        CONFIG_DIR: Path = BASE_DIR / "configuration"
        CONFIG_PATH: Path = CONFIG_DIR / "config.json"
        OUTPUT_DIR: Path = BASE_DIR / "output"
        TEMP_DIR: Path = BASE_DIR / "temp"
        BLENDER_FILES_DIR: Path = BASE_DIR / "blender_files"
        ASSETS_DIR: Path = BASE_DIR / "assets"
        ASSETS_TEXTURES_DIR: Path = ASSETS_DIR / "textures"
        ASSETS_TEXTURES_GRASS_DIR: Path = ASSETS_TEXTURES_DIR / "grass"

    class Render:
        THREADS_MODE: str = "FIXED"
        THREADS: int = 54
        FILE_FORMAT: str = "PNG"
        COLOR_MODE: str = "RGBA"
        COMPRESSION: int = 0

    class FileExtension:
        PNG: str = "png"
        BLEND: str = "blend"
