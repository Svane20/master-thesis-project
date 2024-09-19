from pathlib import Path


class Constants:
    class Default:
        WORLD_SIZE: float = 100.0
        IMAGE_SIZE: int = 2048
        IMAGE_WIDTH: int = 2048
        IMAGE_HEIGHT: int = 2048

        NOISE_THRESHOLD: float = 0.01
        NOISE_BASIS: str = "PERLIN_ORIGINAL"

    class Directory:
        BASE_DIR: Path = Path(__file__).resolve().parent

        CONFIG_DIR: Path = BASE_DIR / "configs"
        CONFIG_PATH: Path = CONFIG_DIR / "config.json"

        TEMP_DIR: Path = BASE_DIR / "tmp"

        OUTPUT_DIR: Path = BASE_DIR / "output"
        OUTPUT_PATH: Path = OUTPUT_DIR / "output.png"

        BLENDER_FILES_DIR: Path = BASE_DIR / "blender_files"
        BLENDER_FILES_PATH: Path = BLENDER_FILES_DIR / "output.blend"
