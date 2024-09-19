from pathlib import Path

CONFIG_DIR: Path = Path(__file__).resolve().parent / "configs"
CONFIG_PATH: str = (CONFIG_DIR / "config.json").as_posix()

OUTPUT_DIR: Path = Path(__file__).resolve().parent / "output"
OUTPUT_PATH: str = (OUTPUT_DIR / "output.png").as_posix()

BLENDER_FILES_DIR: Path = Path(__file__).resolve().parent / "blender_files"
BLENDER_FILES_PATH: str = (BLENDER_FILES_DIR / "output.blend").as_posix()