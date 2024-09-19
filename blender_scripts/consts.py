from pathlib import Path

CONFIG_DIR: Path = Path(__file__).resolve().parent / "configs"
CONFIG_PATH: str = (CONFIG_DIR / "config.json").as_posix()

OUTPUT_DIR: Path = Path(__file__).resolve().parent / "output"
OUTPUT_PATH: str = (OUTPUT_DIR / "output.png").as_posix()