from pathlib import Path

BASE_DIRECTORY: Path = Path(__file__).resolve().parent.parent

CHECKPOINTS_DIRECTORY = BASE_DIRECTORY / "checkpoints"
DATA_DIRECTORY = BASE_DIRECTORY / "data"
