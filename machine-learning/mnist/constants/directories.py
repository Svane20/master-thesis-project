from pathlib import Path

BASE_DIRECTORY: Path = Path(__file__).resolve().parent.parent

MODELS_DIRECTORY = BASE_DIRECTORY / "models"
DATA_DIRECTORY = BASE_DIRECTORY / "data"