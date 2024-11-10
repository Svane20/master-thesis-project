from pathlib import Path

BASE_DIRECTORY: Path = Path(__file__).resolve().parent.parent

CHECKPOINTS_DIRECTORY = BASE_DIRECTORY / "checkpoints"

DATA_DIRECTORY = BASE_DIRECTORY / "data"
DATA_TRAIN_DIRECTORY = DATA_DIRECTORY / "train"
DATA_TEST_DIRECTORY = DATA_DIRECTORY / "test"
