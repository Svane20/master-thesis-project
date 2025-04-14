from fastapi import UploadFile
from abc import ABC, abstractmethod

from libs.configuration.base import Configuration
from libs.fastapi.settings import Settings


class BaseModelService(ABC):
    def __init__(self, settings: Settings, config: Configuration):
        """
        Initializes the BaseModelService with the given configuration.

        Args:
            settings: The configuration settings for the service.
        """
        self.settings = settings
        self.config = config

    @abstractmethod
    def load_model(self) -> None:
        """Load the model from disk and initialize runtime resources."""
        pass

    @abstractmethod
    async def single_predict(self, file: UploadFile) -> bytes:
        """Perform inference on a single image and return a PNG (or similar) result."""
        pass

    # @abstractmethod
    # async def batch_predict(self, files: List[UploadFile]) -> bytes:
    #     """Perform inference on multiple images and return the results, for example, as a ZIP archive."""
    #     pass
    #
    # @abstractmethod
    # async def sky_replacement(self, file: UploadFile, extra: bool = False) -> bytes:
    #     """Perform sky replacement on the input image and return the result."""
    #     pass
