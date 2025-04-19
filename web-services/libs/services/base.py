from fastapi import UploadFile
import torch
from abc import ABC, abstractmethod
from typing import List

from libs.configuration.base import Configuration
from libs.fastapi.settings import Settings
from libs.utils.transforms import get_transforms


class BaseModelService(ABC):
    def __init__(self, settings: Settings, config: Configuration):
        """
        Initializes the BaseModelService with the given configuration.

        Args:
            settings: The configuration settings for the service.
        """
        self.settings = settings
        self.config = config

        self.transforms = get_transforms(
            size=self.config.model.transforms.image_size,
            mean=self.config.model.transforms.mean,
            std=self.config.model.transforms.std,
        )

        self.use_gpu = torch.cuda.is_available() and self.settings.USE_GPU
        self.project_name = self.config.project_info.project_name
        self.model_type = self.config.project_info.model_type
        self.hardware = "GPU" if self.use_gpu else "CPU"
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

    @abstractmethod
    def load_model(self) -> None:
        """Load the model from disk and initialize runtime resources."""
        pass

    @abstractmethod
    async def single_predict(self, file: UploadFile) -> bytes:
        """Perform inference on a single image and return a PNG (or similar) result."""
        pass

    @abstractmethod
    async def batch_predict(self, files: List[UploadFile]) -> bytes:
        """Perform inference on multiple images and return the results, for example, as a ZIP archive."""
        pass

    @abstractmethod
    async def sky_replacement(self, file: UploadFile, extra: bool = False) -> bytes:
        """Perform sky replacement on the input image and return the result."""
        pass

    @abstractmethod
    async def batch_sky_replacement(self, files: List[UploadFile], extra: bool = False) -> bytes:
        """Perform sky replacement on multiple images and return the results, for example, as a ZIP archive."""
        pass
