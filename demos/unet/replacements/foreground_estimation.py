import pymatting
import numpy as np
from PIL import Image


def get_foreground_estimation(image: Image, alpha: np.ndarray) -> np.ndarray:
    """
    Estimate the foreground of an image.

    Args:
        image (np.ndarray): The image to estimate the foreground of.
        alpha (numpy.ndarray): The predicted alpha matte with values between 0 and 1.

    Returns:
        numpy.ndarray: Foreground image.

    """
    normalized_image = np.array(image) / 255.0
    inverted_alpha = 1 - alpha

    return pymatting.estimate_foreground_ml(image=normalized_image, alpha=inverted_alpha)
