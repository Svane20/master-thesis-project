import pymatting
import numpy as np


def get_foreground_estimation(image: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
    """
    Estimate the foreground and background of an image.

    Args:
        image (np.ndarray): The input image.
        alpha_mask (numpy.ndarray): Alpha mask with values between 0 and 1.

    Returns:
        numpy.ndarray: Foreground image.
        numpy.ndarray: Background image.

    """
    # IMPORTANT: In our training data, the alpha matte is generated so that the sky/HDRI (our target subject)
    # is marked as 1 and the foreground/geometry is 0. However, the pymatting.estimate_foreground_ml function
    # is designed to extract the "foreground" (the object of interest) as the opaque region (alpha = 1).
    #
    # In our sky replacement pipeline, we want to preserve the geometry (foreground) from the original image
    # and replace the sky (background) with a new sky. To achieve this using pymatting, we need to extract
    # the geometry as the foreground. Therefore, we invert the alpha mask (using 1 - alpha_mask) so that:
    #    - The original geometry (which was 0) becomes 1 (treated as the foreground),
    #    - The sky (which was 1) becomes 0 (treated as background).
    #
    # This inversion aligns the alpha mask with pymatting’s expectation for foreground extraction.
    inverted_alpha_mask = 1 - alpha_mask

    # Estimate the foreground and background
    foreground = pymatting.estimate_foreground_ml(image=image, alpha=inverted_alpha_mask)

    return foreground
