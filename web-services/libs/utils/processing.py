from fastapi import UploadFile
import numpy as np
from PIL import Image
import io

from libs.logging import logger

IMG_SIZE = (512, 512)


def load_image_to_array(file: UploadFile) -> np.ndarray:
    """
    Load an image from an upload file and convert it to a NumPy array.

    Args:
        file (UploadFile): The file containing the image.

    Returns:
        np.ndarray: The NumPy array representing the image.
    """
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as e:
        logger.error({"event": "load_image_failed", "error": str(e)})
        raise

    return np.array(image)


def preprocess_image(image_array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Preprocess the image array for model inference.

    Args:
        image_array (numpy.ndarray): The NumPy array representing the image.
        mean (numpy.ndarray): The mean values for normalization.
        std (numpy.ndarray): The standard deviation values for normalization.

    Returns:
        np.ndarray: The preprocessed image tensor.
    """
    im = Image.fromarray(image_array)

    # Resize the image to 512x512
    im = im.resize(IMG_SIZE, Image.BILINEAR)

    # Convert to float32 and scale to [0, 1]
    arr = np.array(im, dtype=np.float32) / 255.0

    # Normalize the image
    arr = (arr - mean) / std

    # Transpose to (1, 3, H, W)
    tensor = np.transpose(arr, (2, 0, 1))[np.newaxis, :]

    # Convert to float32
    return tensor.astype(np.float32)


def postprocess_mask(mask_array: np.ndarray, original_size=None) -> bytes:
    """
    Postprocess the mask array to create a PNG image.

    Args:
        mask_array (numpy.ndarray): The NumPy array representing the mask.
        original_size: The original size of the image (optional).

    Returns:
        bytes: The PNG image data.
    """
    # Remove batch dimension if present
    mask = mask_array.squeeze()

    # Ensure the mask is in the range [0, 1]
    mask_min, mask_max = mask.min(), mask.max()
    if mask_max > mask_min:
        mask = (mask - mask_min) / (mask_max - mask_min)

    # Convert to uint8 format
    mask_img = (mask * 255).astype(np.uint8)

    # Resize the mask to the original size if provided
    if original_size is not None:
        mask_pil = Image.fromarray(mask_img).resize(original_size, Image.BILINEAR)
    else:
        mask_pil = Image.fromarray(mask_img)

    # Save the mask as a PNG image in memory
    buffer = io.BytesIO()
    mask_pil.save(buffer, format="PNG")
    return buffer.getvalue()
