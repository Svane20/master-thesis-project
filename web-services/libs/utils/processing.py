from fastapi import UploadFile, HTTPException
import numpy as np
from PIL import Image
import io

from libs.logging import logger

IMG_SIZE = (512, 512)


async def load_image(file: UploadFile) -> Image:
    """
    Load an image from an upload file and convert it to a NumPy array.

    Args:
        file (UploadFile): The file containing the image.

    Returns:
        Image: The loaded image as a PIL Image object.
    """
    try:
        image_bytes = await file.read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.error({"event": "load_image_failed", "error": str(e)})
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")


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
