import io
from PIL import Image
import numpy as np
from fastapi import HTTPException


def convert_image_to_png(image: np.ndarray) -> bytes:
    """Convert an image (assumed in [0,1] range) to PNG bytes."""
    image_uint8 = (image * 255).astype(np.uint8) if image.dtype != 'uint8' else image
    pil_img = Image.fromarray(image_uint8)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def convert_mask_to_rgb(mask: np.ndarray) -> bytes:
    """Convert a 2D mask to a 3-channel RGB PNG image with contrast stretching."""
    norm_mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask_uint8 = (norm_mask * 255).astype(np.uint8)
    rgb = np.stack([mask_uint8] * 3, axis=-1)
    pil_img = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


async def read_image_async(file) -> Image.Image:
    """Asynchronously read an UploadFile and return a PIL Image."""
    try:
        image_bytes = await file.read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")
