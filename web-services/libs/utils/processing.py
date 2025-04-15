from fastapi import UploadFile, HTTPException
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import io
from typing import List, Tuple, Generator, AsyncGenerator
import asyncio
import zipfile
import os

from libs.logging import logger


async def preprocess_images(files: List[UploadFile], transforms: T.Compose) -> torch.Tensor:
    """
    Preprocess a list of image files into a batch tensor.

    Args:
        files (List[UploadFile]): List of image files to preprocess.
        transforms (T.Compose): The transformations to apply.

    Returns:
        torch.Tensor: A batch tensor of preprocessed images.
    """
    images = await asyncio.gather(*[load_image(file) for file in files])
    batch_tensors = []
    for image in images:
        try:
            tensor = transforms(image)
            batch_tensors.append(tensor)
        except Exception as e:
            logger.error("Error processing image", exc_info=e)
            raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

    return torch.stack(batch_tensors)


async def preprocess_image(file: UploadFile, transforms: T.Compose) -> Tuple[torch.Tensor, Image.Image]:
    """
    Preprocess a single image file into a tensor.

    Args:
        file (UploadFile): The file containing the image.
        transforms (T.Compose): The transformations to apply.

    Returns:
        Tuple[torch.Tensor, Image]: A tuple containing the preprocessed image tensor and the original image.
    """
    try:
        image = await load_image(file)
        return transforms(image), image
    except Exception as e:
        logger.error("Error processing image", exc_info=e)
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")


async def load_image(file: UploadFile) -> Image.Image:
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


async def get_alphas_as_zip(alphas: List[np.ndarray], files: List[UploadFile]) -> AsyncGenerator[bytes, None]:
    """
    Convert a list of alpha arrays to a zip file containing PNG images.

    Args:
        alphas (List[np.ndarray]): List of alpha arrays to convert.
        files (List[UploadFile]): List of corresponding image files.

    Returns:
        Generator[bytes, Any, None]: A generator that yields chunks of the zip file.
    """
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file, alpha in zip(files, alphas):
            base, _ = os.path.splitext(file.filename)
            filename = f"{base}.png"
            png_bytes = get_alpha_png_bytes(alpha)
            zip_file.writestr(filename, png_bytes)

    buffer.seek(0)

    while True:
        chunk = buffer.read(4096)
        if not chunk:
            break

        yield chunk


async def get_replacement_as_zip(
        replaced_image: bytes,
        alpha: np.ndarray,
        foreground: np.ndarray
) -> AsyncGenerator[bytes, None]:
    """
    Convert the replaced image and alpha array to a zip file containing PNG images.

    Args:
        replaced_image (bytes): The replaced image in bytes.
        alpha (np.ndarray): The alpha array.
        foreground (np.ndarray): The foreground image.

    Returns:
        AsyncGenerator[bytes, None]: A generator that yields chunks of the zip file.
    """
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("alpha.png", get_alpha_png_bytes(alpha))
        zip_file.writestr("foreground.png", get_image_png_bytes(foreground))
        zip_file.writestr("replaced.png", replaced_image)

    buffer.seek(0)

    while True:
        chunk = buffer.read(4096)
        if not chunk:
            break

        yield chunk


def get_alpha_png_bytes(alpha_array: np.ndarray) -> bytes:
    """
    Postprocess the alpha array to create a PNG image.

    Args:
        alpha_array (numpy.ndarray): The NumPy array representing the alpha.

    Returns:
        bytes: The PNG image data.
    """
    # Remove batch dimension if present
    alpha = alpha_array.squeeze()

    # Ensure the mask is in the range [0, 1]
    alpha_min, alpha_max = alpha.min(), alpha.max()
    if alpha_max > alpha_min:
        alpha = (alpha - alpha_min) / (alpha_max - alpha_min + 1e-8)

    # Convert to uint8 format
    mask_uint8 = (alpha * 255).astype(np.uint8)

    # Create a 3-channel RGB image
    rgb = np.stack([mask_uint8] * 3, axis=-1)

    # Resize the mask to the original size if provided
    pil_img = Image.fromarray(rgb, mode="RGB")

    # Save the mask as a PNG image in memory
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def get_image_png_bytes(image: np.ndarray) -> bytes:
    """
    Convert a NumPy array to a PNG image.

    Args:
        image (numpy.ndarray): The NumPy array representing the image.

    Returns:
        bytes: The PNG image data.
    """
    # Ensure the image is in uint8 format
    if image.dtype != 'uint8':
        image = (image * 255).astype(np.uint8)

    # Convert to PIL Image
    pil_img = Image.fromarray(image)

    # Save the image as a PNG in memory
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()
