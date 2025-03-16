import torch
import torchvision.transforms as T
import onnxruntime as ort
from fastapi import UploadFile

import io
import os
import asyncio
from PIL import Image
import numpy as np
from typing import List
import zipfile

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

transforms = T.Compose([
    T.Resize(size=(512, 512)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

# Global model session
session: ort.InferenceSession


def load_model(model_path: str) -> None:
    """
    Load the ONNX model.

    Args:
        model_path (str): Path to the ONNX model.

    Returns:
        ort.InferenceSession: ONNX model session.
    """
    global session
    session = ort.InferenceSession(model_path)


async def predict(files: List[UploadFile]) -> bytes:
    """
    Asynchronously perform inference on a batch of images.

    Args:
        files (List[UploadFile]): List of input files.
    """
    # Pre-process the images concurrently
    batch = await _pre_process(files)

    # Get input and output names from the model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Offload the blocking ONNX inference call to a thread executor
    loop = asyncio.get_running_loop()
    outputs = await loop.run_in_executor(
        None,
        session.run,
        [output_name],
        {input_name: batch}
    )

    # Post-process the outputs
    masks = _post_process(outputs)

    # Create a zip archive in memory and add each PNG image using the input file names
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Use zip to iterate over files and corresponding masks
        for file, mask in zip(files, masks):
            base, _ = os.path.splitext(file.filename)
            filename = f"{base}_mask.png"
            png_bytes = _convert_mask_to_rgb(mask)
            zip_file.writestr(filename, png_bytes)

    zip_buffer.seek(0)
    return zip_buffer.read()


def _post_process(outputs: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Post-process the inference result.

    Args:
        outputs (np.ndarray): Predicted alpha masks.

    Returns:
        np.ndarray: Post-processed alpha mask
    """
    result = outputs[0]  # expected shape: (batch_size, 1, 512, 512)
    masks = []
    for i in range(result.shape[0]):
        # Remove the channel dimension (which is axis 0 of each individual sample)
        mask = np.squeeze(result[i], axis=0)  # each mask becomes (512, 512)
        masks.append(mask)
    return masks


async def _pre_process(files: List[UploadFile]) -> np.ndarray:
    """
    Asynchronously pre-process input images for inference.

    Args:
        files (List[UploadFile]): List of input files.

    Returns:
        np.ndarray: Batch tensor of pre-processed images
    """
    # Read files concurrently
    images = await asyncio.gather(*[_read_file(file) for file in files])

    # Apply transforms and stack into a batch tensor (shape: [N, 3, 512, 512])
    batch_tensor = torch.stack([transforms(image) for image in images])

    # Convert to numpy array
    return batch_tensor.numpy()


def _convert_mask_to_rgb(mask: np.ndarray) -> bytes:
    """
    Convert a 2D grayscale mask to a 3-channel RGB PNG image.

    Args:
        mask (np.ndarray): Grayscale mask with values between 0 and 1.

    Returns:
        bytes: PNG image bytes.
    """
    # Convert mask values from [0, 1] to [0, 255] and cast to uint8.
    mask_uint8 = (mask * 255).astype(np.uint8)
    # Stack the single channel 3 times to get a (H, W, 3) array.
    rgb = np.stack([mask_uint8, mask_uint8, mask_uint8], axis=-1)
    pil_img = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


async def _read_file(file: UploadFile) -> Image.Image:
    """
    Asynchronously read a file and convert it to a PIL image.

    Args:
        file (UploadFile): File to read.

    Returns:
        Image.Image: PIL image
    """
    image_bytes = await file.read()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
