import gradio as gr
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import os
from timeit import default_timer as timer
from PIL import Image
from typing import Tuple
from pathlib import Path

from model.build_model import build_model
from replacements.foreground_estimation import get_foreground_estimation
from replacements.replacements import sky_replacement

transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
        ToTensorV2(),
    ],
    additional_targets={'mask': 'mask'}
)

checkpoint_path = "unet_v1.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(checkpoint_path, device=str(device), mode="eval")


def preprocess(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess the input image.

    Args:
        image (Image): Input image.

    Returns:
        torch.Tensor: Preprocessed image.
    """
    # Apply transforms
    transformed = transforms(image=image)
    image_tensor = transformed["image"].unsqueeze(0).to(device)  # Add batch dimension

    return image_tensor


def postprocess(predicted_mask: torch.Tensor, width: int, height: int) -> np.ndarray:
    """
    Postprocess the predicted mask.

    Args:
        predicted_mask (torch.Tensor): Predicted mask.
        width (int): Width of the original image.
        height (int): Height of the original image.

    Returns:
        np.ndarray: Post-processed mask.
    """
    # Remove the batch dimension
    predicted_mask = predicted_mask.squeeze(0).cpu().numpy()

    # Normalize the alpha mask to [0, 1]
    if predicted_mask.shape[2] == 4:
        predicted_mask = predicted_mask[..., 3].astype(np.float64) / 255.0
    else:
        predicted_mask = np.squeeze(predicted_mask, axis=0)

    # Upscale the predicted mask to the original image size
    predicted_mask = np.array(
        Image.fromarray((predicted_mask * 255).astype(np.uint8)).resize((width, height))
    ).astype(np.float32) / 255.0

    return predicted_mask


def predict(image_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform binary segmentation on the input image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: Predicted mask, sky replacement and prediction time.
    """
    # Start the timer
    start_time = timer()

    # Load the image
    image = np.array(Image.open(image_path).convert("RGB"))

    # Preprocess the image
    image_tensor = preprocess(image)

    with torch.inference_mode():
        outputs = model(image_tensor.to(device))
        outputs = torch.clamp(outputs, 0, 1)  # Clamp to [0, 1]

    # Postprocess the output
    predicted_alpha_matte = postprocess(outputs, width=image.shape[1], height=image.shape[0])

    # Perform foreground estimation
    foreground = get_foreground_estimation(image_path, predicted_alpha_matte)

    # Perform sky replacement
    current_directory = Path(__file__).parent
    new_sky_path = current_directory / "assets/skies/new_sky.webp"
    replaced_sky = sky_replacement(new_sky_path, foreground, predicted_alpha_matte)

    # Calculate the prediction time
    prediction_time = round(timer() - start_time, 5)

    return predicted_alpha_matte, replaced_sky, prediction_time


title = "Demo: Unet Alpha Matting"
description = "This demo performs alpha matting and sky replacements for houses using a U-Net model. Upload an image to generate the alpha matte."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples") if
                example.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Input Image"),
    outputs=[
        gr.Image(type="numpy", label="Predicted Mask"),
        gr.Image(type="numpy", label="Sky Replacement"),
        gr.Number(label="Prediction Time (s)"),
    ],
    title=title,
    description=description,
    examples=example_list,
)

# Launch the interface
demo.launch()
