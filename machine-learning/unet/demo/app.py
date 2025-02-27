import gradio as gr
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import os
from timeit import default_timer as timer
from PIL.Image import Image
from typing import Tuple

from model.build_model import build_model

transforms = A.Compose([
    A.Resize(224, 224),

    # Normalize and convert to tensor
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model("unet_v1.pt", device=str(device), mode="eval")

def preprocess(image: Image) -> torch.Tensor:
    """
    Preprocess the input image.

    Args:
        image (Image): Input image.

    Returns:
        torch.Tensor: Preprocessed image.
    """
    # Convert PIL image to NumPy array
    image = np.array(image)

    # Apply Albumentations transformations
    transformed = transforms(image=image)

    # Add batch dimension (C, H, W -> 1, C, H, W)
    tensor = transformed["image"].unsqueeze(0)

    return tensor


def postprocess(mask: torch.Tensor) -> np.ndarray:
    """
    Postprocess the predicted mask.

    Args:
        mask (torch.Tensor): Predicted mask.

    Returns:
        np.ndarray: Post-processed mask.
    """
    # Remove batch and channel dimensions, and convert to NumPy array
    pred_alpha = mask.squeeze(0).cpu().numpy()

    return pred_alpha


def predict(image: Image) -> Tuple[np.ndarray, float]:
    """
    Perform binary segmentation on the input image.

    Args:
        image (Image): Input image.

    Returns:
        Tuple[np.ndarray, float]: Predicted mask and prediction time.
    """
    # Start the timer
    start_time = timer()

    # Preprocess the image
    image = preprocess(image)

    with torch.inference_mode():
        outputs = model(image)
        outputs = torch.clamp(outputs, 0, 1)  # Clamp to [0, 1]

    # Calculate the prediction time
    prediction_time = round(timer() - start_time, 5)

    # Postprocess the output
    prediction = postprocess(outputs)

    return prediction, prediction_time


title = "Demo: Unet Alpha Matting"
description = "This demo performs alpha matting using a U-Net model. Upload an image to generate the alpha matte."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples") if example.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Input Image"),
    outputs=[
        gr.Image(type="numpy", label="Predicted Mask"),  # Mask output
        gr.Number(label="Prediction Time (s)"),         # Prediction time
    ],
    title=title,
    description=description,
    examples=example_list,
)

# Launch the interface
demo.launch()
