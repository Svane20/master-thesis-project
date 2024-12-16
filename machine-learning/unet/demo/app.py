import gradio as gr
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import os
from timeit import default_timer as timer
from PIL.Image import Image
from typing import Tuple

from image_encoder import ImageEncoder
from mask_decoder import MaskDecoder
from unet import UNet

# Disable Albumentations update checks
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


# Define model and transforms
model = UNet(
    image_encoder=ImageEncoder(
        pretrained=True,
        freeze_pretrained=True,
    ),
    mask_decoder=MaskDecoder(
        out_channels=1,
        dropout=0.5
    )
)

transforms = A.Compose([
    A.Resize(224, 224),

    # Normalize and convert to tensor
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])

# Load model checkpoint
model.load_state_dict(torch.load("unet_production.pt", map_location="cpu", weights_only=True))


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
    mask = mask.squeeze(0).squeeze(0).cpu().numpy()

    # Normalize and convert to uint8 (values 0-255 for visualization)
    mask = (mask > 0.5).astype(np.uint8) * 255

    return mask


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

    # Put model in evaluation mode
    model.eval()

    with torch.inference_mode():
        # Get predictions
        logits = model(image)
        prediction = torch.sigmoid(logits)  # Apply sigmoid for binary segmentation

    # Calculate the prediction time
    prediction_time = round(timer() - start_time, 5)

    # Postprocess the output
    prediction = postprocess(prediction)

    return prediction, prediction_time


title = "Demo: Binary Segmentation"
description = "This demo performs binary segmentation on input images using a U-Net model."

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
