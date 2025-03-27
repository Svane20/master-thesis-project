import gradio as gr
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import numpy as np
import os
from timeit import default_timer as timer
from PIL import Image
from typing import Tuple

from model.build_model import build_model
from replacements.foreground_estimation import get_foreground_estimation
from replacements.replacements import sky_replacement

transforms = Compose(
    [
        Resize(size=(512, 512)),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ],
)

checkpoint_path = "unet_resnet_34_v2.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(checkpoint_path, device=str(device), mode="eval")


def _inference(image: torch.Tensor) -> np.ndarray:
    """
    Perform inference on the input image.

    Args:
        image (torch.Tensor): Input image.

    Returns:
        torch.Tensor: Predicted mask.
    """
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    enabled = torch.cuda.is_available()

    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=enabled, dtype=dtype):
            outputs = model(image)

    # Convert the output to a numpy array and remove the batch dimension
    predicted_alpha = outputs.detach().cpu().numpy()
    predicted_alpha = np.squeeze(predicted_alpha)

    return predicted_alpha


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
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image
    image_tensor = transforms(image).unsqueeze(0).to(device)

    # Perform inference
    predicted_alpha = _inference(image_tensor)

    # Downscale the image to fit the predicted alpha
    h, w = predicted_alpha.shape
    downscaled_image = image.resize(size=(w, h), resample=Image.Resampling.LANCZOS)

    # Perform foreground estimation
    foreground = get_foreground_estimation(downscaled_image, predicted_alpha)

    # Perform sky replacement
    replaced_sky = sky_replacement(foreground, predicted_alpha)

    # Calculate the prediction time
    prediction_time = round(timer() - start_time, 5)

    return predicted_alpha, replaced_sky, prediction_time


title = "Demo: Sky Replacement with Alpha Matting"
description = """
This demo performs alpha matting and sky replacements for houses using a U-Net model with a ResNet-34 backbone. 
Upload an image to perform sky replacement.
"""

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
