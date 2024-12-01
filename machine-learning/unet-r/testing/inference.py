import torch

from PIL import Image
import albumentations as A
import numpy as np
from transformers import SegformerImageProcessor


def predict_image(
        image: Image.Image,
        model: torch.nn.Module,
        transform: A.Compose,
        image_processor: SegformerImageProcessor,
        device: torch.device,
):
    model.eval()

    # Convert PIL image to NumPy array and apply transforms
    image_np = np.array(image)
    transformed = transform(image=image_np)

    # Ensure the transformed output is a PyTorch tensor
    image_tensor = torch.from_numpy(transformed["image"]).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32)

    with torch.inference_mode():
        # Process images
        inputs = image_processor(images=list(image_tensor), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

        # Perform inference
        outputs = model(pixel_values=pixel_values)
        y_preds = torch.sigmoid(outputs)
        preds = (y_preds > 0.5).float()  # Binary mask

    return preds.squeeze(0).cpu().numpy()  # Remove batch and convert to NumPy
