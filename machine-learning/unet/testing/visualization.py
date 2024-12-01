from typing import Tuple

import torch

from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from constants.directories import OUTPUT_DIRECTORY
from metrics.DICE import calculate_DICE, calculate_DICE_edge


def save_predictions(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        num_batches: int = None,
        directory: Path = OUTPUT_DIRECTORY,
) -> None:
    """
    Save the model predictions as images.

    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (torch.utils.data.DataLoader): Data loader
        device (torch.device): Device to use for evaluation
        num_batches (int, optional): Number of batches to process. If None, process all batches. Default is None.
        directory (Path): Directory to save the images to. Default is "output".
    """
    # Create the directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)

    model.eval()

    data_iter = iter(data_loader)
    batches_to_process = []
    for _ in range(num_batches):
        try:
            batch = next(data_iter)
            batches_to_process.append(batch)
        except StopIteration:
            break
    print(f"Processing the first {len(batches_to_process)} batch(es).")

    with torch.inference_mode():
        for batch_idx, (X, y) in tqdm(enumerate(batches_to_process), desc="Saving predictions"):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            y_logits = model(X)
            y_preds = torch.sigmoid(y_logits)
            preds = (y_preds > 0.5).float()

            # For each sample in the batch
            for idx in range(X.size(0)):
                input_img = X[idx].cpu().numpy().transpose(1, 2, 0)
                target_mask = y[idx].cpu().numpy().squeeze()
                pred_mask = preds[idx].cpu().numpy().squeeze()

                # Un-normalize the image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                input_img = std * input_img + mean
                input_img = np.clip(input_img, 0, 1)

                # Compute metrics per image
                dice_score = calculate_DICE(torch.tensor(pred_mask), torch.tensor(target_mask))
                dice_edge_score = calculate_DICE_edge(torch.tensor(pred_mask), torch.tensor(target_mask))

                # Create figure
                fig, ax = plt.subplots(1, 3, figsize=(18, 6))

                # Display input image
                ax[0].imshow(input_img)
                ax[0].set_title('Input Image')
                ax[0].axis('off')

                # Display ground truth mask
                ax[1].imshow(target_mask, cmap='gray')
                ax[1].set_title('Ground Truth Mask')
                ax[1].axis('off')

                # Display prediction overlay
                ax[2].imshow(input_img)
                ax[2].imshow(pred_mask, cmap='jet', alpha=0.5)
                ax[2].set_title(f'Predicted Mask Overlay\nDice: {dice_score:.4f}, Dice Edge: {dice_edge_score:.4f}')
                ax[2].axis('off')

                sample_idx = batch_idx * data_loader.batch_size + idx
                sample_path = directory / f"sample_{sample_idx}.png"

                plt.tight_layout()
                plt.subplots_adjust(top=0.85)
                plt.savefig(sample_path)
                plt.close(fig)


def save_prediction(
        image: Image.Image,
        predicted_mask: np.ndarray,
        directory: Path = OUTPUT_DIRECTORY,
) -> None:
    """
    Save the input image and predicted mask.

    Args:
        image (Image.Image): Input image
        predicted_mask (np.ndarray): Predicted mask
        directory (Path): Directory to save the image to. Default is "output".
    """
    # Resize the image to 224x224
    image = image.resize((224, 224), Image.Resampling.LANCZOS)

    # Create the directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)

    # Ensure the predicted mask has the correct shape
    if predicted_mask.ndim == 3 and predicted_mask.shape[0] == 1:
        predicted_mask = predicted_mask.squeeze(0)

    # Create figure
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Display input image
    ax[0].imshow(image)
    ax[0].set_title('Input')
    ax[0].axis('off')

    # Display prediction overlay
    ax[1].imshow(predicted_mask, cmap='gray')
    ax[1].set_title(f'Predicted Mask')
    ax[1].axis('off')

    # Display prediction overlay
    ax[2].imshow(image)
    ax[2].imshow(predicted_mask, cmap='jet', alpha=0.5)
    ax[2].set_title(f'Predicted Mask Overlay')
    ax[2].axis('off')

    prediction_path = directory / "prediction.png"

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(prediction_path)
    plt.show()
    plt.close(fig)

    print(f"Prediction saved to {prediction_path}")


def remove_background(
        image: Image.Image,
        predicted_mask: np.ndarray,
        directory: Path = OUTPUT_DIRECTORY,
        output_size: Tuple[int, int] = (224, 224),
) -> None:
    """
    Save the input image and predicted mask with background removed.

    Args:
        image (Image.Image): Input image
        predicted_mask (np.ndarray): Predicted mask
        directory (Path): Directory to save the image to. Default is "output".
        output_size (tuple[int, int]): Desired output size for the saved image (width, height). Default is (224, 224).
    """
    # Create the directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)

    assert len(output_size) == 2, "Output size must be a tuple of two integers (width, height)."
    assert output_size[0] >= 224 and output_size[1] >= 224, "Output size must be at least 224x224."

    # Resize the input image
    image = image.resize(output_size, Image.Resampling.LANCZOS)

    # Ensure predicted_mask is in a format that can be processed by PIL
    if predicted_mask.ndim == 3 and predicted_mask.shape[0] == 1:
        predicted_mask = predicted_mask.squeeze(0)

    # Convert predicted_mask to uint8 format
    predicted_mask = (predicted_mask * 255).astype(np.uint8)

    # Create a PIL image for the mask and resize it
    mask_image = Image.fromarray(predicted_mask).resize(output_size, Image.Resampling.NEAREST)

    # Convert the resized mask back to a NumPy array and ensure it is binary
    binary_mask = (np.array(mask_image) > 127).astype(np.uint8)

    # Add an alpha channel based on the binary mask
    image_array = np.array(image)
    alpha_channel = (binary_mask * 255).astype(np.uint8)
    image_with_alpha = np.dstack((image_array, alpha_channel))

    # Save the background removed image with transparency
    background_removed_path = directory / "background_removed.png"
    Image.fromarray(image_with_alpha, mode="RGBA").save(background_removed_path)

    # Create figure
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Display input image
    ax[0].imshow(image)
    ax[0].set_title('Input')
    ax[0].axis('off')

    # Display prediction overlay
    ax[1].imshow(predicted_mask, cmap='gray')
    ax[1].set_title(f'Predicted Mask')
    ax[1].axis('off')

    # Display background removed
    ax[2].imshow(Image.fromarray(image_with_alpha, mode="RGBA"))
    ax[2].set_title('Background Removed')
    ax[2].axis('off')

    prediction_path = directory / "background.png"

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(prediction_path)
    plt.show()
    plt.close(fig)

    print(f"Background saved to {prediction_path}")
