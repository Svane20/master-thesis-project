from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from constants.directories import OUTPUT_DIRECTORY


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
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    # Display input image
    ax[0].imshow(image)
    ax[0].set_title('Input')
    ax[0].axis('off')

    # Display prediction overlay
    ax[1].imshow(predicted_mask, cmap='gray')
    ax[1].set_title(f'Predicted Mask')
    ax[1].axis('off')

    prediction_path = directory / "prediction.png"

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(prediction_path)
    plt.show()
    plt.close(fig)

    print(f"Prediction saved to {prediction_path}")
