import torch
import torch.nn.functional as F

from PIL import Image
from typing import Tuple
import numpy as np

from constants.directories import CHECKPOINTS_DIRECTORY, PREDICTIONS_DIRECTORY, DATA_TEST_DIRECTORY
from constants.outputs import IMAGE_SIZE
from dataset.transforms import get_test_transforms
from metrics.DICE import calculate_DICE, calculate_edge_DICE
from metrics.IoU import calculate_IoU, calculate_edge_IoU
from model.unet import UNetV0
from utils import get_device


def load_trained_model(checkpoint_path: str, device: torch.device):
    """
    Load the trained U-Net model.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded model.
    """
    model = UNetV0(
        in_channels=3,
        out_channels=1,
        dropout=0.5
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)['model_state_dict'])
    model.eval()

    return model


def generate_prediction_mask(
        model: torch.nn.Module,
        pil_image: Image.Image,
        device: torch.device,
        image_size: Tuple[int, int] = IMAGE_SIZE,
        threshold: float = 0.5
) -> Image.Image:
    """
    Generate a prediction mask for a given PIL image using the trained model.

    Args:
        model (torch.nn.Module): Trained segmentation model.
        pil_image (Image.Image): Input PIL image.
        device (torch.device): Device to perform computation on.
        image_size (Tuple[int, int], optional): Expected input size for the model. Defaults to (224, 224).
        threshold (float, optional): Threshold for converting probabilities to binary mask. Defaults to 0.5.

    Returns:
        Image.Image: Predicted mask as a PIL image.
    """
    # Convert PIL image to NumPy array
    image_np = np.array(pil_image)

    # Get transforms
    preprocess = get_test_transforms(image_size)

    # Apply transformations
    augmented = preprocess(image=image_np)
    transformed_image = augmented['image']  # This is a tensor of shape [C, H, W]

    # Add batch dimension
    input_tensor = transformed_image.unsqueeze(0).to(device)  # Shape: [1, C, H, W]

    with torch.inference_mode():
        output = model(input_tensor)  # Output shape: [1, 1, H, W]

        # Apply sigmoid activation to get probabilities
        probs = torch.sigmoid(output)

        # Resize output to original image size if necessary
        original_size = pil_image.size  # PIL size is (width, height)
        probs = F.interpolate(probs, size=(original_size[1], original_size[0]), mode='bilinear', align_corners=False)

        # Convert probabilities to binary mask
        mask = (probs > threshold).float()

    # Remove batch and channel dimensions
    mask = mask.squeeze().cpu().numpy()  # Shape: [H, W]

    # Convert mask to PIL image
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))

    return mask_image


def main():
    # Path to the saved model checkpoint
    checkpoint_path = CHECKPOINTS_DIRECTORY / "UNetV0_EdgeDetection_Tuned_best_checkpoint.pth"

    # Device configuration
    device = get_device()

    # Load the trained model
    model = load_trained_model(checkpoint_path, device)

    # Load test image
    image_path = DATA_TEST_DIRECTORY / "images" / "cf89c3220bc4_03.jpg"
    pil_image = Image.open(image_path).convert('RGB')

    # Load ground truth mask
    mask_name = image_path.name.replace('.jpg', '_mask.gif')
    mask_path = DATA_TEST_DIRECTORY / "masks" / mask_name
    ground_truth_mask = Image.open(mask_path).convert('L')

    predicted_mask = generate_prediction_mask(model, pil_image, device, threshold=0.5)

    # Convert masks to numpy arrays and normalize
    predicted_mask_np = np.array(predicted_mask, dtype=np.float32) / 255.0
    ground_truth_mask_np = np.array(ground_truth_mask, dtype=np.float32) / 255.0

    # Ensure masks are the same size
    if predicted_mask_np.shape != ground_truth_mask_np.shape:
        # Resize predicted mask to match ground truth mask
        predicted_mask_np = np.array(
            predicted_mask.resize(ground_truth_mask.size, resample=Image.NEAREST),
            dtype=np.float32
        ) / 255.0

    # Convert masks to tensors and move to device
    pred_mask_tensor = torch.from_numpy(predicted_mask_np).float().to(device)
    gt_mask_tensor = torch.from_numpy(ground_truth_mask_np).float().to(device)

    # Compute metrics
    dice_score = calculate_DICE(pred_mask_tensor, gt_mask_tensor)
    dice_edge_score = calculate_edge_DICE(pred_mask_tensor, gt_mask_tensor)
    iou_score = calculate_IoU(pred_mask_tensor, gt_mask_tensor)
    iou_edge_score = calculate_edge_IoU(pred_mask_tensor, gt_mask_tensor)

    print(f"Dice Score: {dice_score:.4f}")
    print(f"Dice Edge Score: {dice_edge_score:.4f}")
    print(f"IoU Score: {iou_score:.4f}")
    print(f"IoU Edge Score: {iou_edge_score:.4f}")

    # Create the prediction directory if it does not exist
    PREDICTIONS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Save or display the predicted mask
    predicted_mask.save(PREDICTIONS_DIRECTORY / "prediction_mask.png")
    predicted_mask.show()

    # Overlay the predicted mask on the input image
    overlay = Image.blend(pil_image, predicted_mask.convert('RGB'), alpha=0.5)
    overlay.save(PREDICTIONS_DIRECTORY / "prediction_overlay.png")
    overlay.show()


if __name__ == "__main__":
    main()
