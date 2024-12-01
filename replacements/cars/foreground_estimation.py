from pymatting import *
import numpy as np
import cv2
import os


def convert_binary_mask_to_alpha_mask(
        binary_mask: np.ndarray,
        blur_ksize: int = 7,
        blur_sigma: int = 2,
) -> np.ndarray:
    """
    Convert a binary mask to a smooth alpha mask.

    Parameters:
        binary_mask (numpy.ndarray): Binary mask with values 0 or 255.
        blur_ksize (int): Kernel size for Gaussian blur.
        blur_sigma (int): Standard deviation for Gaussian blur.

    Returns:
        numpy.ndarray: Smooth alpha mask with values between 0 and 1.
    """
    # Ensure binary_mask is binary (values 0 or 255)
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255

    # Apply Gaussian blur to create smooth transitions
    blurred_mask = cv2.GaussianBlur(binary_mask, ksize=(blur_ksize, blur_ksize), sigmaX=blur_sigma, sigmaY=blur_sigma)

    # Normalize to range [0, 1]
    alpha = blurred_mask / 255.0

    return alpha


if __name__ == "__main__":
    # Define the file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, r".\data\images\cf89c3220bc4_01.jpg")
    binary_path = os.path.join(current_dir, r".\data\masks\cf89c3220bc4_01_mask.png")
    background_path = os.path.join(current_dir, r".\data\backgrounds\background_01.jpg")

    # Check if files exist
    if not os.path.exists(image_path):
        raise ValueError(f"File not found: {image_path}")
    if not os.path.exists(binary_path):
        raise ValueError(f"File not found: {binary_path}")
    if not os.path.exists(background_path):
        raise ValueError(f"File not found: {background_path}")

    # Load the image
    image = load_image(image_path, "RGB")
    if image is None:
        raise ValueError(f"Could not load the image file: {image_path}")

    # Load the binary segmentation mask
    binary_mask = load_image(binary_path, "GRAY")
    if binary_mask is None:
        raise ValueError(f"Could not load the binary mask: {binary_path}")

    # Convert binary mask to alpha mask
    alpha_mask = convert_binary_mask_to_alpha_mask(binary_mask)

    # Load the new background image
    background_image = load_image(background_path, "RGB")
    if background_image is None:
        raise ValueError(f"Could not load the new sky image: {background_path}")

    # Estimate the foreground and background
    foreground = estimate_foreground_cf(image, alpha_mask)

    # Ensure output directory exists
    output_dir = os.path.join(current_dir, "./output/")
    os.makedirs(output_dir, exist_ok=True)

    # Ensure the new sky image matches the dimensions of the foreground
    background_image = cv2.resize(background_image, dsize=(foreground.shape[1], foreground.shape[0]))

    # Perform alpha compositing
    blended_image = alpha_mask[:, :, None] * foreground + (1 - alpha_mask[:, :, None]) * background_image

    # Save the blended image
    blended_image_path = os.path.join(output_dir, "cf89c3220bc4_01_blended.png")
    save_image(blended_image_path, blended_image)

    print(f"Image saved at: {blended_image_path}")
