import numpy as np
import cv2
import os


def generate_alpha_mask(binary_mask: np.ndarray) -> np.ndarray:
    """
    Generates an alpha mask from a binary mask.

    Parameters:
        binary_mask (numpy.ndarray): Binary mask (values: 0 or 255).

    Returns:
        numpy.ndarray: Alpha mask with values between 0 (transparent) and 255 (opaque).
    """
    # Ensure binary_mask is binary (0 or 255)
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255

    # Optionally smooth the edges using a GaussianBlur
    alpha_mask = cv2.GaussianBlur(binary_mask, (7, 7), sigmaX=2, sigmaY=2)

    return alpha_mask


if __name__ == "__main__":
    # Define the file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, r".\data\masks\cf89c3220bc4_01_mask.png")
    image_title = os.path.basename(file_path).split("_mask")[0]
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    # Load the mask
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"cv2.imread could not read the file: {file_path}")

    # Generate the alpha mask and save the result
    alpha_mask = generate_alpha_mask(mask)
    cv2.imwrite(filename=f"./output/{image_title}_alpha.png", img=alpha_mask)
