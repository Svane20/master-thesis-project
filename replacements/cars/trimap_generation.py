import numpy as np
import cv2
import os


def generate_trimap(binary_mask: np.ndarray, kernel_size: int = 5, iterations: int = 5) -> np.ndarray:
    """
    Generate a trimap from a binary segmentation mask.

    Args:
        binary_mask (numpy.ndarray): Binary mask (0s and 1s or 0s and 255s).
        kernel_size (int): Size of the kernel used for dilation and erosion.
        iterations (int): Number of iterations for morphological operations.

    Returns:
        numpy.ndarray: Trimap with values 0 (background), 128 (unknown), and 255 (foreground).
    """
    # Ensure binary_mask is binary (0 or 1)
    binary_mask = (binary_mask > 0).astype(np.uint8)

    # Define a kernel
    kernel = np.ones(shape=(kernel_size, kernel_size), dtype=np.uint8)

    # Perform dilation and erosion
    dilated_mask = cv2.dilate(src=binary_mask, kernel=kernel, iterations=iterations)
    eroded_mask = cv2.erode(src=binary_mask, kernel=kernel, iterations=iterations)

    # Initialize the trimap
    trimap = np.zeros_like(a=binary_mask, dtype=np.uint8)

    # Assign values to the trimap
    trimap[dilated_mask == 1] = 255
    trimap[eroded_mask == 0] = 0
    trimap[(dilated_mask == 1) & (eroded_mask == 0)] = 128

    return trimap


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

    # Generate the trimap and save the result
    trimap = generate_trimap(mask)
    cv2.imwrite(filename=f"./output/{image_title}_trimap.png", img=trimap)
