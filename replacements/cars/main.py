from alpha_generation import generate_smooth_alpha_mask_from_binary_mask
from constants import DATA_IMAGES_DIRECTORY, DATA_MASKS_DIRECTORY, DATA_BACKGROUNDS_DIRECTORY
from foreground_estimation import get_foreground_estimation
from replacement import replace_background

if __name__ == "__main__":
    # Define the file paths
    image_path = DATA_IMAGES_DIRECTORY / "cf89c3220bc4_01.jpg"
    binary_mask_path = DATA_MASKS_DIRECTORY / "cf89c3220bc4_01_mask.png"
    background_image_path = DATA_BACKGROUNDS_DIRECTORY / "background_02.jpg"

    # Define image title
    image_title = image_path.stem

    # Generate alpha mask
    alpha_mask = generate_smooth_alpha_mask_from_binary_mask(binary_mask_path, image_title)

    # Foreground estimation
    foreground, background = get_foreground_estimation(image_path, alpha_mask, image_title)

    # Replace background
    replaced_image = replace_background(background_image_path, foreground, alpha_mask, image_title)

