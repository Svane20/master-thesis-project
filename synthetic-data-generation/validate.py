from PIL import Image


def has_alpha_channel(image_path: str) -> bool:
    """
    Returns True if the image has an alpha channel (e.g., RGBA or LA mode).
    """
    with Image.open(image_path) as img:
        print(f"Image mode: {img.mode}")

        # Pillow image modes that include alpha:
        # - RGBA (Red, Green, Blue, Alpha)
        # - LA   (Luminance, Alpha)
        # - "P" can also sometimes include a transparency, but let's keep it simple here.
        return img.mode in ("RGBA", "LA")


def is_strict_binary_alpha_mask(image_path: str) -> bool:
    """
    Returns True if the image has an alpha channel AND all alpha values are either 0 or 255.
    """
    with Image.open(image_path) as img:
        # Ensure we're working in RGBA mode
        img_rgba = img.convert("RGBA")

        # Extract alpha channel
        alpha_channel = img_rgba.split()[-1]

        # Check each pixel to see if it's 0 or 255
        for alpha_value in alpha_channel.getdata():
            if alpha_value not in (0, 255):
                return False
        return True


def is_black_white_mask(image_path: str) -> bool:
    """
    Returns True if the image (regardless of alpha) is purely black or white
    in grayscale form (i.e., each pixel is 0 or 255).
    """
    with Image.open(image_path) as img:
        # Convert to grayscale
        grayscale = img.convert("L")

        for pixel in grayscale.getdata():
            if pixel not in (0, 255):
                return False
        return True


if __name__ == "__main__":
    image_path = "D:/OneDrive/Master Thesis/datasets/processed/synthetic-data/test/masks/4996.png"

    if has_alpha_channel(image_path):
        print("Image has an alpha channel!")
    else:
        print("No alpha channel detected.")

    if is_strict_binary_alpha_mask(image_path):
        print("This image is a strict black-and-white (binary) alpha mask.")
    else:
        print("Alpha channel contains partial transparency or no alpha channel.")


    if is_black_white_mask(image_path):
        print("This image is a black-and-white mask.")
    else:
        print("This image is not a black-and-white mask.")
