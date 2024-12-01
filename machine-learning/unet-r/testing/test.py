import torch

from PIL import Image
import os

from transformers import AutoImageProcessor

from constants.directories import DATA_TEST_DIRECTORY
from constants.outputs import TRAINED_MODEL_CHECKPOINT_NAME
from constants.seg_former import MODEL_NAME
from dataset.transforms import get_test_transforms
from models.unet_r import UNETR
from testing.inference import predict_image
from testing.visualization import save_prediction
from utils.checkpoints import load_model_checkpoint
from utils.device import get_device, get_torch_compile_backend

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def main():
    # Get test transforms and data loader
    transform = get_test_transforms()

    # Setup device
    device = get_device()

    # Load trained model
    model = UNETR(out_channels=1)
    model, _ = load_model_checkpoint(model=model, model_name=TRAINED_MODEL_CHECKPOINT_NAME, device=device)
    model = torch.compile(model, backend=get_torch_compile_backend())

    # Instantiate the image processor
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    # Single image evaluation
    image_path = DATA_TEST_DIRECTORY / "images" / "cf89c3220bc4_03.jpg"
    image = Image.open(image_path).convert("RGB")

    predicted_mask = predict_image(image, model, transform, image_processor, device)
    save_prediction(image, predicted_mask)

if __name__ == "__main__":
    main()
