import gradio as gr
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import onnxruntime as ort

import pymatting
import numpy as np
import os
from PIL import Image
from typing import Tuple
import random
from pathlib import Path


def _load_model(checkpoint):
    """
    Load the ONNX model for inference.

    Args:
        checkpoint (str): Path to the ONNX model file.

    Returns:
        session (onnxruntime.InferenceSession): The ONNX runtime session.
        input_name (str): The name of the input tensor.
        output_name (str): The name of the output tensor.
    """
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = min(1, os.cpu_count() - 1)
    providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    session = ort.InferenceSession(checkpoint, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    return session, input_name, output_name


transforms = Compose(
    [
        Resize(size=(512, 512)),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ],
)

share_repo = False
checkpoint_path = "swin_small_patch4_window7_224_512_v1_latest.onnx"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
session, input_name, output_name = _load_model(checkpoint_path)


def _get_foreground_estimation(image, alpha):
    """
    Estimate the foreground using the image and the predicted alpha mask.

    Args:
        image (np.ndarray): The input image.
        alpha (np.ndarray): The predicted alpha mask.

    Returns:
        np.ndarray: The estimated foreground.
    """
    # Normalize the image to [0, 1] range
    normalized_image = np.array(image) / 255.0

    # Invert the alpha mask since the pymatting library expects the sky to be the background
    inverted_alpha = 1 - alpha

    return pymatting.estimate_foreground_ml(image=normalized_image, alpha=inverted_alpha)


def _sky_replacement(foreground, alpha_mask):
    """
    Perform sky replacement using the estimated foreground and predicted alpha mask.

    Args:
        foreground (np.ndarray): The estimated foreground.
        alpha_mask (np.ndarray): The predicted alpha mask.

    Returns:
        np.ndarray: The sky-replaced image.
    """
    new_sky_path = Path(__file__).parent / "assets/skies/francesco-ungaro-i75WTJn-RBY-unsplash.jpg"
    new_sky_img = Image.open(new_sky_path).convert("RGB")

    # Get the target size from the foreground image
    h, w = foreground.shape[:2]

    # Check the size of the sky image
    sky_width, sky_height = new_sky_img.size

    # If the sky image is smaller than the target size
    if sky_width < w or sky_height < h:
        scale = max(w / sky_width, h / sky_height)
        new_size = (int(sky_width * scale), int(sky_height * scale))
        new_sky_img = new_sky_img.resize(new_size, resample=Image.Resampling.LANCZOS)
        sky_width, sky_height = new_sky_img.size

    # Determine the maximum possible top-left coordinates for the crop
    max_left = sky_width - w
    max_top = sky_height - h

    # Choose random offsets for left and top within the valid range
    left = random.randint(a=0, b=max_left) if max_left > 0 else 0
    top = random.randint(a=0, b=max_top) if max_top > 0 else 0

    # Crop the sky image to the target size using the random offsets
    new_sky_img = new_sky_img.crop((left, top, left + w, top + h))

    new_sky = np.asarray(new_sky_img).astype(np.float32) / 255.0
    if foreground.dtype != np.float32:
        foreground = foreground.astype(np.float32) / 255.0
    if foreground.shape[2] == 4:
        foreground = foreground[:, :, :3]

    # Ensure that the alpha mask values are within the range [0, 1]
    alpha_mask = np.clip(alpha_mask, a_min=0, a_max=1)

    # Blend the foreground with the new sky using the alpha mask
    return (1 - alpha_mask[:, :, None]) * foreground + alpha_mask[:, :, None] * new_sky


def _inference(image):
    """
    Perform inference on the input image using the ONNX model.

    Args:
        image (Image): The input image.

    Returns:
        np.ndarray: The predicted alpha mask.
    """
    output = session.run(output_names=[output_name], input_feed={input_name: image.cpu().numpy()})[0]

    # Ensure the output is in valid range [0, 1]
    output = np.clip(output, a_min=0, a_max=1)

    return np.squeeze(output, axis=0).squeeze()


def predict(image):
    """
    Perform sky replacement on the input image.

    Args:
        image (Image): The input image.

    Returns:
        Tuple[Image, Image]: The predicted alpha mask and the sky-replaced image.
    """
    image_tensor = transforms(image).unsqueeze(0).to(device)
    predicted_alpha = _inference(image_tensor)

    # Downscale the input image to match predicted_alpha
    h, w = predicted_alpha.shape
    downscaled_image = image.resize((w, h), Image.Resampling.LANCZOS)

    # Estimate foreground and run sky_replacement
    foreground = _get_foreground_estimation(downscaled_image, predicted_alpha)
    replaced_sky = _sky_replacement(foreground, predicted_alpha)

    # Resize the predicted alpha and replaced sky to original dimensions
    predicted_alpha_pil = Image.fromarray((predicted_alpha * 255).astype(np.uint8), mode='L')
    predicted_alpha_pil = predicted_alpha_pil.resize((h, w), Image.Resampling.LANCZOS)
    replaced_sky_pil = Image.fromarray((replaced_sky * 255).astype(np.uint8))  # mode='RGB' typically
    replaced_sky_pil = replaced_sky_pil.resize((h, w), Image.Resampling.LANCZOS)

    return predicted_alpha_pil, replaced_sky_pil


real_example_list = [
    ["examples/real/1901.jpg", "Real", "Good"],
    ["examples/real/2022.jpg", "Real", "Good"],
    ["examples/real/2041.jpg", "Real", "Good"],
    ["examples/real/2196.jpg", "Real", "Good"],
    ["examples/real/2188.jpg", "Real", "Good"],
    ["examples/real/0001.jpg", "Real", "Acceptable, missing minor detail around the lamppost"],
    ["examples/real/0054.jpg", "Real", "Acceptable, missing sky details between the houses"],
    ["examples/real/2043.jpg", "Real", "Acceptable, missing minor detail in the window in the background"],
    ["examples/real/0211.jpg", "Real", "Okay, misclassified a cloud in the left corner as the sky"],
    ["examples/real/0894.jpg", "Real", "Okay, missing details in the trees"],
    ["examples/real/2184.jpg", "Real", "Okay, lacks tree details in the background"],
    ["examples/real/2026.jpg", "Real", "Okay, lacks tree details in the left background"],
    ["examples/real/1975.jpg", "Real", "Okay, lacks tree branch details"],
    ["examples/real/0069.jpg", "Real", "Bad, didn't replace the sky between the houses"],
    ["examples/real/2079.jpg", "Real", "Bad, couldn't get the complete details of the tree"],
    ["examples/real/2038.jpg", "Real", "Bad, lacks overall details in both trees and tree branches"],
]

synthetic_example_list = [
    ["examples/synthetic/0055.jpg", "Synthetic", "Good"],
    ["examples/synthetic/0059.jpg", "Synthetic", "Good"],
    ["examples/synthetic/0086.jpg", "Synthetic", "Good"],
    ["examples/synthetic/10406.jpg", "Synthetic", "Good"],
    ["examples/synthetic/10515.jpg", "Synthetic", "Good"],
    ["examples/synthetic/10416.jpg", "Synthetic", "Acceptable, missing minor detail in the tree leaves"],
    ["examples/synthetic/0150.jpg", "Synthetic", "Acceptable, missing minor detail in the tree"],
    ["examples/synthetic/0097.jpg", "Synthetic", "Okay, missing minor detail in the trees"],
    ["examples/synthetic/0124.jpg", "Synthetic", "Okay, missing minor detail in the trees"],
    ["examples/synthetic/0127.jpg", "Synthetic", "Bad, missing many details in the trees"],
    ["examples/synthetic/10467.jpg", "Synthetic", "Bad, misclassified the windows as sky"],
]

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown(
        """
        # Demo: Sky Replacement with Alpha Matting
        This demo performs alpha matting and sky replacements for houses using a U-Net architecture with a Swin backbone. \t
        This model is trained solely on synthetic data generated using Blender. \n
        Upload an image to perform sky replacement.
        """
    )

    data_type = gr.Radio(choices=["Real", "Synthetic"], value="Real", label="Select Data Type for Examples")

    with gr.Row():
        # Left Column: Input Image and Run/Clear Buttons
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image")
            with gr.Row():
                clear_button = gr.Button("Clear")
                run_button = gr.Button("Submit", variant="primary")

        # Right Column: Output Images
        with gr.Column(scale=1):
            output_mask = gr.Image(type="pil", label="Predicted Mask")
            output_sky = gr.Image(type="pil", label="Sky Replacement")

    metadata_display = gr.Markdown(None)

    with gr.Column(visible=True) as real_examples_container:
        real_examples_component = gr.Examples(
            examples=real_example_list,
            inputs=[input_image,
                    gr.Textbox(label="Data Type", value="", interactive=False, visible=False),
                    gr.Textbox(label="Result", value="", interactive=False, visible=False)],
            outputs=[input_image, metadata_display],
            fn=lambda example, dtype, desc: (example, f"**Type:** {dtype}\n\n**Result:** {desc}"),
            cache_examples=False,
            label="Real Data Examples"
        )

    with gr.Column(visible=False) as synthetic_examples_container:
        synthetic_examples_component = gr.Examples(
            examples=synthetic_example_list,
            inputs=[input_image,
                    gr.Textbox(label="Data Type", value="", interactive=False, visible=False),
                    gr.Textbox(label="Result", value="", interactive=False, visible=False)],
            outputs=[input_image, metadata_display],
            fn=lambda example, dtype, desc: (example, f"**Type:** {dtype}\n\n**Result:** {desc}"),
            cache_examples=False,
            label="Synthetic Data Examples"
        )


    # Callback to toggle the container visibility based on selection.
    def switch_examples(selected):
        if selected == "Real":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)


    data_type.change(
        fn=switch_examples,
        inputs=data_type,
        outputs=[real_examples_container, synthetic_examples_container]
    )


    def clear_all():
        return gr.update(value=None), gr.update(value=None), gr.update(value=None)


    clear_button.click(fn=clear_all, inputs=[], outputs=[input_image, output_mask, output_sky])
    run_button.click(fn=predict, inputs=input_image, outputs=[output_mask, output_sky])

# Launch the interface
demo.launch(share=share_repo)
