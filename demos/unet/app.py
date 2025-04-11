import gradio as gr
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np
import os
from timeit import default_timer as timer
from PIL import Image
import onnxruntime as ort

from replacements.foreground_estimation import get_foreground_estimation
from replacements.replacements import sky_replacement


def _load_model(checkpoint):
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
checkpoint_path = "resnet_50_512_v1.onnx"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
session, input_name, output_name = _load_model(checkpoint_path)


def inference(image):
    output = session.run([output_name], {input_name: image.cpu().numpy()})[0]

    # Ensure the output is in valid range [0, 1]
    output = np.clip(output, 0, 1)

    return np.squeeze(output, axis=0).squeeze()


def predict(image):
    image_tensor = transforms(image).unsqueeze(0).to(device)

    # Perform inference
    predicted_alpha = inference(image_tensor)

    # Perform sky replacement
    h, w = predicted_alpha.shape
    downscaled_image = image.resize(size=(w, h), resample=Image.Resampling.LANCZOS)
    foreground = get_foreground_estimation(downscaled_image, predicted_alpha)
    replaced_sky = sky_replacement(foreground, predicted_alpha)

    return predicted_alpha, replaced_sky


title = "Demo: Sky Replacement with Alpha Matting"
description = """
This demo performs alpha matting and sky replacements for houses using a U-Net architecture with a ResNet-50 backbone. \t
This model is trained solely on synthetic data generated using Blender. \n
Upload an image to perform sky replacement.
"""

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown(
        """
        # Sky Replacement with Alpha Matting
        This demo performs alpha matting and sky replacements for houses using a U-Net architecture with a ResNet-50 backbone. \t
        This model is trained solely on synthetic data generated using Blender. \n
        Upload an image to perform sky replacement.
        """
    )

    with gr.Row():
        # Left Column: Input Image and Run/Clear Buttons
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image")

            with gr.Row():
                clear_button = gr.Button("Clear")
                run_button = gr.Button("Submit", variant="primary")

        # Right Column: Output Images
        with gr.Column(scale=1):
            output_mask = gr.Image(type="numpy", label="Predicted Mask")
            output_sky = gr.Image(type="numpy", label="Sky Replacement")

    metadata_display = gr.Markdown(None)


    def load_example(example_image, example_type, example_desc):
        info = f"**Type:** {example_type}\n\n**Description:** {example_desc}"
        return example_image, info


    example_list = [
        ["examples/real_0054.jpg", "Real", "Good"],
        ["examples/real_0116.jpg", "Real", "Good"],
        ["examples/real_0585.jpg", "Real", "Good"],
        ["examples/synthetic_10635.jpg", "Synthetic", "Good"],
        ["examples/synthetic_10512.jpg", "Synthetic", "Good"],
        ["examples/real_0765.jpg", "Real", "Decent"],
        ["examples/real_0822.jpg", "Real", "Decent"],
        ["examples/synthetic_10795.jpg", "Synthetic", "Decent"],
        ["examples/synthetic_10560.jpg", "Synthetic", "Decent"],
        ["examples/synthetic_10679.jpg", "Synthetic", "Decent"],
        ["examples/real_0823.jpg", "Real", "Decent, lacks some details in the trees"],
        ["examples/real_bad_0007.jpg", "Real", "Bad, lacks details in the trees"],
        ["examples/real_bad_0934.jpg", "Real", "Bad, lacks details in the trees"],
    ]

    examples_component = gr.Examples(
        examples=example_list,
        inputs=[
            input_image,
            gr.Textbox(label="Real or Synthetic", value="", interactive=False, visible=False),
            gr.Textbox(label="Description", value="", interactive=False, visible=False),
        ],
        outputs=[input_image, metadata_display],
        fn=load_example,
        cache_examples=True,
        label="Examples (click an image to load it and see details)"
    )


    def clear_all():
        return gr.update(value=None), gr.update(value=None), gr.update(value=None)


    clear_button.click(fn=clear_all, inputs=[], outputs=[input_image, output_mask, output_sky])
    run_button.click(fn=predict, inputs=input_image, outputs=[output_mask, output_sky])

# Launch the interface
demo.launch(share=share_repo)
