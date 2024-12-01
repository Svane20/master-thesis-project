import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, SegformerModel

from constants.seg_former import MODEL_NAME
from models.model_builder import DecoderBlock


class UNETR(nn.Module):
    """
    UNET-R architecture using a SegFormer encoder

    Args:
        out_channels (int): Number of output channels
        model_name (str): Pre-trained SegFormer model. Default is 'nvidia/mit-b0'
    """

    def __init__(self, out_channels: int = 1, model_name: str = MODEL_NAME):
        super().__init__()

        # Load pre-trained SegFormer encoder
        self.encoder = SegformerModel.from_pretrained(
            model_name, output_hidden_states=True, ignore_mismatched_sizes=True
        )
        hidden_sizes = self.encoder.config.hidden_sizes

        # Unfreeze the encoder for fine-tuning
        for param in self.encoder.parameters():
            param.requires_grad = True

        # Decoder blocks
        self.decoder4 = DecoderBlock(
            in_channels=hidden_sizes[-1],  # 256
            out_channels=256,
            skip_channels=hidden_sizes[-2]  # 160
        )
        self.decoder3 = DecoderBlock(
            in_channels=256,
            out_channels=128,
            skip_channels=hidden_sizes[-3]  # 64
        )
        self.decoder2 = DecoderBlock(
            in_channels=128,
            out_channels=64,
            skip_channels=hidden_sizes[-4]  # 32
        )
        self.decoder1 = DecoderBlock(
            in_channels=64,
            out_channels=32
        )

        # Final prediction head
        self.output_head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Pass through encoder
        outputs = self.encoder(pixel_values=pixel_values)
        encoder_hidden_states = outputs.hidden_states  # Tuple of hidden states

        # Extract the last four hidden states
        enc_features = list(encoder_hidden_states[-4:])  # [32, 64, 160, 256]

        # Reverse the list to match the order [C3, C2, C1, C0]
        enc_features = enc_features[::-1]

        # Decoder with skip connections
        x = self.decoder4(enc_features[0], skip=enc_features[1])  # x4, skip x3
        x = self.decoder3(x, skip=enc_features[2])  # x3, skip x2
        x = self.decoder2(x, skip=enc_features[3])  # x2, skip x1
        x = self.decoder1(x)  # No skip connection

        # Upsample to original image size
        x = F.interpolate(
            x, size=(pixel_values.shape[2], pixel_values.shape[3]), mode='bilinear', align_corners=False
        )

        # Output head
        return self.output_head(x)


if __name__ == "__main__":
    # Create dummy input and mask
    dummy_input = torch.rand(1, 3, 224, 224)
    dummy_mask = torch.randint(0, 2, (1, 1, 224, 224)).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model_name = 'nvidia/mit-b0'
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = UNETR(out_channels=1, model_name=model_name)
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Prepare the input
    input_images = dummy_input.permute(0, 2, 3, 1).numpy()
    inputs = image_processor(
        images=input_images,
        return_tensors="pt",
        do_resize=True,
        size={"height": 224, "width": 224},
        do_rescale=False,
        do_normalize=True,
    )

    # Move pixel_values and masks to device
    pixel_values = inputs["pixel_values"].to(device)
    masks = dummy_mask.to(device)

    # Create a dummy dataloader
    dataloader = [{'image': pixel_values, 'mask': masks}]

    with torch.inference_mode():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(pixel_values=images)

    print(f"Output shape: {outputs.shape}")
    assert outputs.shape == masks.shape, f"Expected output shape {masks.shape} but got {outputs.shape}"
