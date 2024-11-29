import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoImageProcessor, SegformerModel


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = None):
        """
        Decoder block for the UNet architecture

        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
            skip_channels (int): The number of channels in the skip connection (optional)
        """
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # If skip_channels is provided, align its channels to out_channels
        self.reduce_channels = nn.Conv2d(skip_channels, out_channels, kernel_size=1) if skip_channels else None

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, skip: Tensor = None) -> Tensor:
        # Upsample the input tensor
        x = self.upsample(x)

        if skip is not None:
            # Align channels of skip connection with x
            if self.reduce_channels:
                skip = self.reduce_channels(skip)

            # Align spatial dimensions with x
            if x.shape[2:] != skip.shape[2:]:
                skip = nn.functional.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)

            x = x + skip  # Add skip connection

        x = self.activation(self.norm(self.conv1(x)))
        x = self.activation(self.norm(self.conv2(x)))
        return x


class UNETR(nn.Module):
    def __init__(self, out_channels: int = 1, model_name: str = 'nvidia/mit-b0'):
        super().__init__()

        # Load pre-trained SegFormer encoder
        self.encoder = SegformerModel.from_pretrained(
            model_name, output_hidden_states=True, ignore_mismatched_sizes=True
        )
        hidden_sizes = self.encoder.config.hidden_sizes

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

    def forward(self, pixel_values: Tensor) -> Tensor:
        # Pass through encoder
        outputs = self.encoder(pixel_values=pixel_values)
        hidden_states = outputs.hidden_states  # A tuple of hidden states

        # Get the hidden states from the encoder
        enc_features = hidden_states

        # Decoder with skip connections
        x = self.decoder4(enc_features[-1], skip=enc_features[-2])
        x = self.decoder3(x, skip=enc_features[-3])
        x = self.decoder2(x, skip=enc_features[-4])
        x = self.decoder1(x)  # No skip connection

        # Upsample to original image size
        x = F.interpolate(x, size=(pixel_values.shape[2], pixel_values.shape[3]), mode='bilinear', align_corners=False)

        # Output head
        return self.output_head(x)

if __name__ == "__main__":
    # Create dummy input
    dummy_input = torch.rand(1, 3, 224, 224)
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

    # Move pixel_values to device
    pixel_values = inputs["pixel_values"].to(device)

    # Perform a forward pass
    with torch.no_grad():
        out = model(pixel_values=pixel_values)

    print(f"Output shape: {out.shape}")
    assert out.shape == (1, 1, 224, 224)
