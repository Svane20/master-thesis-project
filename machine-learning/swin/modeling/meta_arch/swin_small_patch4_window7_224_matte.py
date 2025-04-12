import torch
import torch.nn as nn
from torchsummary import summary

from typing import Any, Dict

from swin.modeling.backbone.swin_small_patch4_window7_224 import SwinEncoder
from swin.modeling.decoder.matting_decoder import MattingDecoder
from libs.utils.mem_utils import estimate_max_batch_size


class SwinMattingModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        encoder_config = config['encoder']
        decoder_config = config['decoder']

        self.encoder = SwinEncoder(model_name=encoder_config["model_name"])
        self.decoder = MattingDecoder(
            use_attn=decoder_config["use_attn"],
            refine_channels=decoder_config["refine_channels"]
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image [B, 3, 512, 512], normalized as needed for Swin.
        Returns:
            torch.Tensor: Alpha matte [B, 1, 512, 512].
        """
        features = self.encoder(x)  # list of 4 feature maps
        return self.decoder(features, x)  # decoded and refined alpha matte


if __name__ == "__main__":
    image_size = 512
    input_size = (3, image_size, image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)  # (1, 3, H, W)

    config = {
        "encoder": {
            "model_name": "microsoft/swin-small-patch4-window7-224"
        },
        "decoder": {
            "use_attn": True,
            "refine_channels": 16
        }
    }

    model = SwinMattingModel(config).to(device)
    summary(model, input_size=input_size)

    with torch.no_grad():
        # Encoder forward
        enc_features = model.encoder(dummy_input)
        print("Encoder output feature shapes:")
        for i, feat in enumerate(enc_features, start=1):
            print(f"  Stage {i} feature: {feat.shape}")
        # Decoder forward (with intermediate shape prints)
        f1, f2, f3, f4 = enc_features
        # Bottom (1/32) feature
        x = nn.functional.relu(model.decoder.bn_bottom(model.decoder.conv_bottom(f4)))
        print(f"Decoder: bottom 1/32 processed shape: {x.shape}")

        # 1/16 stage
        x = nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        if model.decoder.use_attn:
            x_reduced = model.decoder.reduce_768_to_384(x)
            g = model.decoder.gate_16(x_reduced)
            skip = model.decoder.skip_16(f3)
            att = torch.sigmoid(g + skip)
            f3_used = f3 * att
        else:
            f3_used = f3
        x = torch.cat([x, f3_used], dim=1)
        x = nn.functional.relu(model.decoder.bn_up3(model.decoder.conv_up3(x)))
        print(f"Decoder: after upsample to 1/16 and fusion shape: {x.shape}")

        # 1/8 stage
        x = nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        if model.decoder.use_attn:
            x_reduced = model.decoder.reduce_384_to_192(x)
            g = model.decoder.gate_8(x_reduced)
            skip = model.decoder.skip_8(f2)
            att = torch.sigmoid(g + skip)
            f2_used = f2 * att
        else:
            f2_used = f2
        x = torch.cat([x, f2_used], dim=1)
        x = nn.functional.relu(model.decoder.bn_up2(model.decoder.conv_up2(x)))
        print(f"Decoder: after upsample to 1/8 and fusion shape: {x.shape}")

        # 1/4 stage
        x = nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        if model.decoder.use_attn:
            x_reduced = model.decoder.reduce_192_to_96(x)
            g = model.decoder.gate_4(x_reduced)
            skip = model.decoder.skip_4(f1)
            att = torch.sigmoid(g + skip)
            f1_used = f1 * att
        else:
            f1_used = f1
        x = torch.cat([x, f1_used], dim=1)
        x = nn.functional.relu(model.decoder.bn_up1(model.decoder.conv_up1(x)))
        print(f"Decoder: after upsample to 1/4 and fusion shape: {x.shape}")

        # Full resolution stage
        x = nn.functional.interpolate(x, scale_factor=4.0, mode='nearest')
        coarse_alpha = model.decoder.conv_out(x)
        print(f"Decoder: coarse alpha shape: {coarse_alpha.shape}")

        # Detail refinement
        refine_in = torch.cat([coarse_alpha, dummy_input], dim=1)
        r = nn.functional.relu(model.decoder.bn_refine1(model.decoder.refine_conv1(refine_in)))
        r = nn.functional.relu(model.decoder.bn_refine2(model.decoder.refine_conv2(r)))
        refined_alpha = model.decoder.refine_conv3(r)
        alpha_out = torch.sigmoid(refined_alpha)
        print(f"Decoder: refined alpha output shape: {alpha_out.shape}")


    def generate_inputs(batch_size, input_size, device):
        return (torch.randn(batch_size, *input_size, device=device),)


    # Input resolutions to test
    for res in [224, 512, 1024]:
        input_size = (3, res, res)
        print(f"\nTesting input size: {input_size}...")

        # Estimate max batch size for this resolution
        max_batch_size = estimate_max_batch_size(
            model,
            input_size=input_size,
            max_memory_gb=12.0,
            safety_factor=0.9,
            input_generator_fn=generate_inputs
        )

        print(f"Estimated max batch size for {res}x{res}: {max_batch_size}")
