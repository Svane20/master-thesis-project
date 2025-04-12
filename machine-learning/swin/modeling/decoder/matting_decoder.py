import torch
import torch.nn as nn
import torch.nn.functional as F


class MattingDecoder(nn.Module):
    def __init__(self, use_attn=False, refine_channels=16):
        super().__init__()
        self.use_attn = use_attn
        self.refine_channels = refine_channels

        # Bottom convolution (process 1/32 feature)
        self.conv_bottom = nn.Conv2d(768, 768, kernel_size=3, padding=1)
        self.bn_bottom = nn.BatchNorm2d(768)

        # Upsample + fuse with skip connections
        self.conv_up3 = nn.Conv2d(768 + 384, 384, kernel_size=3, padding=1)
        self.bn_up3 = nn.BatchNorm2d(384)

        self.conv_up2 = nn.Conv2d(384 + 192, 192, kernel_size=3, padding=1)
        self.bn_up2 = nn.BatchNorm2d(192)

        self.conv_up1 = nn.Conv2d(192 + 96, 96, kernel_size=3, padding=1)
        self.bn_up1 = nn.BatchNorm2d(96)

        self.conv_out = nn.Conv2d(96, 1, kernel_size=3, padding=1)

        # Detail refinement
        self.refine_conv1 = nn.Conv2d(4, self.refine_channels, kernel_size=3, padding=1)
        self.bn_refine1 = nn.BatchNorm2d(self.refine_channels)

        self.refine_conv2 = nn.Conv2d(self.refine_channels, self.refine_channels, kernel_size=3, padding=1)
        self.bn_refine2 = nn.BatchNorm2d(self.refine_channels)

        self.refine_conv3 = nn.Conv2d(self.refine_channels, 1, kernel_size=3, padding=1)

        # Attention gates
        if self.use_attn:
            self.reduce_768_to_384 = nn.Conv2d(768, 384, kernel_size=1)
            self.reduce_384_to_192 = nn.Conv2d(384, 192, kernel_size=1)
            self.reduce_192_to_96 = nn.Conv2d(192, 96, kernel_size=1)

            self.gate_16 = nn.Conv2d(384, 384, kernel_size=1)
            self.skip_16 = nn.Conv2d(384, 384, kernel_size=1)

            self.gate_8 = nn.Conv2d(192, 192, kernel_size=1)
            self.skip_8 = nn.Conv2d(192, 192, kernel_size=1)

            self.gate_4 = nn.Conv2d(96, 96, kernel_size=1)
            self.skip_4 = nn.Conv2d(96, 96, kernel_size=1)

    def forward(self, features, original_image):
        f1, f2, f3, f4 = features  # [1/4, 1/8, 1/16, 1/32]

        # Bottom (1/32)
        x = F.relu(self.bn_bottom(self.conv_bottom(f4)))

        # 1/16 stage
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')  # -> [B, 768, 32, 32]
        if self.use_attn:
            x_reduced = self.reduce_768_to_384(x)
            g = self.gate_16(x_reduced)
            skip = self.skip_16(f3)
            att = torch.sigmoid(g + skip)
            f3 = f3 * att
        x = torch.cat([x, f3], dim=1)
        x = F.relu(self.bn_up3(self.conv_up3(x)))  # -> [B, 384, 32, 32]

        # 1/8 stage
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.use_attn:
            x_reduced = self.reduce_384_to_192(x)
            g = self.gate_8(x_reduced)
            skip = self.skip_8(f2)
            att = torch.sigmoid(g + skip)
            f2 = f2 * att
        x = torch.cat([x, f2], dim=1)
        x = F.relu(self.bn_up2(self.conv_up2(x)))  # -> [B, 192, 64, 64]

        # 1/4 stage
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.use_attn:
            x_reduced = self.reduce_192_to_96(x)
            g = self.gate_4(x_reduced)
            skip = self.skip_4(f1)
            att = torch.sigmoid(g + skip)
            f1 = f1 * att
        x = torch.cat([x, f1], dim=1)
        x = F.relu(self.bn_up1(self.conv_up1(x)))  # -> [B, 96, 128, 128]

        # Upsample to full resolution and predict coarse alpha
        x = F.interpolate(x, size=original_image.shape[-2:], mode='nearest')  # -> [B, 96, 512, 512]
        coarse_alpha = self.conv_out(x)

        # Detail refinement
        refine_input = torch.cat([coarse_alpha, original_image], dim=1)
        r = F.relu(self.bn_refine1(self.refine_conv1(refine_input)))
        r = F.relu(self.bn_refine2(self.refine_conv2(r)))
        refined_alpha = self.refine_conv3(r)

        return torch.sigmoid(refined_alpha)
