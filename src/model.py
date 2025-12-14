import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampleBlock(nn.Module):
    """Conv → Conv → MaxPool with skip connection"""
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        skip = x
        x = self.pool(x)
        return x, skip



class FlowNetSingleOutput(nn.Module):
    """3-level deep FlowNet with a single final flow output."""
    def __init__(self, in_channels=6):
        super().__init__()

        # ---- Encoder (3 levels) ----
        self.enc1 = DownsampleBlock(in_channels, 64, 7)    # Output: H×W, 64 channels
        self.enc2 = DownsampleBlock(64, 128, 5)            # Output: H/2×W/2, 128 channels
        self.enc3 = DownsampleBlock(128, 256, 5)           # Output: H/4×W/4, 256 channels

        # ---- Bottleneck ----
        self.bottleneck = nn.Sequential(                   # Input: H/8×W/8, 256 channels
            nn.Conv2d(256, 512, 3, padding=1),             # Output: H/8×W/8, 512 channels
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ---- Decoder ----
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)   # 512 → 256, H/8 → H/4
        self.up2 = nn.ConvTranspose2d(512, 128, 2, stride=2)   # 512 (256+256) → 128, H/4 → H/2
        self.up1 = nn.ConvTranspose2d(256, 64, 2, stride=2)    # 256 (128+128) → 64, H/2 → H

        # ---- Final flow prediction (only 1!) ----
        self.flow_final = nn.Conv2d(128, 2, 1)             # 128 (64+64) → 2, H×W

    def forward(self, x):
        # Encoder
        x, skip1 = self.enc1(x)    # x: H/2×W/2, skip1: H×W
        x, skip2 = self.enc2(x)    # x: H/4×W/4, skip2: H/2×W/2
        x, skip3 = self.enc3(x)    # x: H/8×W/8, skip3: H/4×W/4

        # Bottleneck
        x = self.bottleneck(x)     # x: H/8×W/8, 512 channels

        # Decoder
        x = self.up3(x)            # x: H/4×W/4, 256 channels
        x = torch.cat([x, skip3], dim=1)  # x: H/4×W/4, 512 channels (256+256)

        x = self.up2(x)            # x: H/2×W/2, 128 channels
        x = torch.cat([x, skip2], dim=1)  # x: H/2×W/2, 256 channels (128+128)

        x = self.up1(x)            # x: H×W, 64 channels
        x = torch.cat([x, skip1], dim=1)  # x: H×W, 128 channels (64+64)

        # Final full-resolution output
        return self.flow_final(x)  # Output: H×W, 2 channels (flow vectors)



class FlowNetMultiOutput(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()

        # ---- Encoder (3 levels) ----
        self.enc1 = DownsampleBlock(in_channels, 64, 7)    # Output: H×W, 64 channels
        self.enc2 = DownsampleBlock(64, 128, 5)            # Output: H/2×W/2, 128 channels
        self.enc3 = DownsampleBlock(128, 256, 5)           # Output: H/4×W/4, 256 channels

        # ---- Bottleneck ----
        self.bottleneck = nn.Sequential(                   # Input: H/8×W/8, 256 channels
            nn.Conv2d(256, 512, 3, padding=1),             # Output: H/8×W/8, 512 channels
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ---- Decoder feature upsampling ----
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256 + 256 + 16, 128, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(128 + 128 + 16, 64, 2, stride=2)

        # ---- Flow prediction heads ----
        self.flow3 = nn.Conv2d(512, 2, 1)      # Lowest resolution (H/8)
        self.up_flow3 = nn.ConvTranspose2d(2, 16, 2, stride=2)

        self.flow2 = nn.Conv2d(528, 2, 1)      # 256 + 256 + 16 (H/4)
        self.up_flow2 = nn.ConvTranspose2d(2, 16, 2, stride=2)

        self.flow1 = nn.Conv2d(272, 2, 1)      # 128 + 128 + 16 (H/2)
        self.up_flow1 = nn.ConvTranspose2d(2, 16, 2, stride=2)

        self.flow0 = nn.Conv2d(144, 2, 1)      # 64 + 64 + 16 (H - full resolution!)

    def forward(self, x):

        # Encoder
        x, s1 = self.enc1(x)    # x: H/2, s1: H (64 ch)
        x, s2 = self.enc2(x)    # x: H/4, s2: H/2 (128 ch)
        x, s3 = self.enc3(x)    # x: H/8, s3: H/4 (256 ch)

        # Bottleneck
        bottleneck = self.bottleneck(x)  # H/8, 512 ch

        # ---- Flow Level 3 at H/8 ----
        flow3 = self.flow3(bottleneck)   # H/8, 2 ch
        upf3 = self.up_flow3(flow3)      # H/4, 16 ch

        # ---- Level 2 at H/4 ----
        x = self.up3(bottleneck)         # H/4, 256 ch
        x = torch.cat([x, s3, upf3], dim=1)    # H/4, 528 ch (256+256+16)
        flow2 = self.flow2(x)            # H/4, 2 ch
        upf2 = self.up_flow2(flow2)      # H/2, 16 ch

        # ---- Level 1 at H/2 ----
        x = self.up2(x)                  # H/2, 128 ch
        x = torch.cat([x, s2, upf2], dim=1)    # H/2, 272 ch (128+128+16)
        flow1 = self.flow1(x)            # H/2, 2 ch
        upf1 = self.up_flow1(flow1)      # H, 16 ch

        # ---- Level 0 at H (FULL RESOLUTION) ----
        x = self.up1(x)                  # H, 64 ch
        x = torch.cat([x, s1, upf1], dim=1)    # H, 144 ch (64+64+16)
        flow0 = self.flow0(x)            # H, 2 ch

        return flow0, flow1, flow2, flow3



if __name__ == "__main__":
    # ---------------------------------------
    # Test input: batch of 1, 6 channels, H=W=256
    # ---------------------------------------
    x = torch.randn(1, 6, 256, 256)

    print("\n===== Testing FlowNetSingleOutput =====")
    single_model = FlowNetSingleOutput()
    out_single = single_model(x)
    print("Single-output flow shape:", out_single.shape)


    print("\n===== Testing FlowNetMultiOutput =====")
    multi_model = FlowNetMultiOutput()
    out_multi = multi_model(x)

    flow0, flow1, flow2, flow3 = out_multi

    print("flow0 (H, W):     ", flow0.shape)   # FULL resolution 1024x1024
    print("flow1 (H/2, W/2): ", flow1.shape)   # 512x512
    print("flow2 (H/4, W/4): ", flow2.shape)   # 256x256
    print("flow3 (H/8, W/8): ", flow3.shape)   # 128x128

    print("\nShape test completed successfully.\n")