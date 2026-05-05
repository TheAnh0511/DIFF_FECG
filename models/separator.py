from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 9) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor):
        feat = self.conv(x)
        down = self.pool(feat)
        return feat, down


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            if diff > 0:
                x = F.pad(x, (0, diff))
            elif diff < 0:
                x = x[..., :skip.shape[-1]]

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SharedUNetEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 32) -> None:
        super().__init__()
        self.down1 = DownBlock(in_ch, base_ch)
        self.down2 = DownBlock(base_ch, base_ch * 2)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

    def forward(self, x: torch.Tensor):
        s1, x1 = self.down1(x)
        s2, x2 = self.down2(x1)
        s3, x3 = self.down3(x2)
        b = self.bottleneck(x3)
        return b, (s1, s2, s3)


class DecoderHead(nn.Module):
    def __init__(self, base_ch: int = 32, out_ch: int = 1) -> None:
        super().__init__()
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch)
        self.out_head = nn.Conv1d(base_ch, out_ch, kernel_size=1)

    def forward(self, b: torch.Tensor, skips):
        s1, s2, s3 = skips
        x = self.up3(b, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.out_head(x)


class DualHeadSeparator(nn.Module):
    """
    Single-channel separator:
      input  : aECG [B, 1, L]
      output : pred_mecg [B, 1, L], pred_fecg [B, 1, L]
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 32) -> None:
        super().__init__()
        self.encoder = SharedUNetEncoder(in_ch=in_ch, base_ch=base_ch)
        self.mecg_head = DecoderHead(base_ch=base_ch, out_ch=1)
        self.fecg_head = DecoderHead(base_ch=base_ch, out_ch=1)

    def forward(self, x: torch.Tensor):
        b, skips = self.encoder(x)
        pred_mecg = self.mecg_head(b, skips)
        pred_fecg = self.fecg_head(b, skips)
        return pred_mecg, pred_fecg