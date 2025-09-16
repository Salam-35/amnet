"""
Multi-scale Fusion Decoder for AMNet
Combines 2D and 3D features with learnable fusion weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class MultiScaleFusionDecoder(nn.Module):
    """Multi-scale fusion decoder with learnable weights"""

    def __init__(self,
                 fusion_dim: int,
                 num_classes: int,
                 scales: List[int] = [1, 2, 4, 8],
                 decoder_channels: List[int] = [512, 256, 128, 64],
                 feature_dim_3d: int = 2048):
        super().__init__()

        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        self.scales = scales
        self.decoder_channels = decoder_channels
        self.feature_dim_3d = feature_dim_3d

        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(4))  # γ1, γ2, γ3, γ4

        # Feature projection layers
        self.proj_2d = nn.Linear(fusion_dim, fusion_dim)
        self.proj_3d = nn.Linear(fusion_dim, fusion_dim)

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()

        # Initial projection from fused features (including multiscale)
        # Total channels = fusion_dim + (decoder_channels[0] // len(scales)) * len(scales)
        total_input_channels = fusion_dim + decoder_channels[0]
        self.initial_conv = nn.Sequential(
            nn.Conv3d(total_input_channels, decoder_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        # Progressive upsampling blocks
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]

            block = DecoderBlock(in_ch, out_ch)
            self.decoder_blocks.append(block)

        # Final classification head
        self.classifier = nn.Conv3d(
            decoder_channels[-1],
            num_classes,
            kernel_size=1
        )

        # Multi-scale feature processing - ensure consistent spatial output
        target_spatial_size = (8, 12, 12)  # D//8, H//8, W//8 for test input size
        
        self.multiscale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(feature_dim_3d, decoder_channels[0] // len(scales),
                          kernel_size=3, padding=1),
                nn.BatchNorm3d(decoder_channels[0] // len(scales)),
                nn.ReLU(inplace=True)
            ) for s in scales
        ])

    def forward(self,
                features_2d: torch.Tensor,
                features_3d: torch.Tensor,
                multiscale_3d: Dict[str, torch.Tensor],
                target_shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
        """
        Decode fused features to segmentation predictions

        Args:
            features_2d: 2D branch attended features [B, fusion_dim]
            features_3d: 3D branch attended features [B, fusion_dim]
            multiscale_3d: Multi-scale 3D features
            target_shape: Target output shape [B, C, D, H, W]

        Returns:
            Segmentation predictions [B, num_classes, D, H, W]
        """
        B, _, D, H, W = target_shape

        # Project features
        proj_2d = self.proj_2d(features_2d)  # [B, fusion_dim]
        proj_3d = self.proj_3d(features_3d)  # [B, fusion_dim]

        # Normalize fusion weights
        weights = F.softmax(self.fusion_weights, dim=0)

        # Fuse features with learnable weights
        # F_fused = γ1·F_2D + γ2·F_3D + γ3·Attn_2D→3D + γ4·Attn_3D→2D
        fused_features = (
                weights[0] * proj_2d +
                weights[1] * proj_3d +
                weights[2] * features_2d +  # Already attended 2D→3D
                weights[3] * features_3d  # Already attended 3D→2D
        )

        # Reshape to spatial dimensions for 3D processing
        # Start with a small spatial size and upsample
        init_d, init_h, init_w = D // 8, H // 8, W // 8
        fused_spatial = fused_features.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fused_spatial = fused_spatial.expand(B, self.fusion_dim, init_d, init_h, init_w)

        # Process multi-scale 3D features
        multiscale_processed = []
        for i, (scale_name, scale_features) in enumerate(multiscale_3d.items()):
            # Scale features should already have shape [B, C, 1, 1, 1] from AdaptiveAvgPool3d
            # Just expand to desired spatial size
            scale_spatial = scale_features.expand(B, -1, init_d, init_h, init_w)

            processed = self.multiscale_processors[i](scale_spatial)
            multiscale_processed.append(processed)

        # Combine multi-scale features
        if multiscale_processed:
            multiscale_combined = torch.cat(multiscale_processed, dim=1)
            fused_spatial = torch.cat([fused_spatial, multiscale_combined], dim=1)

        # Initial convolution
        x = self.initial_conv(fused_spatial)

        # Progressive decoding with upsampling
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)

        # Final classification
        predictions = self.classifier(x)

        # Upsample to target size if needed
        if predictions.shape[2:] != (D, H, W):
            predictions = F.interpolate(
                predictions,
                size=(D, H, W),
                mode='trilinear',
                align_corners=False
            )

        return predictions


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and feature refinement"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Squeeze-and-excitation for channel attention
        self.se_block = SEBlock3D(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv_block(x)
        x = self.se_block(x)
        return x


class SEBlock3D(nn.Module):
    """3D Squeeze-and-Excitation block"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        # Ensure at least 1 channel after reduction
        reduced_channels = max(1, channels // reduction)
        self.excitation = nn.Sequential(
            nn.Conv3d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.squeeze(x)
        weights = self.excitation(weights)
        return x * weights