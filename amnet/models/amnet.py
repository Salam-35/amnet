"""
AMNet: Anatomically-aware Multi-scale Network
Main architecture implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import logging

from .encoders import ConvNeXtV2Encoder, ResNet3DEncoder
from .attention import SliceAttentionModule, CrossDimensionalAttention
from .constraints import AnatomicalConstraintModule
from .decoder import MultiScaleFusionDecoder

logger = logging.getLogger(__name__)


class AMNet(nn.Module):
    """
    AMNet: Anatomically-aware Multi-scale Network for Abdominal Organ Segmentation

    Combines 2D and 3D processing with anatomical constraints for robust segmentation.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes

        # 2D Branch - Process orthogonal views
        self.encoder_2d = ConvNeXtV2Encoder(
            feature_dim=config.feature_dim_2d,
            depths=[3, 3, 27, 3]
        )
        self.slice_attention = SliceAttentionModule(config.feature_dim_2d)

        # 3D Branch - Volumetric processing
        self.encoder_3d = ResNet3DEncoder(
            feature_dim=config.feature_dim_3d,
            layers=[3, 4, 6, 3]
        )

        # Multi-scale 3D feature extraction
        self.multiscale_pooling = nn.ModuleList([
            nn.AdaptiveAvgPool3d(1)  # Always pool to 1x1x1 for global features
            for s in config.scales
        ])

        # Anatomical Constraint Module
        self.constraint_module = AnatomicalConstraintModule(
            num_classes=config.num_classes,
            feature_dim_2d=config.feature_dim_2d,
            feature_dim_3d=config.feature_dim_3d
        )

        # Cross-dimensional attention
        self.cross_attention = CrossDimensionalAttention(
            dim_2d=config.feature_dim_2d,
            dim_3d=config.feature_dim_3d,
            fusion_dim=config.fusion_dim
        )

        # Fusion decoder - adaptive channels based on fusion dim
        if config.fusion_dim <= 32:
            decoder_channels = [64, 32, 16, 8]  # Ultra-lite
        elif config.fusion_dim <= 64:
            decoder_channels = [128, 64, 32, 16]  # Lite
        else:
            decoder_channels = [512, 256, 128, 64]  # Full
        self.fusion_decoder = MultiScaleFusionDecoder(
            fusion_dim=config.fusion_dim,
            num_classes=config.num_classes,
            scales=config.scales,
            decoder_channels=decoder_channels,
            feature_dim_3d=config.feature_dim_3d
        )

        # Initialize weights
        self._init_weights()

        logger.info(f"AMNet initialized with {self.count_parameters():,} parameters")

    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def extract_2d_views(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract orthogonal 2D views from 3D volume"""
        B, C, D, H, W = x.shape

        views = {}

        # Axial view (slice along depth)
        views['axial'] = x.transpose(2, 4).contiguous().view(B * D, C, H, W)

        # Coronal view (slice along height)
        views['coronal'] = x.transpose(2, 3).contiguous().view(B * H, C, D, W)

        # Sagittal view (slice along width)
        views['sagittal'] = x.permute(0, 1, 4, 2, 3).contiguous().view(B * W, C, D, H)

        return views

    def process_2d_branch(self, x: torch.Tensor) -> torch.Tensor:
        """Process 2D branch with multi-view encoding"""
        B, C, D, H, W = x.shape

        # Extract orthogonal views
        views = self.extract_2d_views(x)

        # Encode each view
        view_features = {}
        for view_name, view_data in views.items():
            features = self.encoder_2d(view_data)

            # Reshape back to batch dimension
            if view_name == 'axial':
                features = features.view(B, D, -1)
            elif view_name == 'coronal':
                features = features.view(B, H, -1)
            else:  # sagittal
                features = features.view(B, W, -1)

            # Apply slice attention
            view_features[view_name] = self.slice_attention(features, view_name)

        # Combine view features
        combined_2d = torch.stack([
            view_features['axial'],
            view_features['coronal'],
            view_features['sagittal']
        ], dim=1).mean(dim=1)  # Average across views

        return combined_2d

    def process_3d_branch(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process 3D branch with multi-scale extraction"""
        # 3D encoding
        features_3d_dict = self.encoder_3d(x)
        spatial_features_3d = features_3d_dict['spatial_features']
        global_features_3d = features_3d_dict['global_features']

        # Multi-scale feature extraction using spatial features
        multiscale_features = {}
        for i, scale in enumerate(self.config.scales):
            if scale == 1:
                multiscale_features[f'scale_{scale}'] = self.multiscale_pooling[i](spatial_features_3d)
            else:
                multiscale_features[f'scale_{scale}'] = self.multiscale_pooling[i](spatial_features_3d)

        return {
            'base': global_features_3d,
            'spatial': spatial_features_3d,
            'multiscale': multiscale_features
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of AMNet

        Args:
            x: Input CT volume [B, C, D, H, W]

        Returns:
            Dictionary containing predictions and auxiliary outputs
        """
        B, C, D, H, W = x.shape

        # Process 2D branch
        features_2d = self.process_2d_branch(x)

        # Process 3D branch
        features_3d_dict = self.process_3d_branch(x)
        features_3d = features_3d_dict['base']
        multiscale_3d = features_3d_dict['multiscale']

        # Cross-dimensional attention
        attended_features = self.cross_attention(features_2d, features_3d)

        # Anatomical constraints
        constraint_loss = self.constraint_module(features_2d, features_3d, x.shape)

        # Fusion and decoding
        predictions = self.fusion_decoder(
            features_2d=attended_features['2d_to_3d'],
            features_3d=attended_features['3d_to_2d'],
            multiscale_3d=multiscale_3d,
            target_shape=(B, self.num_classes, D, H, W)
        )

        return {
            'predictions': predictions,
            'constraint_loss': constraint_loss,
            'features_2d': features_2d,
            'features_3d': features_3d,
            'attention_weights': attended_features.get('attention_weights', None)
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_attention_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention maps for visualization"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            return {
                'slice_attention': self.slice_attention.get_attention_weights(),
                'cross_attention': output.get('attention_weights'),
                'constraint_activation': self.constraint_module.get_activation_maps()
            }