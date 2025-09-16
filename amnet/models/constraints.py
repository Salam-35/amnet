"""
Anatomical Constraint Module for AMNet
Enforces spatial relationships between organs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class AnatomicalConstraintModule(nn.Module):
    """Enforces anatomical constraints between organs"""

    def __init__(self, num_classes: int = 16, feature_dim_2d: int = 1024, feature_dim_3d: int = 2048):
        super().__init__()
        self.num_classes = num_classes

        # Learnable spatial relationship matrix
        self.register_parameter(
            'spatial_relationships',
            nn.Parameter(torch.ones(num_classes, num_classes) * 0.5)
        )

        # Initialize with prior anatomical knowledge
        self._init_anatomical_priors()

        # Feature projection layers to map high-dim features to constraint space
        self.feature_proj_2d = nn.Linear(feature_dim_2d, 32)
        self.feature_proj_3d = nn.Linear(feature_dim_3d, 32)
        
        # Constraint network for learning complex relationships
        self.constraint_net = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),  # 32 + 32 = 64
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.activation_maps = None  # For visualization

    def _init_anatomical_priors(self):
        """Initialize spatial relationship matrix with anatomical knowledge"""
        # Set impossible relationships to 0 (non-learnable)
        impossible_pairs = [
            (1, 2), (1, 3),  # spleen cannot be adjacent to kidneys
            (4, 8), (4, 9),  # gallbladder cannot be adjacent to major vessels
            (12, 13),  # adrenal glands cannot be adjacent
        ]

        with torch.no_grad():
            for i, j in impossible_pairs:
                self.spatial_relationships[i, j] = 0.0
                self.spatial_relationships[j, i] = 0.0

            # Set known adjacent relationships to 1
            adjacent_pairs = [
                (6, 4),  # liver-gallbladder
                (6, 11),  # liver-pancreas
                (2, 12),  # right kidney-right adrenal
                (3, 13),  # left kidney-left adrenal
                (8, 9),  # aorta-IVC
            ]

            for i, j in adjacent_pairs:
                self.spatial_relationships[i, j] = 1.0
                self.spatial_relationships[j, i] = 1.0

    def compute_adjacency(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute organ adjacency from predictions"""
        B, C, D, H, W = predictions.shape

        # Convert to binary masks
        masks = F.softmax(predictions, dim=1)
        binary_masks = (masks > 0.5).float()

        # Compute adjacency using morphological operations
        adjacency_maps = torch.zeros(B, C, C, D, H, W, device=predictions.device)

        # 3D structuring element for adjacency
        kernel = torch.ones(1, 1, 3, 3, 3, device=predictions.device)

        for i in range(C):
            for j in range(i + 1, C):
                mask_i = binary_masks[:, i:i + 1]  # [B, 1, D, H, W]
                mask_j = binary_masks[:, j:j + 1]

                # Dilate mask_i
                dilated_i = F.conv3d(mask_i, kernel, padding=1)
                dilated_i = (dilated_i > 0).float()

                # Check overlap with mask_j
                adjacency = dilated_i * mask_j

                adjacency_maps[:, i, j] = adjacency.squeeze(1)
                adjacency_maps[:, j, i] = adjacency.squeeze(1)

        return adjacency_maps

    def constraint_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute anatomical constraint loss"""
        B, C, D, H, W = predictions.shape

        # Compute adjacency from predictions
        adjacency_maps = self.compute_adjacency(predictions)

        # Get spatial relationship constraints
        relationships = torch.sigmoid(self.spatial_relationships)  # [C, C]

        # Compute constraint violations
        total_loss = 0.0
        count = 0

        for i in range(C):
            for j in range(i + 1, C):
                # Get relationship constraint
                constraint = relationships[i, j]

                # Get predicted adjacency
                predicted_adj = adjacency_maps[:, i, j]

                # If constraint is 0 (impossible), penalize any adjacency
                if constraint < 0.1:
                    violation = torch.mean(predicted_adj)
                    total_loss += violation
                    count += 1

                # If constraint is 1 (must be adjacent), this could be enforced
                # but we focus on preventing impossible relationships

        return total_loss / max(count, 1)

    def forward(self,
                features_2d: torch.Tensor,
                features_3d: torch.Tensor,
                volume_shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
        """
        Forward pass of anatomical constraint module

        Args:
            features_2d: 2D branch features
            features_3d: 3D branch features
            volume_shape: Original volume shape [B, C, D, H, W]

        Returns:
            Constraint loss value
        """
        B, C, D, H, W = volume_shape

        # Project features to prediction space for constraint evaluation
        # This is a simplified version - in practice, you'd use the actual predictions

        # Project features to lower dimensions
        proj_2d = self.feature_proj_2d(features_2d)  # B, 32
        proj_3d = self.feature_proj_3d(features_3d)  # B, 32
        
        # Combine 2D and 3D projected features for constraint evaluation
        combined_features = torch.cat([
            proj_2d.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, D // 8, H // 8, W // 8),
            proj_3d.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, D // 8, H // 8, W // 8)
        ], dim=1)  # B, 64, D//8, H//8, W//8

        # Generate constraint activation maps
        constraint_activation = self.constraint_net(combined_features)
        self.activation_maps = constraint_activation.detach().cpu()

        # Return constraint regularization loss
        return torch.mean(constraint_activation)

    def get_activation_maps(self) -> Optional[torch.Tensor]:
        """Get constraint activation maps for visualization"""
        return self.activation_maps


class FeatureFusionAttention(nn.Module):
    """Attention-based feature fusion for multi-scale features"""

    def __init__(self, feature_dims: list, fusion_dim: int):
        super().__init__()
        self.feature_dims = feature_dims
        self.fusion_dim = fusion_dim

        # Feature projections
        self.projections = nn.ModuleList([
            nn.Linear(dim, fusion_dim) for dim in feature_dims
        ])

        # Attention computation
        self.attention_net = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim // 4, 1)
        )

    def forward(self, features: list) -> torch.Tensor:
        """Fuse multiple features using attention"""
        # Project all features to same dimension
        projected_features = []
        for i, feat in enumerate(features):
            projected = self.projections[i](feat)
            projected_features.append(projected)

        # Stack features
        stacked = torch.stack(projected_features, dim=1)  # [B, N_features, fusion_dim]

        # Compute attention weights
        attention_logits = self.attention_net(stacked)  # [B, N_features, 1]
        attention_weights = F.softmax(attention_logits, dim=1)

        # Weighted fusion
        fused = torch.sum(stacked * attention_weights, dim=1)

        return fused