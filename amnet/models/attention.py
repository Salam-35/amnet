"""
Attention mechanisms for AMNet
Slice attention and cross-dimensional attention modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


class SliceAttentionModule(nn.Module):
    """Slice attention for 2D feature aggregation"""

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Attention network
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        self.attention_weights = None  # Store for visualization

    def forward(self, features: torch.Tensor, view_name: str) -> torch.Tensor:
        """
        Apply slice attention to aggregate features

        Args:
            features: [B, N_slices, feature_dim]
            view_name: 'axial', 'coronal', or 'sagittal'

        Returns:
            Aggregated features: [B, feature_dim]
        """
        B, N, D = features.shape

        # Compute attention weights
        attention_logits = self.attention_net(features)  # [B, N, 1]
        attention_weights = F.softmax(attention_logits, dim=1)  # [B, N, 1]

        # Store for visualization
        self.attention_weights = {
            f'{view_name}_weights': attention_weights.detach().cpu()
        }

        # Weighted aggregation
        aggregated = torch.sum(features * attention_weights, dim=1)  # [B, D]

        return aggregated

    def get_attention_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get stored attention weights for visualization"""
        return self.attention_weights


class CrossDimensionalAttention(nn.Module):
    """Cross-dimensional attention between 2D and 3D features"""

    def __init__(self,
                 dim_2d: int,
                 dim_3d: int,
                 fusion_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.dim_2d = dim_2d
        self.dim_3d = dim_3d
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.head_dim = fusion_dim // num_heads

        assert fusion_dim % num_heads == 0, "fusion_dim must be divisible by num_heads"

        # Projection layers for 2D features
        self.proj_2d_q = nn.Linear(dim_2d, fusion_dim)
        self.proj_2d_k = nn.Linear(dim_2d, fusion_dim)
        self.proj_2d_v = nn.Linear(dim_2d, fusion_dim)

        # Projection layers for 3D features
        self.proj_3d_q = nn.Linear(dim_3d, fusion_dim)
        self.proj_3d_k = nn.Linear(dim_3d, fusion_dim)
        self.proj_3d_v = nn.Linear(dim_3d, fusion_dim)

        # Output projections
        self.proj_out_2d = nn.Linear(fusion_dim, fusion_dim)
        self.proj_out_3d = nn.Linear(fusion_dim, fusion_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        self.attention_weights_2d_to_3d = None
        self.attention_weights_3d_to_2d = None

    def multi_head_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple:
        """Multi-head attention computation"""
        B = q.size(0)

        # Reshape for multi-head attention
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        attended = torch.matmul(attention_weights, v)

        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(B, -1, self.fusion_dim)

        return attended, attention_weights

    def forward(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Cross-dimensional attention between 2D and 3D features

        Args:
            features_2d: [B, feature_dim_2d]
            features_3d: [B, feature_dim_3d]

        Returns:
            Dictionary with attended features and attention weights
        """
        B = features_2d.size(0)

        # Add sequence dimension for attention
        features_2d = features_2d.unsqueeze(1)  # [B, 1, dim_2d]
        features_3d = features_3d.unsqueeze(1)  # [B, 1, dim_3d]

        # 2D -> 3D attention (2D queries, 3D keys/values)
        q_2d = self.proj_2d_q(features_2d)
        k_3d = self.proj_3d_k(features_3d)
        v_3d = self.proj_3d_v(features_3d)

        attended_2d_to_3d, attn_weights_2d_to_3d = self.multi_head_attention(q_2d, k_3d, v_3d)
        attended_2d_to_3d = self.proj_out_2d(attended_2d_to_3d).squeeze(1)

        # 3D -> 2D attention (3D queries, 2D keys/values)
        q_3d = self.proj_3d_q(features_3d)
        k_2d = self.proj_2d_k(features_2d)
        v_2d = self.proj_2d_v(features_2d)

        attended_3d_to_2d, attn_weights_3d_to_2d = self.multi_head_attention(q_3d, k_2d, v_2d)
        attended_3d_to_2d = self.proj_out_3d(attended_3d_to_2d).squeeze(1)

        # Store attention weights for visualization
        self.attention_weights_2d_to_3d = attn_weights_2d_to_3d.detach().cpu()
        self.attention_weights_3d_to_2d = attn_weights_3d_to_2d.detach().cpu()

        return {
            '2d_to_3d': attended_2d_to_3d,
            '3d_to_2d': attended_3d_to_2d,
            'attention_weights': {
                '2d_to_3d': self.attention_weights_2d_to_3d,
                '3d_to_2d': self.attention_weights_3d_to_2d
            }
        }


class SpatialAttention3D(nn.Module):
    """3D spatial attention for feature refinement"""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 3D spatial attention"""
        # Channel attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights

        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weights = self.spatial_attention(spatial_input)
        x = x * spatial_weights

        return x