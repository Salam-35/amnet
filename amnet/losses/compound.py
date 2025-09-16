"""
Compound Loss Function for AMNet
Combines Dice, Focal, Boundary, and Constraint losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
import numpy as np
from typing import Dict, Optional

class DiceLoss(nn.Module):
    """Multi-class Dice Loss"""

    def __init__(self, smooth: float = 1e-6, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, C, D, H, W] - logits
            targets: [B, D, H, W] - class indices
        """
        # Convert predictions to probabilities
        probs = F.softmax(predictions, dim=1)

        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=predictions.size(1))
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()

        # Compute Dice for each class
        dice_scores = []
        for c in range(predictions.size(1)):
            if c == self.ignore_index:
                continue

            pred_c = probs[:, c]
            target_c = targets_one_hot[:, c]

            intersection = torch.sum(pred_c * target_c, dim=(1, 2, 3))
            union = torch.sum(pred_c, dim=(1, 2, 3)) + torch.sum(target_c, dim=(1, 2, 3))

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        # Average Dice across classes and batch
        dice_loss = 1.0 - torch.stack(dice_scores, dim=1).mean()
        return dice_loss

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, ignore_index: int = -1,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, C, D, H, W] - logits
            targets: [B, D, H, W] - class indices
        """
        # Reshape for cross entropy
        B, C, D, H, W = predictions.shape
        predictions = predictions.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        targets = targets.view(-1)

        # Mask out ignore index
        if self.ignore_index >= 0:
            valid_mask = targets != self.ignore_index
            predictions = predictions[valid_mask]
            targets = targets[valid_mask]

        # Compute cross entropy with class weights (move to same device)
        class_weights = self.class_weights.to(predictions.device) if self.class_weights is not None else None
        ce_loss = F.cross_entropy(predictions, targets, weight=class_weights, reduction='none')

        # Compute focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()

class BoundaryLoss(nn.Module):
    """Boundary-aware loss for precise segmentation"""

    def __init__(self, theta0: float = 3.0, theta: float = 5.0):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def compute_distance_transform(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute distance transform for boundary loss"""
        # Convert to numpy for scipy operations
        mask_np = mask.detach().cpu().numpy()

        distance_maps = []
        for b in range(mask_np.shape[0]):
            for c in range(mask_np.shape[1]):
                # Compute distance transform
                dt = ndimage.distance_transform_edt(1 - mask_np[b, c])
                distance_maps.append(dt)

        # Convert back to tensor
        distance_tensor = torch.from_numpy(np.stack(distance_maps)).float()
        distance_tensor = distance_tensor.view_as(mask).to(mask.device)

        return distance_tensor

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, C, D, H, W] - logits
            targets: [B, D, H, W] - class indices
        """
        # Convert predictions to probabilities
        probs = F.softmax(predictions, dim=1)

        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=predictions.size(1))
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()

        # Compute distance transforms
        distance_maps = self.compute_distance_transform(targets_one_hot)

        # Boundary loss computation
        boundary_loss = torch.mean(probs * distance_maps)

        return boundary_loss

class CompoundLoss(nn.Module):
    """Combined loss function for AMNet"""

    def __init__(self,
                 alpha: float = 1.0,    # Dice weight
                 beta: float = 0.5,     # Focal weight
                 gamma: float = 0.3,    # Boundary weight
                 delta: float = 0.2):   # Constraint weight
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.dice_loss = DiceLoss()
        # Create class weights for medical data (heavily weight foreground classes)
        class_weights = torch.tensor([
            0.1,  # background (low weight)
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # organs (normal weight)
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  # more organs
        ], dtype=torch.float32)
        self.focal_loss = FocalLoss(class_weights=class_weights)
        self.boundary_loss = BoundaryLoss()

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                constraint_loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute compound loss

        Args:
            predictions: Model predictions [B, C, D, H, W]
            targets: Ground truth masks [B, D, H, W]
            constraint_loss: Anatomical constraint loss

        Returns:
            Dictionary with individual and total losses
        """

        # Individual losses
        dice = self.dice_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)
        boundary = self.boundary_loss(predictions, targets)
        constraint = constraint_loss

        # Total weighted loss
        total_loss = (
            self.alpha * dice +
            self.beta * focal +
            self.gamma * boundary +
            self.delta * constraint
        )

        return {
            'total_loss': total_loss,
            'dice_loss': dice,
            'focal_loss': focal,
            'boundary_loss': boundary,
            'constraint_loss': constraint,
            'loss_weights': {
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
                'delta': self.delta
            }
        }