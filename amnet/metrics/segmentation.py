"""
Segmentation Metrics for AMNet
Comprehensive evaluation metrics including Dice, IoU, HD95, ASD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """Comprehensive segmentation metrics calculator"""

    def __init__(self, num_classes: int = 16, ignore_background: bool = True):
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.class_names = [
            "background", "spleen", "right_kidney", "left_kidney", "gallbladder",
            "esophagus", "liver", "stomach", "aorta", "IVC", "portal_vein",
            "pancreas", "right_adrenal", "left_adrenal", "duodenum", "bladder"
        ]

    def dice_coefficient(self, pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
        """Compute Dice coefficient for a specific class"""
        pred_mask = (pred == class_id).astype(np.float32)
        target_mask = (target == class_id).astype(np.float32)

        intersection = np.sum(pred_mask * target_mask)
        union = np.sum(pred_mask) + np.sum(target_mask)

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return (2.0 * intersection) / union

    def iou_score(self, pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
        """Compute IoU (Jaccard) score for a specific class"""
        pred_mask = (pred == class_id).astype(np.float32)
        target_mask = (target == class_id).astype(np.float32)

        intersection = np.sum(pred_mask * target_mask)
        union = np.sum(pred_mask) + np.sum(target_mask) - intersection

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return intersection / union

    def hausdorff_distance(self, pred: np.ndarray, target: np.ndarray,
                          class_id: int, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
        """Compute maximum Hausdorff distance"""
        pred_mask = (pred == class_id).astype(np.uint8)
        target_mask = (target == class_id).astype(np.uint8)

        # Extract surface points
        pred_surface = self._extract_surface_points(pred_mask, spacing)
        target_surface = self._extract_surface_points(target_mask, spacing)

        if len(pred_surface) == 0 or len(target_surface) == 0:
            return float('inf')

        # Compute directed Hausdorff distances
        hd1 = directed_hausdorff(pred_surface, target_surface)[0]
        hd2 = directed_hausdorff(target_surface, pred_surface)[0]

        # Return maximum Hausdorff distance
        return max(hd1, hd2)

    def hausdorff_distance_95(self, pred: np.ndarray, target: np.ndarray,
                              class_id: int, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
        """Compute 95th percentile Hausdorff distance"""
        pred_mask = (pred == class_id).astype(np.uint8)
        target_mask = (target == class_id).astype(np.uint8)

        # Extract surface points
        pred_surface = self._extract_surface_points(pred_mask, spacing)
        target_surface = self._extract_surface_points(target_mask, spacing)

        if len(pred_surface) == 0 or len(target_surface) == 0:
            return float('inf')

        # Compute all distances from pred to target
        from scipy.spatial.distance import cdist
        distances_pt = cdist(pred_surface, target_surface)
        distances_tp = cdist(target_surface, pred_surface)
        
        # Get minimum distance for each point
        min_distances_pt = np.min(distances_pt, axis=1)
        min_distances_tp = np.min(distances_tp, axis=1)
        
        # Combine all minimum distances
        all_distances = np.concatenate([min_distances_pt, min_distances_tp])
        
        # Return 95th percentile
        return np.percentile(all_distances, 95)

    def average_surface_distance(self, pred: np.ndarray, target: np.ndarray,
                                 class_id: int, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
        """Compute average surface distance"""
        pred_mask = (pred == class_id).astype(np.uint8)
        target_mask = (target == class_id).astype(np.uint8)

        # Extract surface points
        pred_surface = self._extract_surface_points(pred_mask, spacing)
        target_surface = self._extract_surface_points(target_mask, spacing)

        if len(pred_surface) == 0 or len(target_surface) == 0:
            return float('inf')

        # Compute distances from each predicted surface point to target surface
        distances = []
        for pred_point in pred_surface:
            min_dist = np.min(np.linalg.norm(target_surface - pred_point, axis=1))
            distances.append(min_dist)

        return np.mean(distances)

    def _extract_surface_points(self, mask: np.ndarray,
                                spacing: Tuple[float, float, float]) -> np.ndarray:
        """Extract surface points from binary mask"""
        if np.sum(mask) == 0:
            return np.array([]).reshape(0, 3)

        # Compute gradient to find boundaries
        grad_z = ndimage.sobel(mask.astype(np.float32), axis=0)
        grad_y = ndimage.sobel(mask.astype(np.float32), axis=1)
        grad_x = ndimage.sobel(mask.astype(np.float32), axis=2)

        gradient_magnitude = np.sqrt(grad_z ** 2 + grad_y ** 2 + grad_x ** 2)
        surface_mask = gradient_magnitude > 0.1

        # Get surface coordinates
        surface_coords = np.where(surface_mask)
        surface_points = np.column_stack(surface_coords).astype(np.float32)

        # Apply spacing
        surface_points[:, 0] *= spacing[0]  # Z
        surface_points[:, 1] *= spacing[1]  # Y
        surface_points[:, 2] *= spacing[2]  # X

        return surface_points

    def compute_all_metrics(self,
                            predictions: torch.Tensor,
                            targets: torch.Tensor,
                            spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[str, Dict[str, float]]:
        """
        Compute all segmentation metrics

        Args:
            predictions: [B, C, D, H, W] - model predictions (logits)
            targets: [B, D, H, W] - ground truth masks
            spacing: Voxel spacing (z, y, x)

        Returns:
            Dictionary of metrics per class and averages
        """
        # Convert predictions to class predictions
        pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
        targets_np = targets.cpu().numpy()

        metrics = {
            'dice': {},
            'iou': {},
            'hd': {},
            'hd95': {},
            'asd': {}
        }

        # Compute metrics for each class
        start_idx = 1 if self.ignore_background else 0

        for class_id in range(start_idx, self.num_classes):
            class_name = self.class_names[class_id]

            dice_scores = []
            iou_scores = []
            hd_scores = []
            hd95_scores = []
            asd_scores = []

            # Compute metrics for each sample in batch
            for b in range(pred_classes.shape[0]):
                pred_sample = pred_classes[b]
                target_sample = targets_np[b]

                # Dice and IoU
                dice = self.dice_coefficient(pred_sample, target_sample, class_id)
                iou = self.iou_score(pred_sample, target_sample, class_id)

                dice_scores.append(dice)
                iou_scores.append(iou)

                # Surface metrics (skip if class not present)
                if np.sum(target_sample == class_id) > 0 and np.sum(pred_sample == class_id) > 0:
                    hd = self.hausdorff_distance(pred_sample, target_sample, class_id, spacing)
                    hd95 = self.hausdorff_distance_95(pred_sample, target_sample, class_id, spacing)
                    asd = self.average_surface_distance(pred_sample, target_sample, class_id, spacing)

                    hd_scores.append(hd)
                    hd95_scores.append(hd95)
                    asd_scores.append(asd)

            # Store class metrics
            metrics['dice'][class_name] = np.mean(dice_scores)
            metrics['iou'][class_name] = np.mean(iou_scores)

            if hd_scores:
                metrics['hd'][class_name] = np.mean([h for h in hd_scores if h != float('inf')])
                metrics['hd95'][class_name] = np.mean([h for h in hd95_scores if h != float('inf')])
                metrics['asd'][class_name] = np.mean([a for a in asd_scores if a != float('inf')])
            else:
                metrics['hd'][class_name] = float('inf')
                metrics['hd95'][class_name] = float('inf')
                metrics['asd'][class_name] = float('inf')

        # Compute averages
        for metric_name in ['dice', 'iou', 'hd', 'hd95', 'asd']:
            valid_scores = [v for v in metrics[metric_name].values() if v != float('inf')]
            metrics[metric_name]['mean'] = np.mean(valid_scores) if valid_scores else 0.0

        return metrics


class CompoundLoss(nn.Module):
    """Combined loss function for AMNet"""

    def __init__(self,
                 alpha: float = 1.0,  # Dice weight
                 beta: float = 0.5,  # Focal weight
                 gamma: float = 0.3,  # Boundary weight
                 delta: float = 0.2):  # Constraint weight
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
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