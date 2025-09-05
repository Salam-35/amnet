# test_metrics.py
"""
Test evaluation metrics
"""

import pytest
import numpy as np
import torch
from amnet.metrics.segmentation import SegmentationMetrics


class TestSegmentationMetrics:
    """Test segmentation evaluation metrics"""

    @pytest.fixture
    def metrics_calculator(self):
        """Create metrics calculator"""
        return SegmentationMetrics(num_classes=5)

    def test_dice_coefficient(self, metrics_calculator):
        """Test Dice coefficient computation"""
        # Create perfect prediction
        pred = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]])
        target = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]])

        dice = metrics_calculator.dice_coefficient(pred, target, class_id=1)
        assert dice == 1.0  # Perfect match

        # Create no overlap prediction
        pred_no_overlap = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]])
        dice_no_overlap = metrics_calculator.dice_coefficient(pred_no_overlap, target, class_id=1)
        assert dice_no_overlap == 0.0  # No overlap

    def test_iou_score(self, metrics_calculator):
        """Test IoU score computation"""
        pred = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]])
        target = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]])

        iou = metrics_calculator.iou_score(pred, target, class_id=1)
        assert iou == 1.0  # Perfect match

    def test_hausdorff_distance(self, metrics_calculator):
        """Test Hausdorff distance computation"""
        # Create two identical 3D masks
        pred = np.zeros((5, 5, 5))
        target = np.zeros((5, 5, 5))
        
        # Create simple shapes
        pred[1:4, 1:4, 1:4] = 1
        target[1:4, 1:4, 1:4] = 1
        
        # Perfect match should give HD = 0
        hd = metrics_calculator.hausdorff_distance(pred, target, class_id=1)
        assert hd == 0.0
        
        # Create clearly different shapes
        pred_diff = np.zeros((5, 5, 5))
        target_diff = np.zeros((5, 5, 5))
        
        # Two separate non-overlapping cubes
        pred_diff[0:2, 0:2, 0:2] = 1  # Top-left cube
        target_diff[3:5, 3:5, 3:5] = 1  # Bottom-right cube
        
        hd_diff = metrics_calculator.hausdorff_distance(pred_diff, target_diff, class_id=1)
        assert hd_diff > 0.0  # Should be positive for non-overlapping shapes

    def test_hausdorff_distance_95(self, metrics_calculator):
        """Test 95th percentile Hausdorff distance computation"""
        # Create two 3D masks with some difference
        pred = np.zeros((6, 6, 6))
        target = np.zeros((6, 6, 6))
        
        # Create overlapping but different shapes
        pred[1:4, 1:4, 1:4] = 1
        target[2:5, 2:5, 2:5] = 1
        
        hd95 = metrics_calculator.hausdorff_distance_95(pred, target, class_id=1)
        hd = metrics_calculator.hausdorff_distance(pred, target, class_id=1)
        
        # HD95 should be <= HD (95th percentile <= maximum)
        assert hd95 <= hd
        assert hd95 >= 0.0

    def test_compute_all_metrics(self, metrics_calculator):
        """Test comprehensive metrics computation"""
        # Create mock predictions and targets
        batch_size, num_classes = 2, 5
        spatial_dims = (16, 24, 24)

        predictions = torch.randn(batch_size, num_classes, *spatial_dims)
        targets = torch.randint(0, num_classes, (batch_size, *spatial_dims))

        metrics = metrics_calculator.compute_all_metrics(predictions, targets)

        assert isinstance(metrics, dict)
        assert 'dice' in metrics
        assert 'iou' in metrics
        assert 'hd' in metrics
        assert 'hd95' in metrics
        assert 'asd' in metrics

        # Check that mean values are computed
        assert 'mean' in metrics['dice']
        assert 'mean' in metrics['iou']
        assert 'mean' in metrics['hd']
        assert 'mean' in metrics['hd95']
        assert 'mean' in metrics['asd']

        # Check that all metric values are reasonable
        assert 0.0 <= metrics['dice']['mean'] <= 1.0
        assert 0.0 <= metrics['iou']['mean'] <= 1.0
        # HD metrics can be inf for no overlap, so just check they're non-negative
        assert metrics['hd']['mean'] >= 0.0 or metrics['hd']['mean'] == float('inf')
        assert metrics['hd95']['mean'] >= 0.0 or metrics['hd95']['mean'] == float('inf')
        
        # HD95 should be <= HD (95th percentile <= maximum)
        if metrics['hd']['mean'] != float('inf') and metrics['hd95']['mean'] != float('inf'):
            assert metrics['hd95']['mean'] <= metrics['hd']['mean']

    def test_hausdorff_edge_cases(self, metrics_calculator):
        """Test Hausdorff distance edge cases"""
        # Test with empty masks
        empty_pred = np.zeros((3, 3, 3))
        empty_target = np.zeros((3, 3, 3))
        
        hd_empty = metrics_calculator.hausdorff_distance(empty_pred, empty_target, class_id=1)
        assert hd_empty == float('inf')  # No class 1 present
        
        hd95_empty = metrics_calculator.hausdorff_distance_95(empty_pred, empty_target, class_id=1)
        assert hd95_empty == float('inf')  # No class 1 present


if __name__ == "__main__":
    pytest.main([__file__])