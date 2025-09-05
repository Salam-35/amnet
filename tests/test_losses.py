# test_losses.py
"""
Test loss functions
"""

import pytest
import torch
import torch.nn.functional as F
from amnet.losses.compound import CompoundLoss, DiceLoss, FocalLoss, BoundaryLoss


class TestLossFunctions:
    """Test loss function implementations"""

    def test_dice_loss(self):
        """Test Dice loss computation"""
        dice_loss = DiceLoss()

        # Create mock predictions and targets
        batch_size, num_classes = 2, 5
        spatial_dims = (16, 24, 24)

        predictions = torch.randn(batch_size, num_classes, *spatial_dims)
        targets = torch.randint(0, num_classes, (batch_size, *spatial_dims))

        loss = dice_loss(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss >= 0.0

    def test_focal_loss(self):
        """Test Focal loss computation"""
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)

        # Create mock data
        batch_size, num_classes = 2, 5
        spatial_dims = (16, 24, 24)

        predictions = torch.randn(batch_size, num_classes, *spatial_dims)
        targets = torch.randint(0, num_classes, (batch_size, *spatial_dims))

        loss = focal_loss(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss >= 0.0

    def test_compound_loss(self):
        """Test compound loss computation"""
        compound_loss = CompoundLoss()

        # Create mock data
        batch_size, num_classes = 2, 5
        spatial_dims = (16, 24, 24)

        predictions = torch.randn(batch_size, num_classes, *spatial_dims)
        targets = torch.randint(0, num_classes, (batch_size, *spatial_dims))
        constraint_loss = torch.tensor(0.1)

        loss_dict = compound_loss(predictions, targets, constraint_loss)

        assert isinstance(loss_dict, dict)
        assert 'total_loss' in loss_dict
        assert 'dice_loss' in loss_dict
        assert 'focal_loss' in loss_dict
        assert 'boundary_loss' in loss_dict
        assert 'constraint_loss' in loss_dict

        # Check that all losses are scalars
        for key, value in loss_dict.items():
            if key != 'loss_weights':
                assert isinstance(value, torch.Tensor)
                assert value.dim() == 0


