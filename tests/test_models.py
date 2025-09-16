# test_models.py
"""
Test model architectures and components
"""

import pytest
import torch
import torch.nn as nn
from amnet.models.amnet import AMNet
from amnet.models.encoders import ConvNeXtV2Encoder, ResNet3DEncoder
from amnet.models.attention import SliceAttentionModule, CrossDimensionalAttention
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.config import Config


class TestAMNetModel:
    """Test AMNet model architecture"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = Config()
        config.model.input_size = (64, 96, 96)  # Larger size for testing to accommodate pooling
        config.model.num_classes = 16
        config.model.scales = [1, 2]  # Reduce scales for testing with smaller input
        return config

    def test_model_creation(self, config):
        """Test model creation"""
        model = AMNet(config)

        assert isinstance(model, nn.Module)
        assert model.num_classes == 16

        # Check if model has required components
        assert hasattr(model, 'encoder_2d')
        assert hasattr(model, 'encoder_3d')
        assert hasattr(model, 'slice_attention')
        assert hasattr(model, 'cross_attention')
        assert hasattr(model, 'fusion_decoder')

    def test_model_forward(self, config):
        """Test model forward pass"""
        model = AMNet(config)
        model.eval()

        # Create test input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, *config.model.input_size)

        with torch.no_grad():
            outputs = model(input_tensor)

        assert isinstance(outputs, dict)
        assert 'predictions' in outputs
        assert 'constraint_loss' in outputs

        predictions = outputs['predictions']
        assert predictions.shape == (batch_size, config.model.num_classes, *config.model.input_size)

    def test_parameter_count(self, config):
        """Test parameter counting"""
        model = AMNet(config)
        param_count = model.count_parameters()

        assert isinstance(param_count, int)
        assert param_count > 0


class TestEncoders:
    """Test encoder components"""

    def test_convnext_encoder(self):
        """Test ConvNeXt V2 encoder"""
        encoder = ConvNeXtV2Encoder(in_chans=1, feature_dim=256)

        # Test forward pass
        input_tensor = torch.randn(2, 1, 128, 128)
        output = encoder(input_tensor)

        assert output.shape == (2, 256)

    def test_resnet3d_encoder(self):
        """Test 3D ResNet encoder"""
        encoder = ResNet3DEncoder(in_channels=1, feature_dim=512)

        # Test forward pass
        input_tensor = torch.randn(2, 1, 32, 64, 64)
        output = encoder(input_tensor)

        assert isinstance(output, dict)
        assert 'global_features' in output
        assert output['global_features'].shape == (2, 512)


class TestAttentionModules:
    """Test attention mechanisms"""

    def test_slice_attention(self):
        """Test slice attention module"""
        attention_module = SliceAttentionModule(feature_dim=256)

        # Test forward pass
        features = torch.randn(2, 32, 256)  # [B, N_slices, feature_dim]
        output = attention_module(features, 'axial')

        assert output.shape == (2, 256)

        # Check attention weights are stored
        weights = attention_module.get_attention_weights()
        assert weights is not None
        assert 'axial_weights' in weights

    def test_cross_dimensional_attention(self):
        """Test cross-dimensional attention"""
        attention_module = CrossDimensionalAttention(
            dim_2d=256, dim_3d=512, fusion_dim=384
        )

        # Test forward pass
        features_2d = torch.randn(2, 256)
        features_3d = torch.randn(2, 512)

        outputs = attention_module(features_2d, features_3d)

        assert isinstance(outputs, dict)
        assert '2d_to_3d' in outputs
        assert '3d_to_2d' in outputs
        assert outputs['2d_to_3d'].shape == (2, 384)
        assert outputs['3d_to_2d'].shape == (2, 384)


