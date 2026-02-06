"""
Unit tests for training components.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.losses import (
    discriminator_loss,
    generator_loss,
    gradient_penalty,
    FeatureMatchingLoss
)


class TestLossFunctions:
    """Test cases for loss functions."""
    
    def test_discriminator_loss_bce(self):
        """Test discriminator BCE loss."""
        batch_size = 16
        real_output = torch.rand(batch_size, 1) * 0.5 + 0.5  # [0.5, 1.0]
        fake_output = torch.rand(batch_size, 1) * 0.5  # [0.0, 0.5]
        
        total_loss, real_loss, fake_loss = discriminator_loss(
            real_output, fake_output, label_smoothing=0.1, loss_type='bce'
        )
        
        assert total_loss > 0
        assert real_loss > 0
        assert fake_loss > 0
        assert total_loss == real_loss + fake_loss
    
    def test_generator_loss_bce(self):
        """Test generator BCE loss."""
        batch_size = 16
        fake_output = torch.rand(batch_size, 1)
        
        loss = generator_loss(fake_output, loss_type='bce')
        
        assert loss > 0
        assert isinstance(loss.item(), float)
    
    def test_feature_matching_loss(self):
        """Test feature matching loss."""
        batch_size = 16
        feature_dim = 128
        
        real_features = torch.randn(batch_size, feature_dim)
        fake_features = torch.randn(batch_size, feature_dim)
        
        fm_loss = FeatureMatchingLoss()
        loss = fm_loss(real_features, fake_features)
        
        assert loss >= 0
        assert isinstance(loss.item(), float)
    
    def test_loss_backward(self):
        """Test loss functions support backward pass."""
        batch_size = 8
        real_output = torch.rand(batch_size, 1, requires_grad=True)
        fake_output = torch.rand(batch_size, 1, requires_grad=True)
        
        total_loss, _, _ = discriminator_loss(real_output, fake_output)
        
        # Should be able to backward
        total_loss.backward()
        
        assert real_output.grad is not None
        assert fake_output.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
