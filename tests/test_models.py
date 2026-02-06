"""
Unit tests for GAN models.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Generator, Discriminator


class TestGenerator:
    """Test cases for Generator model."""
    
    def test_generator_creation(self):
        """Test generator can be created."""
        gen = Generator(latent_dim=100)
        assert gen is not None
        assert isinstance(gen, torch.nn.Module)
    
    def test_generator_forward(self):
        """Test generator forward pass."""
        gen = Generator(latent_dim=100)
        batch_size = 16
        z = torch.randn(batch_size, 100)
        
        output = gen(z)
        
        assert output.shape == (batch_size, 1, 28, 28)
        assert output.min() >= -1.0 and output.max() <= 1.0
    
    def test_generator_generate(self):
        """Test generator generate method."""
        gen = Generator(latent_dim=100)
        num_samples = 10
        
        samples = gen.generate(num_samples, device='cpu')
        
        assert samples.shape == (num_samples, 1, 28, 28)
    
    def test_generator_parameters(self):
        """Test generator has trainable parameters."""
        gen = Generator(latent_dim=100)
        params = sum(p.numel() for p in gen.parameters())
        
        assert params > 0
        assert all(p.requires_grad for p in gen.parameters())


class TestDiscriminator:
    """Test cases for Discriminator model."""
    
    def test_discriminator_creation(self):
        """Test discriminator can be created."""
        disc = Discriminator()
        assert disc is not None
        assert isinstance(disc, torch.nn.Module)
    
    def test_discriminator_forward(self):
        """Test discriminator forward pass."""
        disc = Discriminator()
        batch_size = 16
        x = torch.randn(batch_size, 1, 28, 28)
        
        output = disc(x)
        
        assert output.shape == (batch_size, 1)
        assert output.min() >= 0.0 and output.max() <= 1.0
    
    def test_discriminator_features(self):
        """Test discriminator feature extraction."""
        disc = Discriminator()
        batch_size = 16
        x = torch.randn(batch_size, 1, 28, 28)
        
        features = disc.get_features(x)
        
        assert features.shape[0] == batch_size
        assert len(features.shape) == 2
    
    def test_discriminator_parameters(self):
        """Test discriminator has trainable parameters."""
        disc = Discriminator()
        params = sum(p.numel() for p in disc.parameters())
        
        assert params > 0
        assert all(p.requires_grad for p in disc.parameters())


class TestModelsIntegration:
    """Integration tests for Generator and Discriminator."""
    
    def test_gan_pipeline(self):
        """Test complete GAN pipeline."""
        gen = Generator(latent_dim=100)
        disc = Discriminator()
        
        batch_size = 8
        z = torch.randn(batch_size, 100)
        
        # Generate fake images
        fake_images = gen(z)
        
        # Discriminate fake images
        fake_output = disc(fake_images)
        
        assert fake_output.shape == (batch_size, 1)
        
        # Discriminate real images
        real_images = torch.randn(batch_size, 1, 28, 28)
        real_output = disc(real_images)
        
        assert real_output.shape == (batch_size, 1)
    
    def test_gradient_flow(self):
        """Test gradients flow through models."""
        gen = Generator(latent_dim=100)
        disc = Discriminator()
        
        z = torch.randn(1, 100, requires_grad=True)
        fake_images = gen(z)
        output = disc(fake_images)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert z.grad is not None
        assert any(p.grad is not None for p in gen.parameters())
        assert any(p.grad is not None for p in disc.parameters())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
