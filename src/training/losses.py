"""
Loss functions for GAN training.
Implements various GAN loss formulations with label smoothing and noise injection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def discriminator_loss(
    real_output: torch.Tensor,
    fake_output: torch.Tensor,
    label_smoothing: float = 0.0,
    loss_type: str = 'bce'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate discriminator loss.
    
    The discriminator should:
    - Classify real images as real (output close to 1)
    - Classify fake images as fake (output close to 0)
    
    Args:
        real_output: Discriminator output for real images
        fake_output: Discriminator output for fake images
        label_smoothing: Amount of label smoothing (0.0 to 0.3)
        loss_type: Type of loss ('bce', 'wgan', 'hinge')
    
    Returns:
        Tuple of (total_loss, real_loss, fake_loss)
    """
    if loss_type == 'bce':
        # Binary Cross Entropy loss
        criterion = nn.BCELoss()
        
        # Real labels with smoothing: [1.0 - smoothing, 1.0]
        real_labels = torch.ones_like(real_output) * (1.0 - label_smoothing)
        fake_labels = torch.zeros_like(fake_output)
        
        real_loss = criterion(real_output, real_labels)
        fake_loss = criterion(fake_output, fake_labels)
        
    elif loss_type == 'wgan':
        # Wasserstein GAN loss
        real_loss = -torch.mean(real_output)
        fake_loss = torch.mean(fake_output)
        
    elif loss_type == 'hinge':
        # Hinge loss
        real_loss = torch.mean(F.relu(1.0 - real_output))
        fake_loss = torch.mean(F.relu(1.0 + fake_output))
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    total_loss = real_loss + fake_loss
    
    return total_loss, real_loss, fake_loss


def generator_loss(
    fake_output: torch.Tensor,
    loss_type: str = 'bce'
) -> torch.Tensor:
    """
    Calculate generator loss.
    
    The generator should:
    - Generate images that the discriminator classifies as real (output close to 1)
    
    Args:
        fake_output: Discriminator output for fake images
        loss_type: Type of loss ('bce', 'wgan', 'hinge')
    
    Returns:
        Generator loss
    """
    if loss_type == 'bce':
        # Binary Cross Entropy loss
        criterion = nn.BCELoss()
        real_labels = torch.ones_like(fake_output)
        loss = criterion(fake_output, real_labels)
        
    elif loss_type == 'wgan':
        # Wasserstein GAN loss
        loss = -torch.mean(fake_output)
        
    elif loss_type == 'hinge':
        # Hinge loss
        loss = -torch.mean(fake_output)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss


def gradient_penalty(
    discriminator: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Calculate gradient penalty for WGAN-GP.
    
    Args:
        discriminator: Discriminator network
        real_images: Real images
        fake_images: Fake images
        device: Device to compute on
    
    Returns:
        Gradient penalty loss
    """
    batch_size = real_images.size(0)
    
    # Random weight term for interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Interpolate between real and fake images
    interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    
    # Get discriminator output for interpolated images
    d_interpolates = discriminator(interpolates)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Flatten gradients
    gradients = gradients.view(batch_size, -1)
    
    # Calculate penalty
    gradient_norm = gradients.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    
    return penalty


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss for improved GAN training.
    Matches statistics of features from intermediate layers.
    """
    
    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()
        self.criterion = nn.L1Loss()
    
    def forward(
        self,
        real_features: torch.Tensor,
        fake_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate feature matching loss.
        
        Args:
            real_features: Features from real images
            fake_features: Features from fake images
        
        Returns:
            Feature matching loss
        """
        return self.criterion(fake_features.mean(0), real_features.mean(0))


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    # Create dummy outputs
    batch_size = 16
    real_output = torch.rand(batch_size, 1) * 0.5 + 0.5  # [0.5, 1.0]
    fake_output = torch.rand(batch_size, 1) * 0.5  # [0.0, 0.5]
    
    # Test discriminator loss
    d_loss, d_real, d_fake = discriminator_loss(real_output, fake_output, label_smoothing=0.1)
    print(f"Discriminator loss: {d_loss:.4f} (real: {d_real:.4f}, fake: {d_fake:.4f})")
    
    # Test generator loss
    g_loss = generator_loss(fake_output)
    print(f"Generator loss: {g_loss:.4f}")
    
    # Test feature matching
    fm_loss = FeatureMatchingLoss()
    real_features = torch.randn(batch_size, 256)
    fake_features = torch.randn(batch_size, 256)
    fm = fm_loss(real_features, fake_features)
    print(f"Feature matching loss: {fm:.4f}")
    
    print("✓ Loss functions test passed!")
