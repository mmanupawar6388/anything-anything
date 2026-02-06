"""
Generator Network for MNIST GAN.
Implements a Deep Convolutional Generator based on DCGAN architecture.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class Generator(nn.Module):
    """
    Generator network that transforms random noise into MNIST-like images.
    
    Architecture:
        - Input: Latent vector (batch_size, latent_dim)
        - Multiple transposed convolution layers with batch normalization
        - Output: Generated image (batch_size, 1, 28, 28)
    
    Args:
        latent_dim: Dimension of the input latent vector
        channels: List of channel sizes for each layer
        kernel_size: Size of convolutional kernels
        stride: Stride for transposed convolutions
        padding: Padding for transposed convolutions
        use_batch_norm: Whether to use batch normalization
        activation: Activation function ('relu' or 'leaky_relu')
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        channels: Optional[List[int]] = None,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.channels = channels or [256, 128, 64, 1]
        self.use_batch_norm = use_batch_norm
        
        # Build the generator network
        layers = []
        
        # Initial projection: latent_dim -> 256 * 7 * 7
        # This creates a 7x7 feature map with 256 channels
        layers.append(nn.Linear(latent_dim, self.channels[0] * 7 * 7))
        layers.append(nn.BatchNorm1d(self.channels[0] * 7 * 7))
        layers.append(nn.ReLU(True))
        
        self.projection = nn.Sequential(*layers)
        
        # Transposed convolution layers
        conv_layers = []
        
        # 7x7 -> 14x14 (256 -> 128 channels)
        conv_layers.extend(self._make_layer(
            self.channels[0], self.channels[1],
            kernel_size, stride, padding, use_batch_norm, activation
        ))
        
        # 14x14 -> 28x28 (128 -> 64 channels)
        conv_layers.extend(self._make_layer(
            self.channels[1], self.channels[2],
            kernel_size, stride, padding, use_batch_norm, activation
        ))
        
        # Final layer: 28x28 (64 -> 1 channel)
        conv_layers.append(nn.Conv2d(
            self.channels[2], self.channels[3],
            kernel_size=3, stride=1, padding=1
        ))
        conv_layers.append(nn.Tanh())  # Output in range [-1, 1]
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_batch_norm: bool,
        activation: str
    ) -> List[nn.Module]:
        """Create a transposed convolution layer with optional batch norm and activation."""
        layers = []
        
        layers.append(nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size, stride, padding, bias=not use_batch_norm
        ))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation == 'relu':
            layers.append(nn.ReLU(True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, True))
        
        return layers
    
    def _initialize_weights(self):
        """Initialize network weights using best practices."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
        
        Returns:
            Generated images of shape (batch_size, 1, 28, 28)
        """
        # Project and reshape
        x = self.projection(z)
        x = x.view(-1, self.channels[0], 7, 7)
        
        # Apply transposed convolutions
        x = self.conv_layers(x)
        
        return x
    
    def generate(self, num_samples: int, device: str = 'cuda') -> torch.Tensor:
        """
        Generate random samples.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
        
        Returns:
            Generated images of shape (num_samples, 1, 28, 28)
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.forward(z)
        return samples


if __name__ == "__main__":
    # Test the generator
    print("Testing Generator...")
    
    # Create generator
    gen = Generator(latent_dim=100)
    print(f"Generator parameters: {sum(p.numel() for p in gen.parameters()):,}")
    
    # Test forward pass
    batch_size = 16
    z = torch.randn(batch_size, 100)
    output = gen(z)
    print(f"Input shape: {z.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test generation
    samples = gen.generate(10, device='cpu')
    print(f"Generated samples shape: {samples.shape}")
    
    print("✓ Generator test passed!")
