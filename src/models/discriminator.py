"""
Discriminator Network for MNIST GAN.
Implements a Deep Convolutional Discriminator based on DCGAN architecture.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class Discriminator(nn.Module):
    """
    Discriminator network that classifies images as real or fake.
    
    Architecture:
        - Input: Image (batch_size, 1, 28, 28)
        - Multiple convolutional layers with batch normalization and dropout
        - Output: Probability of being real (batch_size, 1)
    
    Args:
        channels: List of channel sizes for each layer
        kernel_size: Size of convolutional kernels
        stride: Stride for convolutions
        padding: Padding for convolutions
        use_batch_norm: Whether to use batch normalization
        activation: Activation function ('relu' or 'leaky_relu')
        leaky_slope: Slope for LeakyReLU activation
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        channels: Optional[List[int]] = None,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_batch_norm: bool = True,
        activation: str = 'leaky_relu',
        leaky_slope: float = 0.2,
        dropout: float = 0.3
    ):
        super(Discriminator, self).__init__()
        
        self.channels = channels or [1, 64, 128, 256]
        self.use_batch_norm = use_batch_norm
        self.leaky_slope = leaky_slope
        self.dropout = dropout
        
        # Build the discriminator network
        layers = []
        
        # First layer: 28x28 -> 14x14 (1 -> 64 channels)
        # No batch norm in first layer (DCGAN best practice)
        layers.append(nn.Conv2d(
            self.channels[0], self.channels[1],
            kernel_size, stride, padding, bias=True
        ))
        layers.append(nn.LeakyReLU(leaky_slope, True))
        layers.append(nn.Dropout2d(dropout))
        
        # Second layer: 14x14 -> 7x7 (64 -> 128 channels)
        layers.extend(self._make_layer(
            self.channels[1], self.channels[2],
            kernel_size, stride, padding, use_batch_norm, activation
        ))
        
        # Third layer: 7x7 -> 3x3 (128 -> 256 channels)
        layers.extend(self._make_layer(
            self.channels[2], self.channels[3],
            kernel_size, stride, padding, use_batch_norm, activation
        ))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate the size after convolutions
        # After 3 conv layers with stride 2: 28 -> 14 -> 7 -> 3
        self.feature_size = self.channels[3] * 3 * 3
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 1),
            nn.Sigmoid()  # Output probability
        )
        
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
        """Create a convolutional layer with optional batch norm and activation."""
        layers = []
        
        layers.append(nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding, bias=not use_batch_norm
        ))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation == 'relu':
            layers.append(nn.ReLU(True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(self.leaky_slope, True))
        
        layers.append(nn.Dropout2d(self.dropout))
        
        return layers
    
    def _initialize_weights(self):
        """Initialize network weights using best practices."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        
        Args:
            x: Input images of shape (batch_size, 1, 28, 28)
        
        Returns:
            Probability of being real, shape (batch_size, 1)
        """
        # Apply convolutional layers
        features = self.conv_layers(x)
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the discriminator (useful for FID score).
        
        Args:
            x: Input images of shape (batch_size, 1, 28, 28)
        
        Returns:
            Feature vectors of shape (batch_size, feature_size)
        """
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        return features


if __name__ == "__main__":
    # Test the discriminator
    print("Testing Discriminator...")
    
    # Create discriminator
    disc = Discriminator()
    print(f"Discriminator parameters: {sum(p.numel() for p in disc.parameters()):,}")
    
    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, 1, 28, 28)
    output = disc(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test feature extraction
    features = disc.get_features(x)
    print(f"Features shape: {features.shape}")
    
    print("✓ Discriminator test passed!")
