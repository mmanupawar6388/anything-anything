"""
Visualization utilities for GAN training and evaluation.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import torchvision.utils as vutils


def save_image_grid(
    images: torch.Tensor,
    output_path: str,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
    padding: int = 2
):
    """
    Save a grid of images to file.
    
    Args:
        images: Tensor of images (N, C, H, W)
        output_path: Path to save the image
        nrow: Number of images per row
        normalize: Whether to normalize images
        value_range: Range of values (min, max) for normalization
        padding: Padding between images
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create grid
    grid = vutils.make_grid(
        images,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
        padding=padding
    )
    
    # Convert to numpy and transpose
    grid_np = grid.cpu().numpy().transpose((1, 2, 0))
    
    # Handle grayscale
    if grid_np.shape[2] == 1:
        grid_np = grid_np.squeeze(2)
        cmap = 'gray'
    else:
        cmap = None
    
    # Save image
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np, cmap=cmap)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(
    metrics: dict,
    output_path: str,
    title: str = "Training Curves"
):
    """
    Plot training curves from metrics.
    
    Args:
        metrics: Dictionary of metric lists
        output_path: Path to save the plot
        title: Plot title
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 4))
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, metrics.items()):
        ax.plot(values)
        ax.set_title(name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_latent_space(
    generator: torch.nn.Module,
    latent_dim: int,
    device: str = 'cuda',
    num_samples: int = 100,
    output_path: str = 'latent_space.png'
):
    """
    Visualize the latent space by interpolating between random points.
    
    Args:
        generator: Generator model
        latent_dim: Dimension of latent space
        device: Device to run on
        num_samples: Number of samples to generate
        output_path: Path to save visualization
    """
    generator.eval()
    
    with torch.no_grad():
        # Generate random latent vectors
        z1 = torch.randn(1, latent_dim, device=device)
        z2 = torch.randn(1, latent_dim, device=device)
        
        # Interpolate
        alphas = torch.linspace(0, 1, num_samples, device=device)
        interpolated = []
        
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            img = generator(z)
            interpolated.append(img)
        
        # Stack and save
        interpolated = torch.cat(interpolated, dim=0)
        interpolated = (interpolated + 1) / 2  # Denormalize
        
        save_image_grid(interpolated, output_path, nrow=10)
    
    generator.train()


def compare_real_fake(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    output_path: str,
    num_samples: int = 32
):
    """
    Create a comparison grid of real and fake images.
    
    Args:
        real_images: Real images tensor
        fake_images: Fake images tensor
        output_path: Path to save comparison
        num_samples: Number of samples to show
    """
    # Select samples
    real_samples = real_images[:num_samples]
    fake_samples = fake_images[:num_samples]
    
    # Denormalize if needed
    if real_samples.min() < 0:
        real_samples = (real_samples + 1) / 2
    if fake_samples.min() < 0:
        fake_samples = (fake_samples + 1) / 2
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(15, 6))
    
    # Real images
    real_grid = vutils.make_grid(real_samples, nrow=8, normalize=True, padding=2)
    axes[0].imshow(real_grid.cpu().numpy().transpose((1, 2, 0)).squeeze(), cmap='gray')
    axes[0].set_title('Real Images', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Fake images
    fake_grid = vutils.make_grid(fake_samples, nrow=8, normalize=True, padding=2)
    axes[1].imshow(fake_grid.cpu().numpy().transpose((1, 2, 0)).squeeze(), cmap='gray')
    axes[1].set_title('Generated Images', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_curves(
    g_losses: List[float],
    d_losses: List[float],
    output_path: str
):
    """
    Plot generator and discriminator loss curves.
    
    Args:
        g_losses: List of generator losses
        d_losses: List of discriminator losses
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 5))
    
    plt.plot(g_losses, label='Generator Loss', alpha=0.7)
    plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")
    
    # Create dummy images
    images = torch.randn(64, 1, 28, 28)
    images = (images - images.min()) / (images.max() - images.min())
    
    # Test save_image_grid
    save_image_grid(images, 'test_grid.png', nrow=8)
    print("✓ Image grid saved")
    
    # Test plot_training_curves
    metrics = {
        'Generator Loss': np.random.rand(100),
        'Discriminator Loss': np.random.rand(100)
    }
    plot_training_curves(metrics, 'test_curves.png')
    print("✓ Training curves plotted")
    
    print("\n✓ Visualization utilities test passed!")
