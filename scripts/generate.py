"""
Image generation script for MNIST GAN.
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Generator
from src.utils.visualization import save_image_grid, visualize_latent_space


def main():
    parser = argparse.ArgumentParser(description='Generate images with MNIST GAN')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--num_images', type=int, default=64,
                        help='Number of images to generate')
    parser.add_argument('--output', type=str, default='outputs/generated.png',
                        help='Output path for generated images')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--nrow', type=int, default=8,
                        help='Number of images per row in grid')
    parser.add_argument('--interpolate', action='store_true',
                        help='Generate latent space interpolation')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Random seed set to {args.seed}")
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create generator
    print("\nCreating generator...")
    generator = Generator(
        latent_dim=config['model']['latent_dim'],
        channels=config['model']['generator']['channels']
    )
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.to(device)
    generator.eval()
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate images
    if args.interpolate:
        print(f"\nGenerating latent space interpolation...")
        visualize_latent_space(
            generator=generator,
            latent_dim=config['model']['latent_dim'],
            device=device,
            num_samples=args.num_images,
            output_path=str(output_path)
        )
    else:
        print(f"\nGenerating {args.num_images} images...")
        
        with torch.no_grad():
            # Generate random latent vectors
            z = torch.randn(args.num_images, config['model']['latent_dim'], device=device)
            
            # Generate images
            generated = generator(z)
            
            # Denormalize from [-1, 1] to [0, 1]
            generated = (generated + 1) / 2
        
        # Save grid
        save_image_grid(
            images=generated,
            output_path=str(output_path),
            nrow=args.nrow,
            normalize=False
        )
    
    print(f"\n✓ Images saved to {output_path}")
    
    # Generate additional visualizations
    if not args.interpolate:
        # Save individual images
        individual_dir = output_path.parent / 'individual'
        individual_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving individual images to {individual_dir}...")
        
        import torchvision.utils as vutils
        from PIL import Image
        
        for i in range(min(10, args.num_images)):
            img_tensor = generated[i]
            img_array = (img_tensor[0].cpu().numpy() * 255).astype('uint8')
            img = Image.fromarray(img_array, mode='L')
            img.save(individual_dir / f'image_{i:03d}.png')
        
        print(f"✓ Saved {min(10, args.num_images)} individual images")


if __name__ == "__main__":
    main()
