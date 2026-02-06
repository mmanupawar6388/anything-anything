"""
Evaluation script for MNIST GAN.
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Generator
from src.evaluation import evaluate_model
from src.utils import get_mnist_loader


def main():
    parser = argparse.ArgumentParser(description='Evaluate MNIST GAN')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to generate for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
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
    
    # Load real data
    print("\nLoading MNIST dataset...")
    real_loader = get_mnist_loader(
        data_dir=config['data']['data_dir'],
        batch_size=config['evaluation']['batch_size'],
        num_workers=config['data']['num_workers'],
        shuffle=False,
        download=True,
        train=False
    )
    
    # Evaluate model
    print("\n" + "="*60)
    print("Starting Evaluation")
    print("="*60 + "\n")
    
    metrics = evaluate_model(
        generator=generator,
        real_loader=real_loader,
        num_samples=args.num_samples,
        latent_dim=config['model']['latent_dim'],
        device=device,
        batch_size=config['evaluation']['batch_size']
    )
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'checkpoint': args.checkpoint,
        'epoch': checkpoint.get('epoch', 'unknown'),
        'num_samples': args.num_samples,
        'metrics': metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
