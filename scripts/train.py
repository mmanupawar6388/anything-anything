"""
Main training script for MNIST GAN.
"""

import argparse
import yaml
import torch
import random
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Generator, Discriminator
from src.training import GANTrainer
from src.utils import get_mnist_loader


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train MNIST GAN')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.device:
        config['system']['device'] = args.device
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    
    # Set device
    if config['system']['device'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config['system']['device']
    
    print(f"Using device: {device}")
    
    # Set random seed
    if config['system']['seed'] is not None:
        set_seed(config['system']['seed'])
        print(f"Random seed set to {config['system']['seed']}")
    
    # Create directories
    Path(config['checkpoint']['save_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['logging']['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['data']['data_dir']).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader = get_mnist_loader(
        data_dir=config['data']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        shuffle=config['data']['shuffle'],
        download=config['data']['download']
    )
    
    # Create models
    print("\nCreating models...")
    generator = Generator(
        latent_dim=config['model']['latent_dim'],
        channels=config['model']['generator']['channels'],
        kernel_size=config['model']['generator']['kernel_size'],
        stride=config['model']['generator']['stride'],
        padding=config['model']['generator']['padding'],
        use_batch_norm=config['model']['generator']['use_batch_norm'],
        activation=config['model']['generator']['activation']
    )
    
    discriminator = Discriminator(
        channels=config['model']['discriminator']['channels'],
        kernel_size=config['model']['discriminator']['kernel_size'],
        stride=config['model']['discriminator']['stride'],
        padding=config['model']['discriminator']['padding'],
        use_batch_norm=config['model']['discriminator']['use_batch_norm'],
        activation=config['model']['discriminator']['activation'],
        leaky_slope=config['model']['discriminator']['leaky_slope'],
        dropout=config['model']['discriminator']['dropout']
    )
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        config=config,
        device=device,
        checkpoint_dir=config['checkpoint']['save_dir'],
        log_dir=config['logging']['log_dir']
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    try:
        trainer.train(num_epochs=config['training']['num_epochs'])
        print("\n✓ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint('interrupted.pth')
        print("✓ Checkpoint saved")
    except Exception as e:
        print(f"\n\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nSaving checkpoint...")
        trainer.save_checkpoint('error.pth')
        print("✓ Checkpoint saved")
        raise


if __name__ == "__main__":
    main()
