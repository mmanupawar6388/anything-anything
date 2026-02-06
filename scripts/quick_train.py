"""
Quick training script for fast results.
Trains for just a few epochs to get identifiable digits quickly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from src.models import Generator, Discriminator
from src.training import GANTrainer
from src.utils import get_mnist_loader
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

print("🚀 Quick Training Mode - Fast Results!")
print("="*60)

# Set seed
set_seed(42)

# Load config
with open('config/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Quick training settings
config['training']['num_epochs'] = 5
config['training']['batch_size'] = 256  # Larger batches
config['logging']['log_interval'] = 50
config['logging']['image_interval'] = 200

device = 'cpu'
print(f"Device: {device}")
print(f"Epochs: {config['training']['num_epochs']}")
print(f"Batch size: {config['training']['batch_size']}")

# Load data
print("\n📦 Loading MNIST...")
train_loader = get_mnist_loader(
    batch_size=config['training']['batch_size'],
    num_workers=0,  # Faster on CPU
    download=True
)

# Create models
print("\n🏗️ Creating models...")
generator = Generator(
    latent_dim=config['model']['latent_dim'],
    channels=config['model']['generator']['channels']
)

discriminator = Discriminator(
    channels=config['model']['discriminator']['channels'],
    dropout=config['model']['discriminator']['dropout']
)

# Create trainer
print("\n🎯 Starting training...")
trainer = GANTrainer(
    generator=generator,
    discriminator=discriminator,
    train_loader=train_loader,
    config=config,
    device=device,
    checkpoint_dir='checkpoints',
    log_dir='logs'
)

# Train
try:
    trainer.train()
    print("\n✅ Training complete!")
    print("🎨 Generating sample images...")
    
    # Generate samples
    from src.utils.visualization import save_image_grid
    
    generator.eval()
    with torch.no_grad():
        z = torch.randn(64, config['model']['latent_dim'], device=device)
        samples = generator(z)
        samples = (samples + 1) / 2
        
        save_image_grid(samples, 'outputs/quick_train_samples.png', nrow=8)
    
    print("✅ Samples saved to: outputs/quick_train_samples.png")
    print("\n🔄 Restart Flask to use the trained model:")
    print("   1. Stop Flask (Ctrl+C)")
    print("   2. Run: python app.py")
    
except KeyboardInterrupt:
    print("\n⚠️ Training interrupted")
    trainer.save_checkpoint('quick_train_interrupted.pth')
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
