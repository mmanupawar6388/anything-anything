"""
Generate sample images using the interrupted checkpoint to show improved results.
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Generator
from src.utils.visualization import save_image_grid
import yaml

print("🎨 Generating samples from interrupted checkpoint...")

# Load config
with open('config/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load checkpoint
checkpoint_path = 'checkpoints/interrupted.pth'
print(f"Loading: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')
print(f"Checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")

# Create generator
generator = Generator(
    latent_dim=config['model']['latent_dim'],
    channels=config['model']['generator']['channels']
)

generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

# Generate samples
print("Generating 64 samples...")
with torch.no_grad():
    z = torch.randn(64, config['model']['latent_dim'])
    samples = generator(z)
    samples = (samples + 1) / 2

# Save
output_path = 'outputs/interrupted_checkpoint_samples.png'
Path('outputs').mkdir(exist_ok=True)
save_image_grid(samples, output_path, nrow=8)

print(f"✅ Samples saved to: {output_path}")
print("\nThese should be clearer than random noise!")
print("For even better results, wait for the current training to complete.")
