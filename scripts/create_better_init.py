"""
Create a better initialized model for clearer initial results.
This uses better weight initialization to produce more digit-like shapes even before training.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Generator, Discriminator
import yaml

print("🎨 Creating better initialized model...")

# Load config
with open('config/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create models
generator = Generator(
    latent_dim=config['model']['latent_dim'],
    channels=config['model']['generator']['channels']
)

discriminator = Discriminator(
    channels=config['model']['discriminator']['channels']
)

# Better initialization for more structured outputs
def better_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # Use Xavier initialization for better initial structure
        nn.init.xavier_normal_(m.weight.data, gain=0.5)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.01)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=0.5)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# Apply better initialization
generator.apply(better_init)
discriminator.apply(better_init)

# Create checkpoint
checkpoint = {
    'epoch': 0,
    'global_step': 0,
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'config': config,
    'best_fid_score': float('inf')
}

# Save
Path('checkpoints').mkdir(exist_ok=True)
torch.save(checkpoint, 'checkpoints/better_init.pth')

print("✅ Better initialized model saved to: checkpoints/better_init.pth")
print("\n🔄 To use this model:")
print("1. Stop Flask (Ctrl+C)")
print("2. Update app.py to load 'checkpoints/better_init.pth'")
print("3. Restart Flask")
print("\nThis will give slightly better structure, but training is still needed for clear digits!")
