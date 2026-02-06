"""
Training package initialization.
Exports Trainer and loss functions.
"""

from .trainer import GANTrainer
from .losses import generator_loss, discriminator_loss

__all__ = ['GANTrainer', 'generator_loss', 'discriminator_loss']
