"""
Model package initialization.
Exports Generator and Discriminator models.
"""

from .generator import Generator
from .discriminator import Discriminator

__all__ = ['Generator', 'Discriminator']
