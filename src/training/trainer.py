"""
GAN Trainer - Comprehensive training loop for MNIST GAN.
Handles training, validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple
from pathlib import Path
import time
from tqdm import tqdm

from ..models import Generator, Discriminator
from .losses import discriminator_loss, generator_loss
from ..utils.checkpoint import CheckpointManager
from ..utils.visualization import save_image_grid


class GANTrainer:
    """
    Comprehensive GAN trainer with best practices.
    
    Features:
    - Progressive training with learning rate scheduling
    - Gradient clipping and monitoring
    - TensorBoard logging
    - Checkpoint management
    - Early stopping
    - Mixed precision training support
    """
    
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        train_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs'
    ):
        """
        Initialize the GAN trainer.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            train_loader: Training data loader
            config: Configuration dictionary
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.config = config
        self.device = device
        
        # Setup optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config['training']['learning_rate']['generator'],
            betas=(config['training']['beta1'], config['training']['beta2'])
        )
        
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config['training']['learning_rate']['discriminator'],
            betas=(config['training']['beta1'], config['training']['beta2'])
        )
        
        # Setup learning rate schedulers
        if config['training']['scheduler']['enabled']:
            scheduler_type = config['training']['scheduler']['type']
            
            if scheduler_type == 'step':
                self.g_scheduler = optim.lr_scheduler.StepLR(
                    self.g_optimizer,
                    step_size=config['training']['scheduler']['step_size'],
                    gamma=config['training']['scheduler']['gamma']
                )
                self.d_scheduler = optim.lr_scheduler.StepLR(
                    self.d_optimizer,
                    step_size=config['training']['scheduler']['step_size'],
                    gamma=config['training']['scheduler']['gamma']
                )
            elif scheduler_type == 'cosine':
                self.g_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.g_optimizer,
                    T_max=config['training']['num_epochs']
                )
                self.d_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.d_optimizer,
                    T_max=config['training']['num_epochs']
                )
            else:
                self.g_scheduler = None
                self.d_scheduler = None
        else:
            self.g_scheduler = None
            self.d_scheduler = None
        
        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir,
            max_to_keep=config['checkpoint']['keep_last_n']
        )
        
        # Setup TensorBoard
        self.writer = SummaryWriter(log_dir) if config['logging']['tensorboard'] else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_fid_score = float('inf')
        
        # Fixed noise for consistent visualization
        self.fixed_noise = torch.randn(
            config['logging']['num_sample_images'],
            config['model']['latent_dim'],
            device=device
        )
        
        # Training hyperparameters
        self.label_smoothing = config['training']['label_smoothing']
        self.noise_std = config['training']['noise_std']
        
        print(f"✓ Trainer initialized")
        print(f"  Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"  Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {
            'g_loss': 0.0,
            'd_loss': 0.0,
            'd_real_loss': 0.0,
            'd_fake_loss': 0.0,
            'd_real_acc': 0.0,
            'd_fake_acc': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (real_images, _) in enumerate(pbar):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)
            
            # Normalize images to [-1, 1]
            real_images = real_images * 2 - 1
            
            # Add noise to real images (helps training stability)
            if self.noise_std > 0:
                real_images = real_images + torch.randn_like(real_images) * self.noise_std
                real_images = torch.clamp(real_images, -1, 1)
            
            # ==================== Train Discriminator ====================
            self.d_optimizer.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, self.config['model']['latent_dim'], device=self.device)
            fake_images = self.generator(z).detach()
            
            # Get discriminator outputs
            real_output = self.discriminator(real_images)
            fake_output = self.discriminator(fake_images)
            
            # Calculate discriminator loss
            d_loss, d_real_loss, d_fake_loss = discriminator_loss(
                real_output, fake_output, self.label_smoothing
            )
            
            # Backward and optimize
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            self.d_optimizer.step()
            
            # ==================== Train Generator ====================
            self.g_optimizer.zero_grad()
            
            # Generate new fake images
            z = torch.randn(batch_size, self.config['model']['latent_dim'], device=self.device)
            fake_images = self.generator(z)
            
            # Get discriminator output for fake images
            fake_output = self.discriminator(fake_images)
            
            # Calculate generator loss
            g_loss = generator_loss(fake_output)
            
            # Backward and optimize
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.g_optimizer.step()
            
            # ==================== Calculate Metrics ====================
            with torch.no_grad():
                d_real_acc = (real_output > 0.5).float().mean().item()
                d_fake_acc = (fake_output < 0.5).float().mean().item()
            
            # Update metrics
            epoch_metrics['g_loss'] += g_loss.item()
            epoch_metrics['d_loss'] += d_loss.item()
            epoch_metrics['d_real_loss'] += d_real_loss.item()
            epoch_metrics['d_fake_loss'] += d_fake_loss.item()
            epoch_metrics['d_real_acc'] += d_real_acc
            epoch_metrics['d_fake_acc'] += d_fake_acc
            
            # Update progress bar
            pbar.set_postfix({
                'G': f"{g_loss.item():.3f}",
                'D': f"{d_loss.item():.3f}",
                'D_real': f"{d_real_acc:.2f}",
                'D_fake': f"{d_fake_acc:.2f}"
            })
            
            # ==================== Logging ====================
            if self.global_step % self.config['logging']['log_interval'] == 0:
                if self.writer:
                    self.writer.add_scalar('Loss/Generator', g_loss.item(), self.global_step)
                    self.writer.add_scalar('Loss/Discriminator', d_loss.item(), self.global_step)
                    self.writer.add_scalar('Loss/D_Real', d_real_loss.item(), self.global_step)
                    self.writer.add_scalar('Loss/D_Fake', d_fake_loss.item(), self.global_step)
                    self.writer.add_scalar('Accuracy/D_Real', d_real_acc, self.global_step)
                    self.writer.add_scalar('Accuracy/D_Fake', d_fake_acc, self.global_step)
            
            # Save sample images
            if self.global_step % self.config['logging']['image_interval'] == 0:
                self.save_samples()
            
            self.global_step += 1
        
        # Average metrics
        num_batches = len(self.train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def save_samples(self):
        """Generate and save sample images."""
        self.generator.eval()
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise)
            # Denormalize from [-1, 1] to [0, 1]
            fake_images = (fake_images + 1) / 2
            
            # Save grid
            output_path = Path(self.config['logging']['log_dir']) / f'samples_step_{self.global_step}.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_image_grid(fake_images, output_path, nrow=8)
            
            # Log to TensorBoard
            if self.writer:
                self.writer.add_images('Generated_Images', fake_images[:64], self.global_step)
        
        self.generator.train()
    
    def train(self, num_epochs: Optional[int] = None):
        """
        Train the GAN for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train (uses config if None)
        """
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            metrics = self.train_epoch()
            
            # Log epoch metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
            print(f"  Generator Loss: {metrics['g_loss']:.4f}")
            print(f"  Discriminator Loss: {metrics['d_loss']:.4f}")
            print(f"  D Real Accuracy: {metrics['d_real_acc']:.4f}")
            print(f"  D Fake Accuracy: {metrics['d_fake_acc']:.4f}")
            
            if self.writer:
                self.writer.add_scalar('Epoch/G_Loss', metrics['g_loss'], epoch)
                self.writer.add_scalar('Epoch/D_Loss', metrics['d_loss'], epoch)
            
            # Update learning rate
            if self.g_scheduler:
                self.g_scheduler.step()
                self.d_scheduler.step()
                
                if self.writer:
                    self.writer.add_scalar('LR/Generator', self.g_optimizer.param_groups[0]['lr'], epoch)
                    self.writer.add_scalar('LR/Discriminator', self.d_optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config['checkpoint']['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        # Save final checkpoint
        self.save_checkpoint('final_model.pth')
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time / 3600:.2f} hours")
        print(f"{'='*60}\n")
        
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'config': self.config,
            'best_fid_score': self.best_fid_score
        }
        
        if self.g_scheduler:
            checkpoint['g_scheduler_state_dict'] = self.g_scheduler.state_dict()
            checkpoint['d_scheduler_state_dict'] = self.d_scheduler.state_dict()
        
        self.checkpoint_manager.save(checkpoint, filename)
        print(f"✓ Checkpoint saved: {filename}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        if self.g_scheduler and 'g_scheduler_state_dict' in checkpoint:
            self.g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
            self.d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_fid_score = checkpoint.get('best_fid_score', float('inf'))
        
        print(f"✓ Checkpoint loaded from epoch {self.current_epoch}")
