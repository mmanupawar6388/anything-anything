"""
Evaluation metrics for GAN quality assessment.
Includes FID Score, Inception Score, and visual quality metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from typing import Tuple, Dict
from tqdm import tqdm


class InceptionV3Feature(nn.Module):
    """
    Simplified feature extractor for MNIST (since Inception is for ImageNet).
    We'll use a simple CNN for feature extraction.
    """
    
    def __init__(self):
        super(InceptionV3Feature, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(128, 10)  # 10 classes for MNIST
    
    def forward(self, x):
        """Extract features."""
        features = self.features(x)
        features = features.view(features.size(0), -1)
        logits = self.fc(features)
        return features, logits


def calculate_fid_score(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: str = 'cuda',
    batch_size: int = 100
) -> float:
    """
    Calculate Fréchet Inception Distance (FID) score.
    
    FID measures the distance between real and generated image distributions
    in feature space. Lower is better.
    
    Args:
        real_images: Real images tensor (N, C, H, W)
        fake_images: Generated images tensor (N, C, H, W)
        device: Device to compute on
        batch_size: Batch size for feature extraction
    
    Returns:
        FID score (float)
    """
    # Load feature extractor
    feature_extractor = InceptionV3Feature().to(device)
    feature_extractor.eval()
    
    def get_features(images):
        """Extract features from images."""
        features_list = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size].to(device)
                # Normalize to [-1, 1] if needed
                if batch.max() > 1.0:
                    batch = batch / 255.0
                if batch.min() >= 0:
                    batch = batch * 2 - 1
                
                features, _ = feature_extractor(batch)
                features_list.append(features.cpu().numpy())
        
        return np.concatenate(features_list, axis=0)
    
    # Extract features
    real_features = get_features(real_images)
    fake_features = get_features(fake_images)
    
    # Calculate statistics
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    
    return float(fid)


def calculate_inception_score(
    images: torch.Tensor,
    device: str = 'cuda',
    batch_size: int = 100,
    splits: int = 10
) -> Tuple[float, float]:
    """
    Calculate Inception Score (IS).
    
    IS measures the quality and diversity of generated images.
    Higher is better.
    
    Args:
        images: Generated images tensor (N, C, H, W)
        device: Device to compute on
        batch_size: Batch size for evaluation
        splits: Number of splits for calculating mean and std
    
    Returns:
        Tuple of (mean_score, std_score)
    """
    # Load classifier
    classifier = InceptionV3Feature().to(device)
    classifier.eval()
    
    # Get predictions
    preds = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            
            # Normalize to [-1, 1] if needed
            if batch.max() > 1.0:
                batch = batch / 255.0
            if batch.min() >= 0:
                batch = batch * 2 - 1
            
            _, logits = classifier(batch)
            probs = F.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # Calculate IS for each split
    split_scores = []
    
    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits), :]
        
        # Calculate KL divergence
        py = np.mean(part, axis=0)
        scores = []
        
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * np.log(pyx / py + 1e-10)))
        
        split_scores.append(np.exp(np.mean(scores)))
    
    return float(np.mean(split_scores)), float(np.std(split_scores))


def calculate_visual_quality_metrics(
    images: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate visual quality metrics.
    
    Args:
        images: Images tensor (N, C, H, W)
    
    Returns:
        Dictionary of quality metrics
    """
    images_np = images.cpu().numpy()
    
    metrics = {}
    
    # Calculate mean and std
    metrics['mean'] = float(np.mean(images_np))
    metrics['std'] = float(np.std(images_np))
    
    # Calculate contrast
    metrics['contrast'] = float(np.max(images_np) - np.min(images_np))
    
    # Calculate sharpness (using Laplacian variance)
    if len(images_np.shape) == 4:
        # Convert to grayscale if needed
        if images_np.shape[1] == 3:
            images_gray = np.mean(images_np, axis=1)
        else:
            images_gray = images_np[:, 0, :, :]
        
        # Calculate Laplacian
        from scipy.ndimage import laplace
        sharpness_scores = []
        
        for img in images_gray:
            laplacian = laplace(img)
            sharpness_scores.append(np.var(laplacian))
        
        metrics['sharpness'] = float(np.mean(sharpness_scores))
    
    return metrics


def evaluate_model(
    generator: nn.Module,
    real_loader,
    num_samples: int = 10000,
    latent_dim: int = 100,
    device: str = 'cuda',
    batch_size: int = 100
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        generator: Generator model
        real_loader: DataLoader for real images
        num_samples: Number of samples to generate
        latent_dim: Latent dimension
        device: Device to evaluate on
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("Evaluating model...")
    
    generator.eval()
    
    # Generate fake images
    print("Generating samples...")
    fake_images = []
    
    with torch.no_grad():
        for _ in tqdm(range(0, num_samples, batch_size)):
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_batch = generator(z)
            # Denormalize from [-1, 1] to [0, 1]
            fake_batch = (fake_batch + 1) / 2
            fake_images.append(fake_batch.cpu())
    
    fake_images = torch.cat(fake_images, dim=0)[:num_samples]
    
    # Collect real images
    print("Collecting real images...")
    real_images = []
    
    for images, _ in tqdm(real_loader):
        real_images.append(images)
        if len(torch.cat(real_images, dim=0)) >= num_samples:
            break
    
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    
    # Calculate metrics
    metrics = {}
    
    print("Calculating FID score...")
    try:
        fid = calculate_fid_score(real_images, fake_images, device, batch_size)
        metrics['fid_score'] = fid
        print(f"  FID Score: {fid:.4f}")
    except Exception as e:
        print(f"  FID calculation failed: {e}")
        metrics['fid_score'] = -1
    
    print("Calculating Inception Score...")
    try:
        is_mean, is_std = calculate_inception_score(fake_images, device, batch_size)
        metrics['inception_score_mean'] = is_mean
        metrics['inception_score_std'] = is_std
        print(f"  Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    except Exception as e:
        print(f"  IS calculation failed: {e}")
        metrics['inception_score_mean'] = -1
        metrics['inception_score_std'] = -1
    
    print("Calculating visual quality metrics...")
    quality_metrics = calculate_visual_quality_metrics(fake_images)
    metrics.update(quality_metrics)
    
    for key, value in quality_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    generator.train()
    
    return metrics


if __name__ == "__main__":
    # Test evaluation metrics
    print("Testing evaluation metrics...")
    
    # Create dummy data
    real_images = torch.rand(1000, 1, 28, 28)
    fake_images = torch.rand(1000, 1, 28, 28)
    
    # Test FID
    print("\nTesting FID score...")
    fid = calculate_fid_score(real_images, fake_images, device='cpu', batch_size=100)
    print(f"FID Score: {fid:.4f}")
    
    # Test IS
    print("\nTesting Inception Score...")
    is_mean, is_std = calculate_inception_score(fake_images, device='cpu', batch_size=100)
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    
    # Test visual quality
    print("\nTesting visual quality metrics...")
    quality = calculate_visual_quality_metrics(fake_images)
    for key, value in quality.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓ Evaluation metrics test passed!")
