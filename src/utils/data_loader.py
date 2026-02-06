"""
Data loading utilities for MNIST dataset.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional


def get_mnist_loader(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    download: bool = True,
    train: bool = True
) -> DataLoader:
    """
    Create MNIST data loader with appropriate transforms.
    
    Args:
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle: Whether to shuffle the data
        download: Whether to download the dataset if not present
        train: Whether to load training or test set
    
    Returns:
        DataLoader for MNIST dataset
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1]
        # Note: We normalize to [-1, 1] in the training loop
    ])
    
    # Load MNIST dataset
    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        transform=transform,
        download=download
    )
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )
    
    print(f"✓ MNIST {'train' if train else 'test'} loader created")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(loader)}")
    
    return loader


def get_data_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Get both train and test data loaders.
    
    Args:
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size for training
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        download: Whether to download the dataset
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_loader = get_mnist_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        download=download,
        train=True
    )
    
    test_loader = get_mnist_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        download=download,
        train=False
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test data loader
    print("Testing MNIST data loader...")
    
    loader = get_mnist_loader(batch_size=64, num_workers=0)
    
    # Get a batch
    images, labels = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Unique labels: {labels.unique().tolist()}")
    
    print("\n✓ Data loader test passed!")
