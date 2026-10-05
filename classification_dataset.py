"""
Simple PathMNIST data loading for classification.
Provides basic data loaders without synthetic modifications.
"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from medmnist import PathMNIST

def create_pathmnist_dataloaders(batch_size=32, num_workers=0, data_root='./data'):
    """
    Create basic PathMNIST data loaders for classification.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        data_root: Root directory for data storage
    
    Returns:
        tuple: (train_loader, val_loader, num_classes)
    """
    print("Creating PathMNIST classification dataloaders...")
    
    # Define transforms for PathMNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        #TODO: Hint, you may want to add "Data Augmentation" here to improve performance.
    ])
    
    # Create datasets (train / val only)
    train_dataset = PathMNIST(
        split='train',
        download=True,
        root=data_root,
        transform=transform
    )
    
    val_dataset = PathMNIST(
        split='val',
        download=True,
        root=data_root,
        transform=transform
    )
    
    print(f"PathMNIST train dataset: {len(train_dataset)} samples")
    print(f"PathMNIST val dataset: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    num_classes = 9  # PathMNIST has 9 classes

    return train_loader, val_loader, num_classes

# Test the data loading
if __name__ == "__main__":
    print("Testing PathMNIST data loading...")
    
    train_loader, val_loader, num_classes = create_pathmnist_dataloaders(
        batch_size=4
    )
    
    print(f"Number of classes: {num_classes}")
    
    # Test batch loading
    for data, target in train_loader:
        print(f"Batch shape: {data.shape}, Labels shape: {target.shape}")
        print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"Sample labels: {target[:4]}")
        break
    
    print("PathMNIST data loading test successful!") 