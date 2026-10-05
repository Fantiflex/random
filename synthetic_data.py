"""
Synthetic colored box border data generation for segmentation tasks.
Adds 1-pixel thick colored borders to PathMNIST images.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import random
from medmnist import PathMNIST
import matplotlib.pyplot as plt

class SegmentationDataGenerator:
    """
    Generate random colored box borders for segmentation tasks.
    Creates 1-pixel thick randomly-colored borders around rectangles.
    Segmentation target covers the entire interior area within the border.
    """
    
    def __init__(self, image_size=28, min_size=8, max_size=14):
        self.image_size = image_size
        self.min_size = min_size
        self.max_size = max_size
    
    def generate_box(self):
        """Generate a random colored box border with mask."""
        width = random.randint(self.min_size, self.max_size)
        height = random.randint(self.min_size, self.max_size)
        
        # Ensure box fits in image
        x = random.randint(0, self.image_size - width)
        y = random.randint(0, self.image_size - height)
        
        return {
            'x': x,
            'y': y,
            'width': width,
            'height': height
        }
    
    def add_box_to_image(self, image, box_info):
        """Add a random-colored border (1-pixel thick) to an image but mask covers the entire interior."""
        # Convert tensor to numpy if needed
        if torch.is_tensor(image):
            img_array = image.clone()
        else:
            img_array = torch.tensor(image)
        
        x, y = box_info['x'], box_info['y']
        width, height = box_info['width'], box_info['height']
        border_thickness = 1
        
        # Generate random border color (normalized to [-1, 1] range like the images)
        # Use strong colors that contrast well with medical images
        border_color = torch.rand(3) * 2 - 1  # Random color in [-1, 1] range
        # Make it more extreme for better visibility
        border_color = torch.sign(border_color) * (0.7 + 0.3 * torch.rand(3))
        
        # Store color info for potential debugging
        box_info['border_color'] = border_color
        
        # Create binary segmentation mask (for the ENTIRE INTERIOR, not just border)
        mask = torch.zeros((1, self.image_size, self.image_size))
        
        # Draw 1-pixel thick colored border around the rectangle (visual cue)
        # Top border
        top_end = min(y + border_thickness, y + height, self.image_size)
        img_array[:, y:top_end, x:x+width] = border_color.view(3, 1, 1)
        
        # Bottom border
        bottom_start = max(y + height - border_thickness, y)
        img_array[:, bottom_start:y+height, x:x+width] = border_color.view(3, 1, 1)
        
        # Left border
        left_end = min(x + border_thickness, x + width, self.image_size)
        img_array[:, y:y+height, x:left_end] = border_color.view(3, 1, 1)
        
        # Right border
        right_start = max(x + width - border_thickness, x)
        img_array[:, y:y+height, right_start:x+width] = border_color.view(3, 1, 1)
        
        # Mask covers the ENTIRE INTERIOR of the box (harder segmentation task!)
        # The model must predict the full area from just the border visual cue
        end_x = min(x + width, self.image_size)
        end_y = min(y + height, self.image_size)
        mask[:, y:end_y, x:end_x] = 1.0
        
        return img_array, mask

class PathMNISTBoxDataset(Dataset):
    """
    PathMNIST dataset with colored box borders for segmentation.
    Every image will have a randomly colored box added for segmentation training.
    """
    
    def __init__(self, split='train', root='./data', download=True, 
                 transform=None, add_boxes=True):
        """
        Initialize dataset with black box borders.
        Every image will have a randomly colored box for segmentation.
        
        Args:
            split: 'train', 'val', or 'test'
            root: Data root directory
            download: Whether to download PathMNIST
            transform: Image transforms to apply
            add_boxes: Whether to add black box borders (always True for this task)
        """
        # Load original PathMNIST dataset
        self.pathmnist = PathMNIST(
            split=split,
            download=download,
            root=root,
            transform=None  # We'll apply transforms manually
        )
        
        self.transform = transform
        self.add_boxes = add_boxes
        
        # Initialize black box generator
        self.box_generator = SegmentationDataGenerator()
        
        print(f"Box PathMNIST {split} dataset loaded: {len(self.pathmnist)} samples")
        if add_boxes:
            print(f"Adding randomly colored boxes to all images")
    
    def __len__(self):
        return len(self.pathmnist)
    
    def __getitem__(self, idx):
        # Get original PathMNIST sample
        image, label = self.pathmnist[idx]
        
        # Convert to tensor if needed
        if not torch.is_tensor(image):
            image = transforms.ToTensor()(image)
        
        # Always add a black box for segmentation task
        if self.add_boxes:
            box_info = self.box_generator.generate_box()
            image, mask = self.box_generator.add_box_to_image(image, box_info)
        else:
            # Empty mask if boxes disabled (shouldn't happen in practice)
            mask = torch.zeros((1, 28, 28))
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
            # Note: We don't transform the mask to keep it binary
        
        return {
            'image': image,
            'mask': mask,
            'pathmnist_label': label.item() if torch.is_tensor(label) else label,
            'has_box': torch.tensor(1.0, dtype=torch.float32)  # Always has box
        }

def create_box_dataloaders(batch_size=32, num_workers=0, data_root='./data'):
    """
    Create data loaders for colored box border segmentation task.
    Every image will have a randomly colored box for segmentation.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes
        data_root: Directory to store data
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    print("Creating colored box border segmentation dataloaders...")
    
    # Define transforms (simple normalization)
    train_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    train_dataset = PathMNISTBoxDataset(
        split='train',
        root=data_root,
        download=True,
        transform=train_transform,
        add_boxes=True
    )
    
    val_dataset = PathMNISTBoxDataset(
        split='val',
        root=data_root,
        download=True,
        transform=test_transform,
        add_boxes=True
    )
    
    test_dataset = PathMNISTBoxDataset(
        split='test',
        root=data_root,
        download=True,
        transform=test_transform,
        add_boxes=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=box_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=box_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=box_collate_fn
    )
    
    num_classes = 9  # PathMNIST has 9 classes
    
    return train_loader, val_loader, test_loader, num_classes

def box_collate_fn(batch):
    """Custom collate function for black box dataset."""
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    pathmnist_labels = torch.tensor([item['pathmnist_label'] for item in batch])
    has_boxes = torch.stack([item['has_box'] for item in batch])
    
    return {
        'image': images,
        'mask': masks,
        'pathmnist_label': pathmnist_labels,
        'has_box': has_boxes
    }

def visualize_box_samples(dataset, num_samples=6, save_path=None):
    """
    Visualize samples from black box dataset.
    
    Args:
        dataset: PathMNISTBoxDataset
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(3, num_samples//3, figsize=(15, 9))
    if num_samples <= 3:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        row = i // (num_samples//3)
        col = i % (num_samples//3)
        
        sample = dataset[i]
        image = sample['image']
        mask = sample['mask']
        has_box = sample['has_box']
        pathmnist_label = sample['pathmnist_label']
        
        # Convert tensor to numpy for visualization
        if image.shape[0] == 3:  # RGB
            img_np = image.permute(1, 2, 0).numpy()
        else:  # Grayscale
            img_np = image.squeeze().numpy()
        
        # Denormalize for visualization
        img_np = (img_np + 1) / 2  # From [-1, 1] to [0, 1]
        
        # Create overlay visualization
        mask_np = mask.squeeze().numpy()
        
        # Show image with red overlay for mask
        axes[row, col].imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)
        
        # Overlay mask in red
        if has_box > 0:
            masked_areas = np.ma.masked_where(mask_np == 0, mask_np)
            axes[row, col].imshow(masked_areas, alpha=0.5, cmap='Reds')
            title = f"Label: {pathmnist_label}, Box: Yes"
        else:
            title = f"Label: {pathmnist_label}, Box: No"
        
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

# Test the black box data generation
if __name__ == "__main__":
    print("Testing black box data generation...")
    
    # Create dataset
    dataset = PathMNISTBoxDataset(
        split='train',
        add_boxes=True,
        box_prob=1.0  # Always add boxes for testing
    )
    
    # Test sample generation
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Has box: {sample['has_box']}")
    print(f"PathMNIST label: {sample['pathmnist_label']}")
    
    # Visualize samples
    print("Generating visualization...")
    visualize_box_samples(dataset, num_samples=6, save_path='visualizations/box_samples.png')
    
    # Test dataloader
    train_loader, val_loader, test_loader, num_classes = create_box_dataloaders(
        batch_size=4, box_prob=0.8
    )
    
    # Test batch loading
    for batch in train_loader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Images shape: {batch['image'].shape}")
        print(f"Masks shape: {batch['mask'].shape}")
        print(f"Has boxes shape: {batch['has_box'].shape}")
        print(f"PathMNIST labels shape: {batch['pathmnist_label'].shape}")
        break
    
    print("Black box data generation test successful!") 