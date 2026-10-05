"""
Training utilities for segmentation models.
Handles U-Net training for black box segmentation.
"""

import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from segmentation_models import CombinedLoss, calculate_iou

def train_segmentation_model(model, train_loader, val_loader, epochs=25, learning_rate=0.001, weight_decay=0.0, max_steps_per_epoch=100):
    """
    Train a segmentation model.
    
    Args:
        model: Segmentation model (U-Net or similar)
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for L2 regularization
        max_steps_per_epoch: Maximum training steps per epoch (for fast exploration)
    
    Returns:
        dict: Training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    criterion = None # TODO: Add your own loss function here (hint: Combined BCE + DICE loss is suggested)
    
    # Training history
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': []
    }
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        train_loss, train_iou = train_segmentation_epoch(model, train_loader, optimizer, criterion, device, max_steps_per_epoch)
        
        # Validation phase
        val_loss, val_iou = validate_segmentation_epoch(model, val_loader, criterion, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        # Print progress
        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")

    
    return history

def train_segmentation_epoch(model, train_loader, optimizer, criterion, device, max_steps=None):
    """Train for one epoch with segmentation loss."""
    model.train()
    
    total_loss = 0
    total_iou = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        # Check if we've reached max steps
        if max_steps is not None and batch_idx >= max_steps:
            break
            
        # Move data to device
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        #TODO: Implement the forward pass
        predictions = 0 # TODO: Compute the predictions

        #TODO: Compute the loss
        loss = 0 # TODO: Compute the loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate IoU for this batch
        with torch.no_grad():
            iou = calculate_iou(predictions, masks)
        
        # Track statistics
        total_loss += loss.item()
        total_iou += iou
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_iou = total_iou / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_iou

def validate_segmentation_epoch(model, val_loader, criterion, device):
    """Validate for one epoch with segmentation loss."""
    model.eval()
    
    total_loss = 0
    total_iou = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to device
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            #TODO: Implement the forward pass
            predictions = 0 # TODO: Compute the predictions

            #TODO: Compute the loss
            loss = 0 # TODO: Compute the loss
            
            # Calculate IoU for this batch
            iou = calculate_iou(predictions, masks)
            
            # Track statistics
            total_loss += loss.item()
            total_iou += iou
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_iou = total_iou / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_iou

def evaluate_segmentation_model(model, test_loader, dataset_name="Test"):
    """
    Evaluate segmentation model on test set.
    
    Returns detailed metrics for segmentation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_ious = []
    total_samples = 0
    samples_with_boxes = 0
    
    # For debugging: collect some statistics
    iou_buckets = {'very_low': 0, 'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {dataset_name}"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            has_boxes = batch['has_box']
            
            #TODO: Implement the forward pass
            predictions = 0 # TODO: Compute the predictions
            
            # Calculate IoU for each sample in batch (all samples have boxes)
            for i in range(images.size(0)):
                pred_mask = predictions[i:i+1]
                true_mask = masks[i:i+1]
                
                # Calculate IoU for this sample
                iou = calculate_iou(pred_mask, true_mask)
                all_ious.append(iou)
                samples_with_boxes += 1
                total_samples += 1
    
    # Calculate metrics
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    std_iou = np.std(all_ious) if all_ious else 0.0
    
    results = {
        'mean_iou': mean_iou,
        'std_iou': std_iou,
        'samples_with_boxes': samples_with_boxes,
        'total_samples': total_samples,
        'box_ratio': samples_with_boxes / total_samples if total_samples > 0 else 0,
        'iou_distribution': iou_buckets
    }
    
    print(f"\n{dataset_name} Results:")
    print(f"  Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"  Total samples: {total_samples} (all have boxes)")
    
    return results

def visualize_predictions(model, dataset, num_samples=6, save_path=None):
    """
    Visualize model predictions on sample images.
    """
    import matplotlib.pyplot as plt
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
    
    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]
            image = sample['image'].unsqueeze(0).to(device)
            true_mask = sample['mask']
            has_box = sample['has_box'].item()
            
            # Get prediction
            pred_mask = model(image).cpu().squeeze()
            
            # Convert image for visualization
            img_np = sample['image'].permute(1, 2, 0).numpy()
            img_np = (img_np + 1) / 2  # Denormalize
            
            # Original image
            axes[0, i].imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)
            axes[0, i].set_title(f"Original {i+1}")
            axes[0, i].axis('off')
            
            # True mask
            axes[1, i].imshow(true_mask.squeeze(), cmap='gray')
            if has_box > 0:
                axes[1, i].set_title(f"True Mask")
            else:
                axes[1, i].set_title(f"No Box")
            axes[1, i].axis('off')
            
            # Predicted mask
            axes[2, i].imshow(pred_mask, cmap='gray')
            if has_box > 0:
                iou = calculate_iou(pred_mask.unsqueeze(0).unsqueeze(0), 
                                  true_mask.unsqueeze(0))
                axes[2, i].set_title(f"Predicted (IoU: {iou:.3f})")
            else:
                axes[2, i].set_title(f"Predicted")
            axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")
    else:
        plt.show()