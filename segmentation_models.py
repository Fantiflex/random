"""
Simple U-Net architecture for box segmentation.
Includes MLP baseline, CNN, and U-Net for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPSegmentation(nn.Module):
    """
    Simple MLP model for segmentation.
    Flattens input, applies multiple linear layers, reshapes to output mask.
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super(MLPSegmentation, self).__init__()
        
        # PathMNIST images are 3x28x28 = 2352 input features
        # Output should be 1x28x28 = 784 features
        
        # TODO: Add your own MLP architecture here
    
    def forward(self, x):
        raise NotImplementedError("MLPSegmentation is not implemented")

class TinyUNet(nn.Module):
    """
    Tiny U-Net for segmentation of 28x28 images.
    Optimized for small images and simple segmentation tasks.
    """
    
    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super(TinyUNet, self).__init__()
        
        # Encoder (contracting path)
        # TODO: Add your own encoder architecture here
    
    def forward(self, x):
        raise NotImplementedError("TinyUNet is not implemented")
        

def get_segmentation_model(model_name, in_channels=3, out_channels=1):
    """Get segmentation model by name."""
    if model_name == 'mlp':
        return MLPSegmentation(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'unet':
        return TinyUNet(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError("Unknown segmentation model: {}".format(model_name))

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Intersection over Union (IoU) metric for segmentation
def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """
    Calculate Intersection over Union for binary segmentation masks.
    
    Args:
        pred_mask: Predicted segmentation mask [B, 1, H, W] or [B, H, W]
        true_mask: Ground truth segmentation mask [B, 1, H, W] or [B, H, W]
        threshold: Threshold for binarizing predictions
    
    Returns:
        IoU score (float)
    """
    # Convert to binary
    if torch.is_tensor(pred_mask):
        pred_binary = (pred_mask > threshold).float()
    else:
        pred_binary = (pred_mask > threshold).astype(float)
    
    if torch.is_tensor(true_mask):
        true_binary = (true_mask > 0.5).float()
    else:
        true_binary = (true_mask > 0.5).astype(float)
    
    # Flatten for easier computation
    if len(pred_binary.shape) > 2:
        pred_binary = pred_binary.view(pred_binary.size(0), -1)
        true_binary = true_binary.view(true_binary.size(0), -1)
    
    # Calculate intersection and union
    intersection = (pred_binary * true_binary).sum(dim=-1)
    union = pred_binary.sum(dim=-1) + true_binary.sum(dim=-1) - intersection
    
    # Handle case where both masks are empty
    iou = intersection / (union + 1e-8)  # Add small epsilon to avoid division by zero
    
    return iou.mean().item() if torch.is_tensor(iou) else iou.mean()

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    Better than BCE for imbalanced segmentation.
    """
    
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        #TODO: Compute the DICE loss
        loss = 0 # TODO: Compute the DICE loss
        return loss 

class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice loss for better segmentation performance.
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = None # TODO: Initialize the BCE loss
        self.dice_loss = None # TODO: Initialize the DICE loss
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice
