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
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels * 28 * 28)
        )
        self.out_channels = out_channels
    
    def forward(self, x):
        x = self.network(x)
        x = x.view(x.size(0), self.out_channels, 28, 28)
        return x

class TinyUNet(nn.Module):
    """
    Tiny U-Net for segmentation of 28x28 images.
    Uses skip connections between encoder and decoder.
    """

    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super(TinyUNet, self).__init__()

        # Encoder block 1: 28x28
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(2)  # 28 -> 14

        # Encoder block 2: 14x14
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(2)  # 14 -> 7

        # Bottleneck: 7x7
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder stage 1: 7 -> 14
        self.up1 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(
                base_channels * 4,   # concatenation: 32 + 32
                base_channels * 2,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                base_channels * 2,
                base_channels * 2,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

        # Decoder stage 2: 14 -> 28
        self.up2 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(
                base_channels * 2,   # concatenation: 16 + 16
                base_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

        # Final pixel-wise prediction
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # [B, 16, 28, 28]
        p1 = self.pool1(e1)        # [B, 16, 14, 14]

        e2 = self.enc2(p1)         # [B, 32, 14, 14]
        p2 = self.pool2(e2)        # [B, 32, 7, 7]

        # Bottleneck
        b = self.bottleneck(p2)    # [B, 64, 7, 7]

        # Decoder stage 1
        d1 = self.up1(b)           # [B, 32, 14, 14]

        # Skip connection from encoder block 2
        d1 = torch.cat([d1, e2], dim=1)   # [B, 64, 14, 14]
        d1 = self.dec1(d1)                 # [B, 32, 14, 14]

        # Decoder stage 2
        d2 = self.up2(d1)          # [B, 16, 28, 28]

        # Skip connection from encoder block 1
        d2 = torch.cat([d2, e1], dim=1)   # [B, 32, 28, 28]
        d2 = self.dec2(d2)                 # [B, 16, 28, 28]

        # Final segmentation mask
        return self.output(d2)     # [B, 1, 28, 28]



    
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
        loss = 1 - (2 * (pred * target).sum(dim=(1, 2, 3)) + self.smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.smooth)  # TODO: Compute the DICE loss
        return loss.mean()

class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice loss for better segmentation performance.
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCELoss()  # Initialize the BCE loss
        self.dice_loss = DiceLoss()  # Initialize the DICE loss
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice
