"""
Main script for PathMNIST box segmentation (Part 2).
U-Net segmentation of colored boxes on PathMNIST images.
"""

import sys
import os
import torch

from synthetic_data import create_box_dataloaders, visualize_box_samples, PathMNISTBoxDataset
from segmentation_models import get_segmentation_model, count_parameters
from segmentation_train import train_segmentation_model, evaluate_segmentation_model, visualize_predictions

def main(args):
    print("=== CPH 100A Project 2 - Part 2: Box Segmentation ===")
    
    
    # Create box datasets
    print("Creating box segmentation datasets...")
    
    # Set seed for reproducible dataset generation
    import torch
    torch.manual_seed(42)
    
    train_loader, val_loader, test_loader, num_classes = create_box_dataloaders(
        batch_size=32,
        num_workers=0,
        data_root='./data'
    )
    
    # Debug: Check dataset characteristics
    print("\nDataset Analysis:")
    val_total = sum(len(batch['has_box']) for batch in val_loader)
    print(f"Validation: {val_total} samples (all have boxes)")
    
    test_total = sum(len(batch['has_box']) for batch in test_loader)
    print(f"Test: {test_total} samples (all have boxes)")
    

    
    # Visualize some samples
    print("\nVisualizing box samples...")
    sample_dataset = PathMNISTBoxDataset(
        split='train',
        add_boxes=True
    )
    visualize_box_samples(sample_dataset, num_samples=6, save_path='visualizations/box_samples.png')
    
    # Train the specified model
    print(f"\n{'='*60}")
    print(f"Training {args.model_name.upper()} Segmentation Model")
    print(f"{'='*60}")
    
    # Create model
    model = get_segmentation_model(args.model_name, in_channels=3, out_channels=1)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Train model
    print(f"\nTraining {args.model_name} model...")
    history = train_segmentation_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps_per_epoch=100  # Fast exploration mode
    )
    
    # Evaluate model on validation set
    print(f"\nEvaluating {args.model_name} model...")
    val_results = evaluate_segmentation_model(model, val_loader, f"{args.model_name.upper()} Validation")
    
    # Show results
    val_iou = val_results['mean_iou']
    print(f"\n📊 {args.model_name.upper()} Results:")
    print(f"   Final Validation IoU: {val_iou:.4f}")
    
    # Visualize predictions
    print(f"\nVisualizing {args.model_name} predictions...")
    visualize_predictions(
        model, sample_dataset, 
        num_samples=6, 
        save_path=f'visualizations/{args.model_name}_predictions.png'
    )
    
    return {'model': model, 'val_results': val_results}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PathMNIST Black Box Segmentation')
    parser.add_argument('--model_name', type=str, default='unet',
                       choices=['mlp', 'unet'], #TODO: add your models names here
                       help='Segmentation model to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of epochs to train')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay for regularization')
    args = parser.parse_args()
    
    
    # Just create and visualize the dataset
    print("Creating and visualizing box dataset...")
    dataset = PathMNISTBoxDataset(
        split='train',
        add_boxes=True
    )
    visualize_box_samples(dataset, num_samples=6, save_path='visualizations/box_samples.png')
    print("Visualization complete!")
    
    results = main(args)