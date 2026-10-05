"""
Main script for PathMNIST classification (Part 1).
Students should implement the TODO sections to achieve >99% accuracy.
"""

import sys
import os
import csv
from datetime import datetime

from classification_dataset import create_pathmnist_dataloaders
from classification_models import get_model, count_parameters
from classification_train import train_model


def save_experiment_to_csv(
    model,
    model_name,
    history,
    args,
    batch_size,
    max_steps_per_epoch,
    csv_path="results/classification_experiments.csv"
):
    """
    Save one row per epoch for each experiment.
    """

    # Create results folder if necessary
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Unique identifier for this training run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Exact PyTorch architecture
    architecture = str(model).replace("\n", " | ")

    # Number of trainable parameters
    num_parameters = count_parameters(model)

    # Count actual layers
    num_layers = sum(
        1 for module in model.modules()
        if len(list(module.children())) == 0
    )

    fieldnames = [
        "run_id",
        "model_name",
        "architecture",
        "num_layers",
        "num_parameters",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "epochs_requested",
        "max_steps_per_epoch",
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "best_val_accuracy"
    ]

    file_exists = os.path.exists(csv_path)

    best_val_accuracy = max(history["val_acc"])

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for epoch in range(len(history["train_loss"])):
            writer.writerow({
                "run_id": run_id,
                "model_name": model_name,
                "architecture": architecture,
                "num_layers": num_layers,
                "num_parameters": num_parameters,
                "batch_size": batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs_requested": args.num_epochs,
                "max_steps_per_epoch": max_steps_per_epoch,
                "epoch": epoch + 1,
                "train_loss": history["train_loss"][epoch],
                "train_accuracy": history["train_acc"][epoch],
                "val_loss": history["val_loss"][epoch],
                "val_accuracy": history["val_acc"][epoch],
                "best_val_accuracy": best_val_accuracy
            })

    print(f"Experiment saved to {csv_path}")
    print(f"Run ID: {run_id}")

def main(args):
    print("=== CPH 100A Project 2 - Part 1: PathMNIST Classification ===")
    
    # Load data
    print("Loading PathMNIST dataset...")
    batch_size = 32
    max_steps_per_epoch = 100

    train_loader, val_loader, num_classes = create_pathmnist_dataloaders(
        batch_size=batch_size,
        num_workers=0,
        data_root='./data'
    )

    save_experiment_to_csv(
        model=model,
        model_name=args.model_name,
        history=history,
        args=args,
        batch_size=batch_size,
        max_steps_per_epoch=max_steps_per_epoch
    )
    
    print(f"\nCreating {args.model_name} model...")
    model = get_model(args.model_name, num_classes=num_classes)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    print(f"\nTraining {args.model_name} model...")
    try:
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_steps_per_epoch=max_steps_per_epoch  # Fast exploration mode. #TODO: Change for your full runs
        )
        
        print("Training completed successfully!")
        
        # Show final results
        if history['val_acc']:
            best_val_acc = max(history['val_acc'])
            print(f"Best validation accuracy: {best_val_acc:.4f}")
        
    except NotImplementedError as e:
        print(f"❌ Training failed: {e}")
        return


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CPH 100A Project 2 - PathMNIST Classification')
    parser.add_argument('--model_name', type=str, default='mlp',
                       choices=['mlp', 'cnn'], #TODO: add your models names here
                       help='Model to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of epochs to train')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay for regularization')
    args = parser.parse_args()
    
    main(args) 