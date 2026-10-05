"""
Simple training utilities for PathMNIST classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs=20, learning_rate=0.001, weight_decay=0.0, max_steps_per_epoch=100):
    """
    Simple training function for PathMNIST models.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for L2 regularization
        max_steps_per_epoch: Maximum training steps per epoch (for fast exploration)
    
    Returns:
        dict: Training history with losses and accuracies
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss() 
    
    # Track training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, max_steps_per_epoch)
        
        # Validation phase  
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        

    return history

def train_epoch(model, train_loader, optimizer, criterion, device, max_steps=None):
    """
    Train for one epoch.
    
    Args:
        max_steps: Maximum number of training steps (for fast exploration)
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    num_batch=0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        # Check if we've reached max steps
        if max_steps is not None and batch_idx >= max_steps:
            break
            
        data, target = data.to(device), target.to(device)
        
        # Flatten target if needed
        if target.dim() > 1:
            target = target.squeeze()
        
        # Zero gradients
        optimizer.zero_grad()
        
        #TODO: Implement the forward pass
        output = model(data)
        #TODO: Compute the loss
        loss = criterion(output, target)  # TODO: Compute the loss

        loss.backward()
        optimizer.step()
        
        # Track statistics
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        num_batch += 1
    
    avg_loss = total_loss / num_batch if num_batch > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):
    """
    Validate for one epoch.
    
    Args:
        model: The model to validate.
        val_loader: DataLoader for the validation set.
        criterion: Loss function.
        device: Device to run the validation on.
    
    Returns:
        tuple: (average_loss, accuracy)
    """

    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validation"):
            data, target = data.to(device), target.to(device)
            
            # Flatten target if needed
            if target.dim() > 1:
                target = target.squeeze()
            
            # Forward pass
            #TODO: Implement the forward pass
            output = model(data)
            #TODO: Compute the loss
            loss = criterion(output, target)  # TODO: Compute the loss
            
            # Track statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy