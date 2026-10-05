"""
Simple evaluation metrics for medical image classification.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score

def calculate_accuracy(predictions, targets):
    """
    Calculate accuracy for classification.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
    
    Returns:
        accuracy: Float accuracy score
    """
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    
    # Convert probabilities to predictions if needed
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    
    return accuracy_score(targets, predictions) 