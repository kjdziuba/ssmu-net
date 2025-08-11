"""
Evaluation metrics for segmentation
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, 
                   num_classes: int = 8, ignore_index: int = -100,
                   exclude_background: bool = True) -> Dict[str, float]:
    """
    Compute segmentation metrics
    
    Args:
        preds: Predicted labels (B, H, W) or flattened
        targets: Ground truth labels (B, H, W) or flattened
        num_classes: Number of classes
        ignore_index: Label to ignore
        exclude_background: Whether to exclude class 0 from mean metrics
    
    Returns:
        Dictionary of metrics
    """
    # Flatten tensors safely
    preds = preds.detach().cpu().numpy().reshape(-1)
    targets = targets.detach().cpu().numpy().reshape(-1)
    
    # Filter out ignored pixels
    valid_mask = targets != ignore_index
    preds = preds[valid_mask]
    targets = targets[valid_mask]
    
    if len(targets) == 0:
        return {
            'miou': 0.0,
            'dice': 0.0,
            'micro_dice': 0.0,
            'accuracy': 0.0
        }
    
    # Compute confusion matrix
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    
    # Per-class IoU
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou = intersection / (union + 1e-10)
    
    # Per-class Dice
    dice_scores = 2 * intersection / (cm.sum(axis=1) + cm.sum(axis=0) + 1e-10)
    
    # Determine which classes to include in means
    present_classes = cm.sum(axis=1) > 0
    if exclude_background:
        present_classes[0] = False  # Exclude background class
    
    # Mean IoU (excluding background if specified)
    if present_classes.any():
        miou = iou[present_classes].mean()
        mean_dice = dice_scores[present_classes].mean()
    else:
        miou = 0.0
        mean_dice = 0.0
    
    # Overall accuracy
    accuracy = np.diag(cm).sum() / (cm.sum() + 1e-10)
    
    # Micro-Dice (computed from global confusion matrix)
    total_intersection = np.diag(cm).sum()
    total_sum = cm.sum()
    # micro_dice = 2 * total_intersection / (2 * total_sum - total_intersection + 1e-10)
    micro_dice = total_intersection / (total_sum + 1e-10)  # == accuracy

    
    metrics = {
        'miou': float(miou),
        'dice': float(mean_dice),  # This is the macro-averaged Dice
        'micro_dice': float(micro_dice),  # This is the micro-averaged Dice
        'accuracy': float(accuracy)
    }
    
    # Add per-class IoU and Dice
    for i in range(num_classes):
        metrics[f'iou_class_{i}'] = float(iou[i])
        metrics[f'dice_class_{i}'] = float(dice_scores[i])
    
    return metrics


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple batches or folds
    
    Args:
        metrics_list: List of metric dictionaries
    
    Returns:
        Aggregated metrics with mean and std
    """
    if not metrics_list:
        return {}
    
    aggregated = {}
    
    # Get all keys
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    
    # Compute mean and std for each metric
    for key in all_keys:
        values = [m.get(key, 0.0) for m in metrics_list]
        aggregated[f'{key}_mean'] = float(np.mean(values))
        aggregated[f'{key}_std'] = float(np.std(values))
    
    return aggregated


if __name__ == "__main__":
    # Test metrics
    B, H, W = 2, 128, 128
    num_classes = 8
    
    # Random predictions and targets
    preds = torch.randint(0, num_classes, (B, H, W))
    targets = torch.randint(0, num_classes, (B, H, W))
    
    # Add some ignored pixels
    targets[0, :10, :10] = -100
    
    # Compute metrics
    metrics = compute_metrics(preds, targets, num_classes)
    
    print("Metrics:")
    for key, value in metrics.items():
        if not key.startswith('iou_class') and not key.startswith('dice_class'):
            print(f"  {key}: {value:.4f}")