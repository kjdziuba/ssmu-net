"""
Loss functions for SSMU-Net training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation"""
    
    def __init__(self, smooth: float = 1e-6, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (B, C, H, W)
            targets: Ground truth labels (B, H, W)
        
        Returns:
            Dice loss value
        """
        B, C, H, W = logits.shape
        
        # Create mask for valid pixels
        valid_mask = (targets != self.ignore_index)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        targets_valid = targets.clone()
        targets_valid[~valid_mask] = 0  # Set ignored pixels to class 0 temporarily
        targets_oh = F.one_hot(targets_valid.long(), num_classes=C)  # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Apply valid mask to both predictions and targets
        valid_mask = valid_mask.unsqueeze(1).expand(-1, C, -1, -1)
        probs = probs * valid_mask
        targets_oh = targets_oh * valid_mask
        
        # Compute Dice coefficient per class
        dice_per_class = []
        for c in range(C):
            pred_c = probs[:, c]
            target_c = targets_oh[:, c]
            
            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
            
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice)
        
        # Average across classes and batch
        dice_score = torch.stack(dice_per_class, dim=1).mean()
        
        # Return loss (1 - dice)
        return 1.0 - dice_score


class FocalCrossEntropyLoss(nn.Module):
    """Focal loss variant of cross-entropy for class imbalance"""
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, 
                 ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (B, C, H, W)
            targets: Ground truth labels (B, H, W)
        
        Returns:
            Focal loss value
        """
        # Compute cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none', 
                                  ignore_index=self.ignore_index)
        
        # Get probabilities for true class
        p = F.softmax(logits, dim=1)
        targets_valid = targets.clone()
        targets_valid[targets == self.ignore_index] = 0
        
        # Gather probabilities for true classes
        p_true = p.gather(1, targets_valid.unsqueeze(1).long()).squeeze(1)
        
        # Apply focal term
        focal_weight = (1 - p_true) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_weight = self.alpha[targets_valid.long()]
            alpha_weight[targets == self.ignore_index] = 0
            focal_loss = alpha_weight * focal_loss
        
        # Average over valid pixels
        valid_mask = (targets != self.ignore_index)
        if valid_mask.sum() > 0:
            return focal_loss[valid_mask].mean()
        else:
            return focal_loss.sum() * 0  # Return 0 if no valid pixels


class SparsityLoss(nn.Module):
    """L1 sparsity penalty on filter bandwidths to encourage narrow bands"""
    
    def __init__(self, alpha: float = 1e-4):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, f_low: torch.Tensor, f_high: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_low: Lower cutoff frequencies (F,)
            f_high: Upper cutoff frequencies (F,)
        
        Returns:
            Sparsity penalty
        """
        bandwidths = f_high - f_low
        return self.alpha * bandwidths.mean()


class OverlapLoss(nn.Module):
    """Penalty for overlapping filters to encourage spectral diversity"""
    
    def __init__(self, beta: float = 1e-4, min_separation: float = 5.0):
        super().__init__()
        self.beta = beta
        self.min_separation = min_separation
    
    def forward(self, f_low: torch.Tensor, f_high: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_low: Lower cutoff frequencies (F,)
            f_high: Upper cutoff frequencies (F,)
        
        Returns:
            Overlap penalty
        """
        n_filters = f_low.shape[0]
        overlap_penalty = f_low.new_tensor(0.0)
        
        # Compute pairwise overlaps
        for i in range(n_filters):
            for j in range(i + 1, n_filters):
                # Check if filters i and j overlap
                overlap_start = torch.maximum(f_low[i], f_low[j])
                overlap_end = torch.minimum(f_high[i], f_high[j])
                overlap_width = torch.relu(overlap_end - overlap_start)
                
                # Penalize overlap
                overlap_penalty = overlap_penalty + overlap_width
                
                # Also penalize filters that are too close (even if not overlapping)
                gap = torch.minimum(
                    torch.abs(f_low[i] - f_high[j]),
                    torch.abs(f_low[j] - f_high[i])
                )
                proximity_penalty = torch.relu(self.min_separation - gap)
                overlap_penalty = overlap_penalty + proximity_penalty
        
        # Normalize by number of pairs
        n_pairs = n_filters * (n_filters - 1) / 2
        return self.beta * overlap_penalty / n_pairs


class CombinedSegmentationLoss(nn.Module):
    """Combined loss for segmentation: CE + Dice + Focal"""
    
    def __init__(self, 
                 ce_weight: float = 1.0,
                 dice_weight: float = 1.0,
                 focal_weight: float = 0.0,
                 class_weights: Optional[torch.Tensor] = None,
                 ignore_index: int = -100):
        super().__init__()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # Initialize individual losses
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, 
                                              ignore_index=ignore_index)
        else:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        
        if focal_weight > 0:
            self.focal_loss = FocalCrossEntropyLoss(alpha=class_weights, 
                                                   ignore_index=ignore_index)
        else:
            self.focal_loss = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: Model predictions (B, C, H, W)
            targets: Ground truth labels (B, H, W)
        
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        # Cross-entropy loss
        if self.ce_weight > 0:
            ce = self.ce_loss(logits, targets)
            losses['ce'] = ce
        else:
            ce = 0
        
        # Dice loss
        if self.dice_weight > 0:
            dice = self.dice_loss(logits, targets)
            losses['dice'] = dice
        else:
            dice = 0
        
        # Focal loss
        if self.focal_weight > 0 and self.focal_loss is not None:
            focal = self.focal_loss(logits, targets)
            losses['focal'] = focal
        else:
            focal = 0
        
        # Combined loss
        total = (self.ce_weight * ce + 
                self.dice_weight * dice + 
                self.focal_weight * focal)
        
        losses['seg_total'] = total
        
        return losses


class SSMUNetLoss(nn.Module):
    """Complete loss function for SSMU-Net training"""
    
    def __init__(self, cfg: Dict):
        super().__init__()
        
        loss_cfg = cfg['loss']
        
        # Segmentation losses
        seg_cfg = loss_cfg['seg']
        ce_weight = 1.0 if seg_cfg.get('ce', True) else 0.0
        dice_weight = 1.0 if seg_cfg.get('dice', True) else 0.0
        focal_weight = seg_cfg.get('focal', 0.0)
        
        # Get class weights if specified
        class_weights = None
        if seg_cfg.get('class_weights') == 'inverse_freq':
            # Will be set later with actual data statistics
            class_weights = None
        
        self.seg_loss = CombinedSegmentationLoss(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            class_weights=class_weights,
            ignore_index=-100
        )
        
        # Sparsity losses
        sparsity_cfg = loss_cfg.get('sparsity', {})
        self.sparsity_alpha = sparsity_cfg.get('alpha', 0.0)
        self.overlap_beta = sparsity_cfg.get('beta', 0.0)
        self.ramp_epochs = sparsity_cfg.get('ramp_epochs', 10)
        
        if self.sparsity_alpha > 0:
            self.sparsity_loss = SparsityLoss(alpha=self.sparsity_alpha)
        else:
            self.sparsity_loss = None
        
        if self.overlap_beta > 0:
            self.overlap_loss = OverlapLoss(beta=self.overlap_beta)
        else:
            self.overlap_loss = None
        
        self.current_epoch = 0
    
    def set_class_weights(self, weights: torch.Tensor):
        """Update class weights for weighted CE loss"""
        self.seg_loss.ce_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        if self.seg_loss.focal_loss is not None:
            self.seg_loss.focal_loss.alpha = weights
    
    def set_epoch(self, epoch: int):
        """Update current epoch for loss scheduling"""
        self.current_epoch = epoch
    
    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor,
                cutoffs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: Model predictions (B, C, H, W)
            targets: Ground truth labels (B, H, W)
            cutoffs: Optional tuple of (f_low, f_high) for regularization
        
        Returns:
            Dictionary with all loss components and total loss
        """
        losses = {}
        
        # Segmentation losses
        seg_losses = self.seg_loss(logits, targets)
        losses.update(seg_losses)
        total = seg_losses['seg_total']
        
        # Sparsity and overlap losses (with ramp-up)
        if cutoffs is not None and (self.sparsity_loss is not None or self.overlap_loss is not None):
            f_low, f_high = cutoffs
            
            # Ramp-up factor for regularization
            if self.current_epoch < self.ramp_epochs:
                ramp_factor = self.current_epoch / self.ramp_epochs
            else:
                ramp_factor = 1.0
            
            # Sparsity loss
            if self.sparsity_loss is not None:
                sparsity = self.sparsity_loss(f_low, f_high) * ramp_factor
                losses['sparsity'] = sparsity
                total = total + sparsity
            
            # Overlap loss
            if self.overlap_loss is not None:
                overlap = self.overlap_loss(f_low, f_high) * ramp_factor
                losses['overlap'] = overlap
                total = total + overlap
        
        losses['total'] = total
        
        return losses


def create_loss(cfg: Dict) -> nn.Module:
    """Factory function to create loss module"""
    return SSMUNetLoss(cfg)


if __name__ == "__main__":
    # Test losses
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent / "ssmu_net" / "config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Create loss
    loss_fn = create_loss(cfg)
    
    # Test inputs
    B, C, H, W = 2, 8, 128, 128
    logits = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))
    targets[0, :10, :10] = -100  # Add some ignored pixels
    
    # Test cutoffs
    n_filters = 32
    f_low = torch.linspace(900, 1700, n_filters)
    f_high = f_low + 50
    
    # Compute losses
    losses = loss_fn(logits, targets, (f_low, f_high))
    
    print("Loss components:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")