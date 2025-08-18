#!/usr/bin/env python3
"""
Simple training script for benchmark models.
Uses proven settings from successful pixel_pixel experiments.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssmu_net.data import NpzCoreDataset
from ssmu_net.bench_models import get_benchmark_model
from ssmu_net.losses import DiceLoss


def compute_class_weights(train_dataset, num_classes=8):
    """Compute class weights from training data (ignoring background)"""
    print("Computing class weights...")
    class_counts = torch.zeros(num_classes)
    
    # Sample a subset of data to compute weights
    for i in range(min(100, len(train_dataset))):
        sample = train_dataset[i]
        if isinstance(sample, dict):
            y = sample['y']
        else:
            y = sample[1] if isinstance(sample, tuple) else sample
        for c in range(num_classes):
            class_counts[c] += (y == c).sum()
    
    # Compute inverse frequency weights (ignore class 0 - background)
    total = class_counts[1:].sum()  # Exclude background
    weights = torch.ones(num_classes)
    weights[0] = 0  # Background weight = 0 (will be ignored)
    
    for c in range(1, num_classes):
        if class_counts[c] > 0:
            weights[c] = total / (num_classes - 1) / class_counts[c]
    
    # Normalize weights
    weights[1:] = weights[1:] / weights[1:].mean()
    
    print("Class weights:", weights.tolist())
    return weights


def train_epoch(model, loader, criterion_ce, criterion_dice, optimizer, device, ignore_index=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_ce = 0
    total_dice = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        if isinstance(batch, dict):
            X = batch['X'].to(device)
            y = batch['y'].to(device)
        else:
            X, y = batch[:2]  # Handle tuple returns
            X = X.to(device)
            y = y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(X)
        
        # Combined loss (like successful pixel_pixel)
        ce_loss = criterion_ce(logits, y)
        dice_loss = criterion_dice(logits, y)
        loss = 0.5 * ce_loss + 0.5 * dice_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_dice += dice_loss.item()
        
        pbar.set_postfix({
            'loss': loss.item(),
            'ce': ce_loss.item(),
            'dice': dice_loss.item()
        })
    
    n = len(loader)
    return total_loss / n, total_ce / n, total_dice / n


def validate(model, loader, criterion_ce, criterion_dice, device, num_classes=8, ignore_index=0):
    """Validate model and compute metrics on full cores"""
    model.eval()
    total_loss = 0
    total_ce = 0
    total_dice = 0
    
    # For mIoU calculation
    confusion_matrix = torch.zeros(num_classes, num_classes)
    valid_samples = 0  # Count samples with valid losses
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for batch in pbar:
            if isinstance(batch, dict):
                X = batch['X'].to(device)
                y = batch['y'].to(device)
            else:
                X, y = batch[:2]
                X = X.to(device)
                y = y.to(device)
            
            # Handle variable sizes by padding to multiple of 32
            B, C, H, W = X.shape
            # Calculate padded dimensions (multiple of 32 for U-Net)
            pad_h = (32 - H % 32) % 32
            pad_w = (32 - W % 32) % 32
            
            if pad_h > 0 or pad_w > 0:
                # Pad input
                X_padded = F.pad(X, (0, pad_w, 0, pad_h), mode='constant', value=0)
                # Forward pass on padded input
                logits = model(X_padded)
                # Crop predictions back to original size
                logits = logits[:, :, :H, :W]
            else:
                # No padding needed
                logits = model(X)
            
            # Skip if this core has no valid pixels (all background)
            if (y != ignore_index).sum() == 0:
                continue
            
            # Losses
            ce_loss = criterion_ce(logits, y)
            dice_loss = criterion_dice(logits, y)
            loss = 0.5 * ce_loss + 0.5 * dice_loss
            
            # Skip if loss is NaN
            if torch.isnan(loss):
                continue
            
            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_dice += dice_loss.item()
            valid_samples += 1
            
            # Predictions for metrics
            preds = logits.argmax(dim=1)
            
            # Update confusion matrix (only for non-background pixels)
            valid_mask = y != ignore_index
            for c_true in range(num_classes):
                for c_pred in range(num_classes):
                    if c_true == ignore_index or c_pred == ignore_index:
                        continue
                    # Ensure all tensors are on CPU for confusion matrix update
                    confusion_matrix[c_true, c_pred] += ((y == c_true) & (preds == c_pred) & valid_mask).sum().cpu().item()
    
    # Use valid_samples for averaging (cores with annotations)
    if valid_samples > 0:
        avg_loss = total_loss / valid_samples
        avg_ce = total_ce / valid_samples
        avg_dice = total_dice / valid_samples
    else:
        # No valid samples
        avg_loss = avg_ce = avg_dice = float('nan')
    
    # Compute mIoU (excluding background)
    iou_per_class = []
    for c in range(1, num_classes):  # Skip background (class 0)
        intersection = confusion_matrix[c, c]
        union = confusion_matrix[c, :].sum() + confusion_matrix[:, c].sum() - intersection
        if union > 0:
            iou = intersection / union
            iou_per_class.append(iou.item())
    
    miou = np.mean(iou_per_class) if iou_per_class else 0.0
    
    return avg_loss, avg_ce, avg_dice, miou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='ssmu_net/config_benchmark.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                       choices=['spectral_attention', 'se_unet', 'simple'],
                       help='Model architecture to use (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    # Auto-detect best device (MPS for Mac, CUDA for GPU, CPU fallback)
    if torch.backends.mps.is_available():
        default_device = 'mps'
    elif torch.cuda.is_available():
        default_device = 'cuda'
    else:
        default_device = 'cpu'
    
    parser.add_argument('--device', type=str, default=default_device,
                       help='Device to use (mps/cuda/cpu)')
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override config with command line args
    if args.model:
        cfg['benchmark']['model_name'] = args.model
    if args.epochs:
        cfg['optim']['epochs'] = args.epochs
    if args.batch_size:
        cfg['data']['batch_size'] = args.batch_size
    if args.lr:
        cfg['optim']['lr'] = args.lr
    
    # Extract values from config
    args.model = cfg['benchmark']['model_name']
    args.epochs = cfg['optim']['epochs']
    args.batch_size = cfg['data']['batch_size']
    args.lr = cfg['optim']['lr']
    args.patch_size = cfg['data']['patch_size']
    
    print(f"\n{'='*60}")
    print(f"Training Benchmark Model: {args.model}")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Patch size: {args.patch_size}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir = Path(f"outputs/benchmark/{args.model}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    npz_dir = cfg['runtime_paths']['npz']
    
    print("Loading datasets...")
    
    # Get NPZ files
    import glob
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "core_*.npz")))
    
    # Split into train/val (80/20)
    n_train = int(0.8 * len(npz_files))
    train_files = npz_files[:n_train]
    val_files = npz_files[n_train:]
    
    train_dataset = NpzCoreDataset(
        npz_paths=train_files,
        patch_size=cfg['data']['patch_size'],
        mode='train',
        augment=True,
        ignore_index=cfg['data']['ignore_index'],
        max_patches_per_core=cfg.get('sampling', {}).get('max_patches_per_core_train', 50)
    )
    
    val_dataset = NpzCoreDataset(
        npz_paths=val_files,
        patch_size=cfg['data']['patch_size'],
        mode='val',  # Keep val mode for full cores
        augment=False,
        ignore_index=cfg['data']['ignore_index'],
        max_patches_per_core=cfg.get('sampling', {}).get('max_patches_per_core_val', 10)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if args.device == 'mps' else 2,  # MPS doesn't work well with multiprocessing
        pin_memory=False if args.device == 'mps' else True  # MPS doesn't need pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # MUST be 1 for full cores with variable sizes
        shuffle=False,
        num_workers=0 if args.device == 'mps' else 2,  # MPS doesn't work well with multiprocessing
        pin_memory=False if args.device == 'mps' else True  # MPS doesn't need pin_memory
    )
    
    # Get number of channels from data
    sample = train_dataset[0]
    if isinstance(sample, dict):
        in_channels = sample['X'].shape[0]
    else:
        in_channels = sample[0].shape[0] if isinstance(sample, tuple) else sample.shape[0]
    print(f"Input channels: {in_channels}")
    
    # Create model
    model = get_benchmark_model(
        model_name=args.model,
        in_channels=in_channels,
        num_classes=8
    )
    model = model.to(args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Compute class weights
    class_weights = compute_class_weights(train_dataset)
    class_weights = class_weights.to(args.device)
    
    # Loss functions (matching successful pixel_pixel)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)
    criterion_dice = DiceLoss(ignore_index=0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'train_ce': [],
        'train_dice': [],
        'val_loss': [],
        'val_ce': [],
        'val_dice': [],
        'val_miou': [],
        'best_miou': 0,
        'best_epoch': 0
    }
    
    # Training loop
    best_miou = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_ce, train_dice = train_epoch(
            model, train_loader, criterion_ce, criterion_dice, 
            optimizer, args.device, ignore_index=0
        )
        
        # Validate
        val_loss, val_ce, val_dice, val_miou = validate(
            model, val_loader, criterion_ce, criterion_dice,
            args.device, num_classes=8, ignore_index=0
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_ce'].append(train_ce)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_ce'].append(val_ce)
        history['val_dice'].append(val_dice)
        history['val_miou'].append(val_miou)
        
        # Print metrics
        print(f"Train - Loss: {train_loss:.4f}, CE: {train_ce:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, CE: {val_ce:.4f}, Dice: {val_dice:.4f}, mIoU: {val_miou:.4f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch + 1
            history['best_miou'] = best_miou
            history['best_epoch'] = best_epoch
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'args': vars(args)
            }, output_dir / 'best_model.pth')
            
            print(f"✅ New best model! mIoU: {best_miou:.4f}")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'args': vars(args)
    }, output_dir / 'final_model.pth')
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best mIoU: {best_miou:.4f} at epoch {best_epoch}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()