"""
Training script for SSMU-Net with cross-validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import json
import yaml
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import time
import shutil
from datetime import datetime

from .models import create_model
from .losses import create_loss
from .data import create_dataloaders, create_data_splits, compute_class_weights
from .utils import load_config, ensure_dirs, set_deterministic
from .evaluation_metrics import compute_metrics, aggregate_metrics


class Trainer:
    """SSMU-Net trainer with cross-validation support"""
    
    def __init__(self, cfg: Dict, fold: int = 0):
        """
        Args:
            cfg: Configuration dictionary
            fold: Current fold index for cross-validation
        """
        self.cfg = cfg
        self.fold = fold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.paths = ensure_dirs(cfg)
        self.model_dir = Path(self.paths['models']) / f'fold_{fold}'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_dir = Path(self.paths['logs']) / f'fold_{fold}'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir / 'tensorboard')
        
        # Set deterministic behavior after directories exist
        deterministic = cfg['optim'].get('deterministic', True)  # Default to True for backward compatibility
        set_deterministic(cfg['data']['seed'] + fold, str(self.log_dir), deterministic)
        
        # Training config
        self.epochs = cfg['optim']['epochs']
        self.lr = cfg['optim']['lr']
        self.weight_decay = cfg['optim']['weight_decay']
        self.grad_clip = cfg['optim']['grad_clip']
        self.amp_enabled = cfg['optim']['amp']
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_miou = 0.0
        self.best_epoch = -1
        self.patience = cfg['optim'].get('patience', 20)
        self.patience_counter = 0
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.scaler = GradScaler() if self.amp_enabled else None
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
    def setup_model_and_optimizer(self, class_weights: Optional[torch.Tensor] = None):
        """Initialize model, optimizer, and loss function"""
        
        # Create model
        self.model = create_model(self.cfg).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Create scheduler
        scheduler_type = self.cfg['optim']['scheduler']
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.epochs,
                eta_min=1e-6
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        else:
            self.scheduler = None
        
        # Create loss function
        self.loss_fn = create_loss(self.cfg)
        if class_weights is not None:
            self.loss_fn.set_class_weights(class_weights.to(self.device))
    
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.loss_fn.set_epoch(epoch)
        
        metrics = {
            'loss': 0.0,
            'ce': 0.0,
            'dice': 0.0,
            'sparsity': 0.0,
            'overlap': 0.0
        }
        
        pbar = tqdm(dataloader, desc=f'Train Epoch {epoch+1}/{self.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            X = batch['X'].to(self.device)
            y = batch['y'].to(self.device)
            # Note: wn is already set on model during setup
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.amp_enabled:
                with autocast():
                    logits, cutoffs = self.model(X)
                    losses = self.loss_fn(logits, y, cutoffs)
                    loss = losses['total']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                logits, cutoffs = self.model(X)
                losses = self.loss_fn(logits, y, cutoffs)
                loss = losses['total']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # Optimizer step
                self.optimizer.step()
            
            # Update metrics - properly track all loss components
            for key in losses:
                if key == 'total':
                    metrics['loss'] += losses[key].item()
                elif key in metrics:
                    metrics[key] += losses[key].item()
                else:
                    # Add any new loss components
                    metrics[key] = losses[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log to tensorboard
            global_step = epoch * len(dataloader) + batch_idx
            if batch_idx % 10 == 0:
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
        
        # Average metrics
        n_batches = len(dataloader)
        for key in metrics:
            metrics[key] /= n_batches
        
        return metrics
    
    @torch.no_grad()
    def validate_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
        valid_samples = 0  # Track cores with valid losses
        
        pbar = tqdm(dataloader, desc=f'Val Epoch {epoch+1}/{self.epochs}')
        
        for batch in pbar:
            # Move to device
            X = batch['X'].to(self.device)
            y = batch['y'].to(self.device)
            # Note: wn is already set on model during setup
            
            # Skip cores with only background pixels (prevents NaN)
            if (y != 0).sum() == 0:  # Assuming 0 is background/ignore_index
                continue
            
            # Forward pass
            if self.amp_enabled:
                with autocast():
                    logits, cutoffs = self.model(X)
                    losses = self.loss_fn(logits, y, cutoffs)
            else:
                logits, cutoffs = self.model(X)
                losses = self.loss_fn(logits, y, cutoffs)
            
            # Skip if loss is NaN (safety check)
            if torch.isnan(losses['total']):
                continue
            
            total_loss += losses['total'].item()
            valid_samples += 1
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            # Store for metrics computation (flatten spatial dimensions)
            all_preds.append(preds.cpu().flatten())
            all_targets.append(y.cpu().flatten())
        
        # Compute metrics
        if valid_samples > 0:
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            metrics = compute_metrics(all_preds, all_targets, num_classes=self.cfg['model']['classes'])
            metrics['loss'] = total_loss / valid_samples
        else:
            # No valid samples - return NaN metrics
            metrics = {
                'loss': float('nan'),
                'miou': 0.0,
                'dice': 0.0
            }
        
        return metrics
    
    def save_best_model(self, metrics: Dict[str, float], epoch: int):
        """Save best model based on validation metrics"""
        
        # Check if this is the best model
        is_best = False
        if metrics['miou'] > self.best_val_miou:
            self.best_val_miou = metrics['miou']
            self.best_epoch = epoch
            is_best = True
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.cfg,
            'fold': self.fold
        }
        
        # Save latest
        torch.save(checkpoint, self.model_dir / 'checkpoint_latest.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.model_dir / 'checkpoint_best.pth')
            
            # Also save model weights only for easy loading
            torch.save(self.model.state_dict(), self.model_dir / 'best_model.pth')
            
            # Log best metrics
            with open(self.model_dir / 'best_metrics.json', 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'metrics': {k: float(v) for k, v in metrics.items()}
                }, f, indent=2)
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        
        print(f"\nStarting training for fold {self.fold}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Set wavenumber array on model (get from first batch)
        first_batch = next(iter(train_loader))
        if hasattr(self.model, 'set_wavenumbers'):
            self.model.set_wavenumbers(first_batch['wn'])
            print(f"Set wavenumber array: {len(first_batch['wn'])} channels")
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Save best model
            self.save_best_model(val_metrics, epoch)
            
            # Log metrics
            epoch_time = time.time() - start_time
            self.log_epoch(epoch, train_metrics, val_metrics, epoch_time)
            
            # Store metrics
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Save final metrics
        self.save_training_history()
        
        print(f"\nTraining completed!")
        print(f"Best model: epoch {self.best_epoch+1} with mIoU={self.best_val_miou:.4f}")
    
    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict, epoch_time: float):
        """Log metrics for current epoch"""
        
        # Console logging
        print(f"\nEpoch {epoch+1}/{self.epochs} - {epoch_time:.1f}s")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"CE: {train_metrics.get('ce', 0):.4f}, "
              f"Dice: {train_metrics.get('dice', 0):.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"mIoU: {val_metrics['miou']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}")
        
        # Tensorboard logging
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)
        
        for key, value in val_metrics.items():
            if not key.startswith('iou_class') and not key.startswith('dice_class'):
                self.writer.add_scalar(f'val/{key}', value, epoch)
        
        self.writer.add_scalar('learning_rate', 
                              self.optimizer.param_groups[0]['lr'], epoch)
        
        # Log filter statistics
        if epoch % 10 == 0:
            with torch.no_grad():
                f_low, f_high = self.model.get_cutoffs_cm1()
                bandwidths = f_high - f_low
                
                self.writer.add_histogram('filters/low_cutoffs', f_low, epoch)
                self.writer.add_histogram('filters/high_cutoffs', f_high, epoch)
                self.writer.add_histogram('filters/bandwidths', bandwidths, epoch)
                self.writer.add_scalar('filters/mean_bandwidth', 
                                      bandwidths.mean().item(), epoch)
    
    def save_training_history(self):
        """Save training history to CSV"""
        
        # Combine metrics
        history = []
        for i, (train, val) in enumerate(zip(self.train_metrics, self.val_metrics)):
            row = {'epoch': i+1, 'fold': self.fold}
            for key, value in train.items():
                row[f'train_{key}'] = value
            for key, value in val.items():
                if not key.startswith('iou_class') and not key.startswith('dice_class'):
                    row[f'val_{key}'] = value
            history.append(row)
        
        # Save to CSV
        df = pd.DataFrame(history)
        df.to_csv(self.model_dir / 'training_history.csv', index=False)
        
        # Save summary
        summary = {
            'fold': self.fold,
            'best_epoch': self.best_epoch + 1,
            'best_val_miou': float(self.best_val_miou),
            'total_epochs': len(history),
            'early_stopped': self.patience_counter >= self.patience
        }
        
        with open(self.model_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


def train_fold(cfg: Dict, fold: int, train_paths: List[str], val_paths: List[str]):
    """Train a single fold"""
    
    # Create dataloaders (disable z-score for SSMU-Net, use raw double-L2)
    loaders = create_dataloaders(cfg, train_paths, val_paths, use_zscore=False)
    
    # Compute class weights with optional clipping
    clip_range = None
    if cfg['loss']['seg'].get('class_weights') == 'inverse_freq':
        clip_range = cfg['loss']['seg'].get('class_weight_clip', None)
    
    class_weights = compute_class_weights(
        loaders['train'].dataset, 
        clip_range=clip_range
    )
    print(f"Class weights: {class_weights}")
    
    # Create trainer (seeding happens inside __init__ after dirs are created)
    trainer = Trainer(cfg, fold)
    trainer.setup_model_and_optimizer(class_weights)
    
    # Train
    trainer.train(loaders['train'], loaders['val'])
    
    # Cleanup
    trainer.writer.close()
    
    return trainer.best_val_miou


def train_cross_validation(cfg: Dict):
    """Run full cross-validation training"""
    
    print("\n" + "="*60)
    print("Starting SSMU-Net Cross-Validation Training")
    print("="*60)
    
    # Load manifest and create splits - FIX: use tables directory
    manifest_path = Path(cfg['runtime_paths']['tables']) / 'npz_manifest.csv'
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run preprocessing first.")
    
    holdout_keys = cfg['audits']['holdout']
    n_folds = cfg['data']['folds']
    
    splits = create_data_splits(
        str(manifest_path),
        n_folds=n_folds,
        holdout_keys=holdout_keys,
        seed=cfg['data']['seed']
    )
    
    # Save splits for reproducibility
    splits_path = Path(cfg['runtime_paths']['logs']) / 'cv_splits.json'
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved CV splits to {splits_path}")
    
    # Train each fold
    fold_scores = []
    for split in splits:
        fold_idx = split['fold']
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx+1}/{n_folds}")
        print(f"{'='*60}")
        
        best_miou = train_fold(
            cfg, 
            fold_idx,
            split['train'],
            split['val']
        )
        
        fold_scores.append(best_miou)
        print(f"Fold {fold_idx} best mIoU: {best_miou:.4f}")
    
    # Report final results
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    for i, score in enumerate(fold_scores):
        print(f"Fold {i}: {score:.4f}")
    print(f"Mean mIoU: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    # Save final results
    results = {
        'fold_scores': fold_scores,
        'mean_miou': float(np.mean(fold_scores)),
        'std_miou': float(np.std(fold_scores)),
        'config': cfg
    }
    
    results_path = Path(cfg['runtime_paths']['logs']) / 'cv_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return fold_scores


if __name__ == "__main__":
    # Load config
    cfg = load_config("ssmu_net/config.yaml")
    
    # Run training
    train_cross_validation(cfg)