"""
DFIR (Discrete Frequency IR) band selection and export
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast
import numpy as np
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Set
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.metrics import mutual_info_score

from .models import create_model, SSMUNet
from .losses import create_loss
from .data import create_dataloaders, NpzCoreDataset
from .evaluation_metrics import compute_metrics
from .utils import load_config


class DFIRExporter:
    """Export optimal DFIR bands from trained SSMU-Net model"""
    
    def __init__(self, cfg: Dict, checkpoint_path: str):
        """
        Args:
            cfg: Configuration dictionary
            checkpoint_path: Path to trained model checkpoint
        """
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained model
        self.model = create_model(cfg).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"Loaded model from {checkpoint_path}")
        
        # DFIR export settings
        self.dfir_cfg = cfg.get('dfir_export', {})
        self.target_K_values = self.dfir_cfg.get('target_K', [12, 16, 20, 24])
        self.min_separation_cm1 = self.dfir_cfg.get('min_sep_cm1', 8.0)
        self.selection_method = self.dfir_cfg.get('selection', 'greedy_nonoverlap')
        self.retrain_epochs = self.dfir_cfg.get('retrain_epochs', 30)
    
    def _forward_with_padding(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic padding for U-Net divisibility requirements
        
        Args:
            X: Input tensor (B, C, H, W)
        
        Returns:
            Logits tensor with original spatial dimensions
        """
        _, _, H, W = X.shape
        ph = (16 - (H % 16)) % 16
        pw = (16 - (W % 16)) % 16
        
        # Pad if necessary
        Xp = F.pad(X, (0, pw, 0, ph), mode="replicate") if (ph or pw) else X
        
        # Use AMP if configured
        use_amp = bool(self.cfg.get('optim', {}).get('amp', False) and torch.cuda.is_available())
        
        with autocast(enabled=use_amp):
            logits, _ = self.model(Xp)
        
        # Crop back to original size
        return logits[..., :H, :W] if (ph or pw) else logits
    
    def extract_learned_bands(self) -> List[Tuple[float, float]]:
        """
        Extract learned band centers and widths from Sinc filters
        
        Returns:
            List of (center_cm1, bandwidth_cm1) tuples
        """
        with torch.no_grad():
            f_low, f_high = self.model.get_cutoffs_cm1()
        
        f_low = f_low.numpy()
        f_high = f_high.numpy()
        
        bands = []
        for i in range(len(f_low)):
            center = (f_low[i] + f_high[i]) / 2
            bandwidth = f_high[i] - f_low[i]
            bands.append((float(center), float(bandwidth)))
        
        # Sort by center frequency
        bands.sort(key=lambda x: x[0])
        
        return bands
    
    @torch.no_grad()
    def compute_band_importance(self, dataset: NpzCoreDataset) -> np.ndarray:
        """
        Compute importance scores for each learned band using gradient-based attribution
        
        Args:
            dataset: Validation dataset
        
        Returns:
            Array of importance scores for each filter
        """
        print("Computing band importance scores...")
        
        # Get filter parameters
        f_low, f_high = self.model.get_cutoffs_cm1()
        f_low = f_low.cpu().numpy()
        f_high = f_high.cpu().numpy()
        n_filters = len(f_low)
        
        # Initialize importance scores
        importance_scores = np.zeros(n_filters)
        
        # Sample a few cores for importance computation
        n_samples = min(10, len(dataset))
        
        for idx in tqdm(range(n_samples), desc="Computing importance"):
            batch = dataset[idx]
            X = batch['X'].unsqueeze(0).to(self.device)
            y = batch['y'].to(self.device)
            
            # Get baseline prediction
            logits_base = self._forward_with_padding(X)
            preds_base = torch.argmax(logits_base, dim=1).squeeze(0)
            
            # Create valid mask (ignore -100 pixels)
            valid = (y != -100)
            
            if not valid.any():
                continue
            
            # Compute baseline accuracy on valid pixels only
            correct_base = (preds_base[valid] == y[valid]).float().mean()
            
            # For each filter, compute importance by masking
            for filter_idx in range(n_filters):
                # Create a model copy for this evaluation
                X_masked = X.clone()
                
                # Find spectral channels corresponding to this filter
                wn = batch['wn'].numpy()
                mask_channels = np.where(
                    (wn >= f_low[filter_idx]) & 
                    (wn <= f_high[filter_idx])
                )[0]
                
                if len(mask_channels) > 0:
                    X_masked[:, mask_channels, :, :] = 0
                    
                    # Get prediction without this filter
                    logits_masked = self._forward_with_padding(X_masked)
                    preds_masked = torch.argmax(logits_masked, dim=1).squeeze(0)
                    
                    # Compute performance drop on valid pixels only
                    correct_masked = (preds_masked[valid] == y[valid]).float().mean()
                    importance = (correct_base - correct_masked).item()
                    
                    importance_scores[filter_idx] += max(0, importance)
        
        # Normalize by number of samples
        importance_scores /= n_samples
        
        return importance_scores
    
    def select_dfir_bands(self, 
                         bands: List[Tuple[float, float]], 
                         importance: np.ndarray,
                         K: int) -> List[int]:
        """
        Select K optimal bands for DFIR
        
        Args:
            bands: List of (center, bandwidth) tuples
            importance: Importance scores for each band
            K: Number of bands to select
        
        Returns:
            Indices of selected bands
        """
        if self.selection_method == 'greedy_nonoverlap':
            return self._greedy_nonoverlap_selection(bands, importance, K)
        elif self.selection_method == 'importance_threshold':
            return self._importance_threshold_selection(bands, importance, K)
        elif self.selection_method == 'uniform_coverage':
            return self._uniform_coverage_selection(bands, importance, K)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def _greedy_nonoverlap_selection(self, 
                                    bands: List[Tuple[float, float]], 
                                    importance: np.ndarray,
                                    K: int) -> List[int]:
        """
        Greedy selection ensuring no overlap and minimum separation
        """
        n_bands = len(bands)
        selected = []
        available = set(range(n_bands))
        
        # Sort by importance
        sorted_indices = np.argsort(importance)[::-1]
        
        for idx in sorted_indices:
            if len(selected) >= K:
                break
            
            if idx not in available:
                continue
            
            center_i, bw_i = bands[idx]
            
            # Check separation from already selected bands
            can_add = True
            for sel_idx in selected:
                center_j, bw_j = bands[sel_idx]
                
                # Check for overlap or insufficient separation
                separation = abs(center_i - center_j) - (bw_i + bw_j) / 2
                if separation < self.min_separation_cm1:
                    can_add = False
                    break
            
            if can_add:
                selected.append(idx)
                available.remove(idx)
        
        return selected
    
    def _importance_threshold_selection(self, 
                                      bands: List[Tuple[float, float]], 
                                      importance: np.ndarray,
                                      K: int) -> List[int]:
        """
        Select bands above importance threshold
        """
        # Find threshold that gives approximately K bands
        sorted_importance = np.sort(importance)[::-1]
        if K < len(sorted_importance):
            threshold = sorted_importance[K]
        else:
            threshold = 0
        
        selected = []
        for i, imp in enumerate(importance):
            if imp >= threshold and len(selected) < K:
                selected.append(i)
        
        return selected
    
    def _uniform_coverage_selection(self, 
                                   bands: List[Tuple[float, float]], 
                                   importance: np.ndarray,
                                   K: int) -> List[int]:
        """
        Select bands for uniform spectral coverage
        """
        # Get spectral range
        all_centers = [b[0] for b in bands]
        wn_min, wn_max = min(all_centers), max(all_centers)
        
        # Divide spectrum into K regions
        region_width = (wn_max - wn_min) / K
        selected = []
        
        for i in range(K):
            region_start = wn_min + i * region_width
            region_end = region_start + region_width
            
            # Find most important band in this region
            candidates = []
            for j, (center, _) in enumerate(bands):
                if region_start <= center <= region_end:
                    candidates.append((j, importance[j]))
            
            if candidates:
                # Select most important in region
                best_idx = max(candidates, key=lambda x: x[1])[0]
                selected.append(best_idx)
        
        return selected
    
    @torch.no_grad()
    def evaluate_dfir_model(self, 
                          selected_indices: List[int],
                          dataset: NpzCoreDataset) -> Dict[str, float]:
        """
        Evaluate performance using only selected DFIR bands
        
        Args:
            selected_indices: Indices of selected bands
            dataset: Evaluation dataset
        
        Returns:
            Performance metrics
        """
        print(f"Evaluating DFIR model with {len(selected_indices)} bands...")
        
        # Get filter ranges for masking
        f_low, f_high = self.model.get_cutoffs_cm1()
        f_low = f_low.cpu().numpy()
        f_high = f_high.cpu().numpy()
        
        all_preds = []
        all_targets = []
        
        for idx in tqdm(range(min(10, len(dataset))), desc="Evaluating"):
            batch = dataset[idx]
            X = batch['X'].unsqueeze(0).to(self.device)  # (1, C, H, W)
            y = batch['y']
            
            # Determine which spectral channels to keep based on selected bands
            wn = batch['wn'].cpu().numpy()
            selected_mask = np.zeros_like(wn, dtype=bool)
            
            for k in selected_indices:
                lo, hi = f_low[k], f_high[k]
                selected_mask |= (wn >= lo) & (wn <= hi)
            
            # Zero out channels NOT in the selected bands
            X_mask = X.clone()
            keep = torch.from_numpy(selected_mask).to(X_mask.device)
            X_mask[:, ~keep, :, :] = 0
            
            # Get predictions with only selected bands
            logits = self._forward_with_padding(X_mask)
            preds = torch.argmax(logits, dim=1).squeeze(0)
            
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = compute_metrics(all_preds, all_targets, 
                                num_classes=self.cfg['model']['classes'])
        
        return metrics
    
    def export_dfir_specification(self, 
                                 bands: List[Tuple[float, float]],
                                 selected_indices: List[int],
                                 importance: np.ndarray,
                                 metrics: Dict[str, float],
                                 K: int,
                                 output_path: str):
        """
        Export DFIR band specification
        
        Args:
            bands: All learned bands
            selected_indices: Selected band indices
            importance: Importance scores
            metrics: Performance metrics
            K: Number of bands
            output_path: Path to save specification
        """
        selected_bands = []
        for idx in selected_indices:
            center, bandwidth = bands[idx]
            selected_bands.append({
                'index': idx,
                'center_cm1': center,
                'bandwidth_cm1': bandwidth,
                'range_cm1': [center - bandwidth/2, center + bandwidth/2],
                'importance': float(importance[idx])
            })
        
        # Sort by center frequency
        selected_bands.sort(key=lambda x: x['center_cm1'])
        
        specification = {
            'K': K,
            'n_selected': len(selected_indices),
            'selection_method': self.selection_method,
            'min_separation_cm1': self.min_separation_cm1,
            'bands': selected_bands,
            'performance': metrics,
            'spectral_coverage': {
                'min_cm1': min(b['range_cm1'][0] for b in selected_bands),
                'max_cm1': max(b['range_cm1'][1] for b in selected_bands),
                'total_bandwidth': sum(b['bandwidth_cm1'] for b in selected_bands)
            }
        }
        
        # Save specification
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(specification, f, indent=2)
        
        # Also save as CSV for easy viewing
        df = pd.DataFrame(selected_bands)
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"Saved DFIR specification to {output_path}")
        
        return specification
    
    def plot_dfir_bands(self, 
                       bands: List[Tuple[float, float]],
                       selected_indices: List[int],
                       importance: np.ndarray,
                       save_path: str):
        """
        Visualize selected DFIR bands
        
        Args:
            bands: All learned bands
            selected_indices: Selected band indices
            importance: Importance scores
            save_path: Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: All bands with importance
        centers = [b[0] for b in bands]
        bandwidths = [b[1] for b in bands]
        
        # Create colormap based on importance (guard against division by zero)
        den = float(importance.max()) if importance.size and importance.max() > 0 else 1.0
        colors = plt.cm.viridis(importance / den)
        
        for i, (center, bw) in enumerate(bands):
            alpha = 0.3 if i not in selected_indices else 0.8
            color = colors[i]
            ax1.bar(center, importance[i], width=bw, 
                   alpha=alpha, color=color, edgecolor='black', linewidth=0.5)
        
        # Mark selected bands
        for idx in selected_indices:
            center = bands[idx][0]
            ax1.axvline(center, color='red', linestyle='--', alpha=0.5)
        
        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax1.set_ylabel('Importance Score')
        ax1.set_title('Learned Bands and Selection')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Selected DFIR bands only
        for idx in selected_indices:
            center, bw = bands[idx]
            ax2.bar(center, 1.0, width=bw, alpha=0.7, 
                   label=f'{center:.0f}±{bw/2:.0f}')
        
        # Add biochemical region annotations (using axvspan for vertical spans)
        biochemical_regions = {
            'Amide I': (1600, 1700),
            'Amide II': (1500, 1600),
            'Lipid': (1700, 1780),
            'Amide III': (1200, 1300),
            'Nucleic': (1000, 1100)
        }
        
        for name, (start, end) in biochemical_regions.items():
            ax2.axvspan(start, end, alpha=0.12, color='gray')
            ax2.text((start+end)/2, 1.05, name, 
                    ha='center', va='bottom', fontsize=8)
        
        ax2.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax2.set_ylabel('Selected Bands')
        ax2.set_title(f'DFIR Band Selection (K={len(selected_indices)})')
        ax2.set_ylim(0, 1.2)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved DFIR band plot to {save_path}")
    
    def generate_dfir_report(self, output_dir: str) -> Dict[str, Any]:
        """
        Generate complete DFIR export report for all target K values
        
        Args:
            output_dir: Directory to save reports
        
        Returns:
            Summary dictionary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract learned bands
        bands = self.extract_learned_bands()
        print(f"Extracted {len(bands)} learned bands")
        
        # Create small validation dataset for importance computation
        npz_dir = Path(self.cfg['runtime_paths']['npz'])
        val_npz = list(npz_dir.glob("core_*.npz"))[:5]
        
        if not val_npz:
            raise ValueError("No NPZ files found for evaluation")
        
        # Load z-score stats if available
        stats_path = Path(self.cfg['runtime_paths']['tables']) / 'zscore_stats.csv'
        if stats_path.exists():
            stats_df = pd.read_csv(stats_path)
            z_mean = stats_df['mean'].values.astype(np.float32)
            z_std = stats_df['std'].values.astype(np.float32)
        else:
            z_mean, z_std = None, None
        
        val_dataset = NpzCoreDataset(
            [str(f) for f in val_npz],
            mode='test',
            augment=False,
            ignore_index=0,
            z_mean=z_mean,
            z_std=z_std
        )
        
        # Compute band importance
        importance = self.compute_band_importance(val_dataset)
        
        # Generate DFIR specifications for each K
        results = {}
        
        for K in self.target_K_values:
            print(f"\nGenerating DFIR specification for K={K}")
            
            # Select bands
            selected_indices = self.select_dfir_bands(bands, importance, K)
            
            # Evaluate performance (approximation with masking)
            metrics = self.evaluate_dfir_model(selected_indices, val_dataset)
            
            # Export specification
            spec_path = output_dir / f'dfir_K{K}.json'
            specification = self.export_dfir_specification(
                bands, selected_indices, importance, metrics, K, str(spec_path)
            )
            
            # Generate plot
            plot_path = output_dir / f'dfir_K{K}_bands.png'
            self.plot_dfir_bands(bands, selected_indices, importance, str(plot_path))
            
            results[f'K_{K}'] = {
                'n_bands': len(selected_indices),
                'miou': metrics['miou'],
                'dice': metrics['dice'],
                'specification_path': str(spec_path),
                'plot_path': str(plot_path)
            }
        
        # Save summary
        summary = {
            'n_learned_bands': len(bands),
            'selection_method': self.selection_method,
            'min_separation_cm1': self.min_separation_cm1,
            'target_K_values': self.target_K_values,
            'results': results,
            'recommendation': self._get_recommendation(results)
        }
        
        summary_path = output_dir / 'dfir_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nDFIR export complete. Summary saved to {summary_path}")
        
        return summary
    
    def _get_recommendation(self, results: Dict) -> Dict[str, Any]:
        """Get recommended K value based on performance vs complexity tradeoff"""
        
        # Extract performance for each K
        k_performance = []
        for k_str, res in results.items():
            K = int(k_str.split('_')[1])
            miou = res['miou']
            k_performance.append((K, miou))
        
        k_performance.sort(key=lambda x: x[0])
        
        # Find knee point (diminishing returns)
        if len(k_performance) > 2:
            improvements = []
            for i in range(1, len(k_performance)):
                K_prev, miou_prev = k_performance[i-1]
                K_curr, miou_curr = k_performance[i]
                improvement = (miou_curr - miou_prev) / (K_curr - K_prev)
                improvements.append((K_curr, improvement))
            
            # Recommend K with best improvement/complexity ratio
            best_K = k_performance[0][0]  # Default to smallest
            for K, imp in improvements:
                if imp > 0.005:  # At least 0.5% improvement per band
                    best_K = K
        else:
            best_K = k_performance[0][0]
        
        return {
            'recommended_K': best_K,
            'expected_miou': float(results[f'K_{best_K}']['miou']),
            'rationale': f"Best tradeoff between performance and complexity"
        }


def export_dfir_bands(cfg: Dict, checkpoint_path: str, output_dir: str):
    """
    Main function to export DFIR bands
    
    Args:
        cfg: Configuration
        checkpoint_path: Path to trained model
        output_dir: Directory for outputs
    """
    exporter = DFIRExporter(cfg, checkpoint_path)
    summary = exporter.generate_dfir_report(output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("DFIR EXPORT SUMMARY")
    print("="*60)
    
    for k_str, res in summary['results'].items():
        K = k_str.replace('K_', '')
        print(f"K={K}: {res['n_bands']} bands, mIoU={res['miou']:.3f}")
    
    print(f"\nRecommendation: K={summary['recommendation']['recommended_K']}")
    print(f"Expected mIoU: {summary['recommendation']['expected_miou']:.3f}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export DFIR bands from SSMU-Net")
    parser.add_argument("--config", type=str, default="ssmu_net/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/dfir")
    
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    export_dfir_bands(cfg, args.checkpoint, args.output)