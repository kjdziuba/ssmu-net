"""
Auditing and sanity checks for SSMU-Net
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from .models import create_model
from .data import NpzCoreDataset
from .evaluation_metrics import compute_metrics
from .utils import load_config


class ModelAuditor:
    """Comprehensive auditing suite for SSMU-Net models"""
    
    def __init__(self, cfg: Dict, checkpoint_path: str):
        """
        Args:
            cfg: Configuration dictionary
            checkpoint_path: Path to model checkpoint
        """
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = create_model(cfg).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"Loaded model from {checkpoint_path}")
        
        # Audit settings from config
        self.audit_cfg = cfg.get('audits', {})
        self.occlusion_cfg = self.audit_cfg.get('occlusion', {})
        self.jitter_cfg = self.audit_cfg.get('jitter', {})
        self.shuffle_cfg = self.audit_cfg.get('shuffle_controls', {})
    
    @torch.no_grad()
    def spectral_occlusion_analysis(self, dataset: NpzCoreDataset) -> Dict[str, Any]:
        """
        Perform spectral occlusion analysis to identify important wavenumber regions
        
        Args:
            dataset: Test dataset
        
        Returns:
            Dictionary with occlusion analysis results
        """
        print("\nPerforming spectral occlusion analysis...")
        
        # Get occlusion parameters
        width_cm1 = self.occlusion_cfg.get('width_cm1', 15)
        stride_cm1 = self.occlusion_cfg.get('stride_cm1', 5)
        
        # Get a representative sample
        sample_idx = min(5, len(dataset))  # Use first 5 cores
        
        results = []
        
        for idx in range(sample_idx):
            batch = dataset[idx]
            X = batch['X'].unsqueeze(0).to(self.device)  # (1, C, H, W)
            y = batch['y'].to(self.device)
            wn = batch['wn'].numpy()
            
            # Get baseline prediction
            logits_base, _ = self.model(X)
            preds_base = torch.argmax(logits_base, dim=1).squeeze(0)
            baseline_score = compute_metrics(
                preds_base.cpu(), y.cpu(), 
                num_classes=self.cfg['model']['classes']
            )['miou']
            
            # Determine occlusion windows
            wn_min, wn_max = wn.min(), wn.max()
            delta = wn[1] - wn[0]
            
            occlusion_scores = []
            occlusion_centers = []
            
            # Slide occlusion window
            for center_cm1 in np.arange(wn_min + width_cm1/2, 
                                        wn_max - width_cm1/2, 
                                        stride_cm1):
                # Create occluded input
                X_occluded = X.clone()
                
                # Find channels to occlude
                mask = np.abs(wn - center_cm1) <= width_cm1/2
                occluded_channels = np.where(mask)[0]
                
                if len(occluded_channels) > 0:
                    # Zero out occluded channels
                    X_occluded[:, occluded_channels, :, :] = 0
                    
                    # Get prediction with occlusion
                    logits_occ, _ = self.model(X_occluded)
                    preds_occ = torch.argmax(logits_occ, dim=1).squeeze(0)
                    
                    # Compute score drop
                    occ_score = compute_metrics(
                        preds_occ.cpu(), y.cpu(),
                        num_classes=self.cfg['model']['classes']
                    )['miou']
                    
                    score_drop = baseline_score - occ_score
                    occlusion_scores.append(score_drop)
                    occlusion_centers.append(center_cm1)
            
            results.append({
                'core_idx': idx,
                'baseline_miou': baseline_score,
                'occlusion_centers': occlusion_centers,
                'occlusion_drops': occlusion_scores
            })
        
        # Aggregate results
        all_drops = []
        for r in results:
            all_drops.extend(r['occlusion_drops'])
        
        # Find critical regions (where drop is highest)
        centers = results[0]['occlusion_centers']
        mean_drops = []
        
        for i in range(len(centers)):
            drops_at_center = [r['occlusion_drops'][i] for r in results 
                              if i < len(r['occlusion_drops'])]
            mean_drops.append(np.mean(drops_at_center))
        
        # Identify top critical regions
        critical_indices = np.argsort(mean_drops)[-5:]  # Top 5
        critical_regions = [(centers[i], mean_drops[i]) for i in critical_indices]
        
        return {
            'occlusion_width_cm1': width_cm1,
            'occlusion_stride_cm1': stride_cm1,
            'wavenumber_centers': centers,
            'mean_miou_drops': mean_drops,
            'critical_regions': critical_regions,
            'raw_results': results
        }
    
    @torch.no_grad()
    def wavenumber_jitter_test(self, dataset: NpzCoreDataset) -> Dict[str, Any]:
        """
        Test model robustness to wavenumber calibration errors
        
        Args:
            dataset: Test dataset
        
        Returns:
            Dictionary with jitter test results
        """
        print("\nTesting wavenumber jitter robustness...")
        
        sigma_cm1 = self.jitter_cfg.get('sigma_cm1', 3.0)
        n_trials = 10
        
        results = []
        
        for idx in range(min(5, len(dataset))):
            batch = dataset[idx]
            X = batch['X'].to(self.device)
            y = batch['y'].to(self.device)
            
            # Baseline without jitter
            X_batch = X.unsqueeze(0)
            logits_base, _ = self.model(X_batch)
            preds_base = torch.argmax(logits_base, dim=1).squeeze(0)
            baseline_miou = compute_metrics(
                preds_base.cpu(), y.cpu(),
                num_classes=self.cfg['model']['classes']
            )['miou']
            
            # Apply jitter trials
            jittered_mious = []
            
            for trial in range(n_trials):
                # Generate random channel permutation (simulating jitter)
                # This is a simplified version - proper jitter would interpolate
                C = X.shape[0]
                jitter_pixels = np.random.normal(0, sigma_cm1/2, C)
                jitter_pixels = np.clip(jitter_pixels, -5, 5).astype(int)
                
                X_jittered = X.clone()
                for c in range(C):
                    shift = jitter_pixels[c]
                    if shift != 0:
                        X_jittered[c] = torch.roll(X[c], shift, dims=0)
                
                # Evaluate with jitter
                X_jit_batch = X_jittered.unsqueeze(0)
                logits_jit, _ = self.model(X_jit_batch)
                preds_jit = torch.argmax(logits_jit, dim=1).squeeze(0)
                
                jit_miou = compute_metrics(
                    preds_jit.cpu(), y.cpu(),
                    num_classes=self.cfg['model']['classes']
                )['miou']
                
                jittered_mious.append(jit_miou)
            
            results.append({
                'core_idx': idx,
                'baseline_miou': baseline_miou,
                'jittered_mious': jittered_mious,
                'mean_jittered': np.mean(jittered_mious),
                'std_jittered': np.std(jittered_mious),
                'robustness_score': 1.0 - abs(baseline_miou - np.mean(jittered_mious))
            })
        
        # Aggregate
        mean_robustness = np.mean([r['robustness_score'] for r in results])
        
        return {
            'jitter_sigma_cm1': sigma_cm1,
            'n_trials': n_trials,
            'mean_robustness_score': mean_robustness,
            'per_core_results': results
        }
    
    @torch.no_grad()
    def shuffle_control_tests(self, dataset: NpzCoreDataset) -> Dict[str, Any]:
        """
        Perform negative control tests with shuffled data
        
        Args:
            dataset: Test dataset
        
        Returns:
            Dictionary with shuffle control results
        """
        print("\nPerforming shuffle control tests...")
        
        results = {}
        
        # Test 1: Channel shuffle (destroy spectral relationships)
        if self.shuffle_cfg.get('channel_shuffle', True):
            print("  - Channel shuffle test...")
            channel_results = []
            
            for idx in range(min(5, len(dataset))):
                batch = dataset[idx]
                X = batch['X'].to(self.device)
                y = batch['y'].to(self.device)
                
                # Baseline
                X_batch = X.unsqueeze(0)
                logits_base, _ = self.model(X_batch)
                preds_base = torch.argmax(logits_base, dim=1).squeeze(0)
                baseline_miou = compute_metrics(
                    preds_base.cpu(), y.cpu(),
                    num_classes=self.cfg['model']['classes']
                )['miou']
                
                # Shuffle channels
                C = X.shape[0]
                perm = torch.randperm(C)
                X_shuffled = X[perm]
                
                X_shuf_batch = X_shuffled.unsqueeze(0)
                logits_shuf, _ = self.model(X_shuf_batch)
                preds_shuf = torch.argmax(logits_shuf, dim=1).squeeze(0)
                shuffled_miou = compute_metrics(
                    preds_shuf.cpu(), y.cpu(),
                    num_classes=self.cfg['model']['classes']
                )['miou']
                
                channel_results.append({
                    'baseline': baseline_miou,
                    'shuffled': shuffled_miou,
                    'degradation': baseline_miou - shuffled_miou
                })
            
            results['channel_shuffle'] = {
                'mean_degradation': np.mean([r['degradation'] for r in channel_results]),
                'details': channel_results
            }
        
        # Test 2: Label shuffle (random labels)
        if self.shuffle_cfg.get('label_shuffle', True):
            print("  - Label shuffle test...")
            label_results = []
            
            for idx in range(min(5, len(dataset))):
                batch = dataset[idx]
                X = batch['X'].to(self.device)
                y = batch['y'].cpu().numpy()
                
                # Create random labels preserving class distribution
                unique_labels = np.unique(y[y != -100])
                y_shuffled = y.copy()
                valid_mask = y != -100
                
                # Randomly reassign labels
                valid_pixels = y_shuffled[valid_mask]
                np.random.shuffle(valid_pixels)
                y_shuffled[valid_mask] = valid_pixels
                
                y_shuffled = torch.from_numpy(y_shuffled).to(self.device)
                
                # Evaluate
                X_batch = X.unsqueeze(0)
                logits, _ = self.model(X_batch)
                preds = torch.argmax(logits, dim=1).squeeze(0)
                
                random_miou = compute_metrics(
                    preds.cpu(), y_shuffled.cpu(),
                    num_classes=self.cfg['model']['classes']
                )['miou']
                
                # Expected random chance
                n_classes = len(unique_labels)
                expected_random = 1.0 / n_classes
                
                label_results.append({
                    'random_miou': random_miou,
                    'expected_random': expected_random,
                    'ratio': random_miou / expected_random
                })
            
            results['label_shuffle'] = {
                'mean_random_miou': np.mean([r['random_miou'] for r in label_results]),
                'expected_random': np.mean([r['expected_random'] for r in label_results]),
                'details': label_results
            }
        
        # Test 3: Spatial misregistration
        if 'mask_misregister' in self.shuffle_cfg:
            print("  - Mask misregistration test...")
            shifts = self.shuffle_cfg['mask_misregister']  # e.g., [2, 4, 8] pixels
            misreg_results = []
            
            for shift in shifts:
                shift_results = []
                
                for idx in range(min(3, len(dataset))):
                    batch = dataset[idx]
                    X = batch['X'].to(self.device)
                    y = batch['y'].to(self.device)
                    
                    # Baseline
                    X_batch = X.unsqueeze(0)
                    logits, _ = self.model(X_batch)
                    preds = torch.argmax(logits, dim=1).squeeze(0)
                    
                    # Shift ground truth
                    y_shifted = torch.roll(y, shifts=(shift, shift), dims=(0, 1))
                    
                    miou_shifted = compute_metrics(
                        preds.cpu(), y_shifted.cpu(),
                        num_classes=self.cfg['model']['classes']
                    )['miou']
                    
                    shift_results.append(miou_shifted)
                
                misreg_results.append({
                    'shift_pixels': shift,
                    'mean_miou': np.mean(shift_results)
                })
            
            results['mask_misregistration'] = misreg_results
        
        return results
    
    def analyze_filter_usage(self) -> Dict[str, Any]:
        """
        Analyze learned Sinc filter characteristics
        
        Returns:
            Dictionary with filter analysis
        """
        print("\nAnalyzing learned Sinc filters...")
        
        with torch.no_grad():
            f_low, f_high = self.model.get_cutoffs_cm1()
        
        f_low = f_low.numpy()
        f_high = f_high.numpy()
        bandwidths = f_high - f_low
        centers = (f_low + f_high) / 2
        
        # Check for overlaps
        overlaps = []
        n_filters = len(f_low)
        
        for i in range(n_filters):
            for j in range(i + 1, n_filters):
                overlap_start = max(f_low[i], f_low[j])
                overlap_end = min(f_high[i], f_high[j])
                if overlap_start < overlap_end:
                    overlaps.append({
                        'filter_i': i,
                        'filter_j': j,
                        'overlap_range': (overlap_start, overlap_end),
                        'overlap_width': overlap_end - overlap_start
                    })
        
        # Identify biochemical associations
        biochemical_bands = {
            'Amide I': (1600, 1700),
            'Amide II': (1500, 1600),
            'Lipid ester': (1700, 1780),
            'Amide III': (1200, 1300),
            'Nucleic acid': (1000, 1100),
            'CH2 bending': (1400, 1500),
            'Carbohydrate': (900, 1200)
        }
        
        filter_assignments = []
        for i, (c, bw) in enumerate(zip(centers, bandwidths)):
            assigned = None
            for name, (b_min, b_max) in biochemical_bands.items():
                if b_min <= c <= b_max:
                    assigned = name
                    break
            filter_assignments.append({
                'filter_idx': i,
                'center_cm1': c,
                'bandwidth_cm1': bw,
                'range': (f_low[i], f_high[i]),
                'biochemical': assigned or 'Unknown'
            })
        
        return {
            'n_filters': n_filters,
            'mean_bandwidth': float(np.mean(bandwidths)),
            'std_bandwidth': float(np.std(bandwidths)),
            'min_bandwidth': float(np.min(bandwidths)),
            'max_bandwidth': float(np.max(bandwidths)),
            'n_overlaps': len(overlaps),
            'overlaps': overlaps[:5],  # Top 5 overlaps
            'filter_assignments': filter_assignments,
            'coverage': {
                'total_range': (float(f_low.min()), float(f_high.max())),
                'gaps': self._find_gaps(f_low, f_high)
            }
        }
    
    def _find_gaps(self, f_low: np.ndarray, f_high: np.ndarray) -> List[Tuple[float, float]]:
        """Find spectral gaps not covered by any filter"""
        # Sort filters by lower bound
        indices = np.argsort(f_low)
        gaps = []
        
        for i in range(len(indices) - 1):
            idx_curr = indices[i]
            idx_next = indices[i + 1]
            
            if f_high[idx_curr] < f_low[idx_next]:
                gaps.append((float(f_high[idx_curr]), float(f_low[idx_next])))
        
        return gaps
    
    def generate_audit_report(self, 
                            occlusion_results: Dict,
                            jitter_results: Dict,
                            shuffle_results: Dict,
                            filter_analysis: Dict,
                            output_path: str):
        """
        Generate comprehensive audit report
        
        Args:
            occlusion_results: Results from occlusion analysis
            jitter_results: Results from jitter test
            shuffle_results: Results from shuffle controls
            filter_analysis: Results from filter analysis
            output_path: Path to save report
        """
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_config': self.cfg['model'],
            
            'occlusion_analysis': {
                'summary': {
                    'critical_regions': occlusion_results['critical_regions'],
                    'most_important': occlusion_results['critical_regions'][0] if occlusion_results['critical_regions'] else None
                },
                'parameters': {
                    'width_cm1': occlusion_results['occlusion_width_cm1'],
                    'stride_cm1': occlusion_results['occlusion_stride_cm1']
                }
            },
            
            'robustness_tests': {
                'wavenumber_jitter': {
                    'sigma_cm1': jitter_results['jitter_sigma_cm1'],
                    'robustness_score': jitter_results['mean_robustness_score'],
                    'interpretation': 'Good' if jitter_results['mean_robustness_score'] > 0.9 else 'Poor'
                }
            },
            
            'negative_controls': {
                'channel_shuffle': {
                    'mean_degradation': shuffle_results.get('channel_shuffle', {}).get('mean_degradation', 0),
                    'passes': shuffle_results.get('channel_shuffle', {}).get('mean_degradation', 0) > 0.3
                },
                'label_shuffle': {
                    'mean_random_miou': shuffle_results.get('label_shuffle', {}).get('mean_random_miou', 0),
                    'expected': shuffle_results.get('label_shuffle', {}).get('expected_random', 0),
                    'passes': shuffle_results.get('label_shuffle', {}).get('mean_random_miou', 1) < 0.2
                }
            },
            
            'filter_analysis': {
                'summary': {
                    'n_filters': filter_analysis['n_filters'],
                    'mean_bandwidth_cm1': filter_analysis['mean_bandwidth'],
                    'n_overlaps': filter_analysis['n_overlaps'],
                    'spectral_gaps': len(filter_analysis['coverage']['gaps'])
                },
                'biochemical_coverage': self._summarize_biochemical_coverage(filter_analysis['filter_assignments'])
            },
            
            'audit_passed': self._check_audit_pass(
                jitter_results, shuffle_results, filter_analysis
            )
        }
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nAudit report saved to {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("AUDIT SUMMARY")
        print("="*60)
        print(f"Overall: {'PASSED' if report['audit_passed'] else 'FAILED'}")
        print(f"Robustness score: {jitter_results['mean_robustness_score']:.3f}")
        print(f"Critical spectral regions: {len(occlusion_results['critical_regions'])}")
        print(f"Filter overlaps: {filter_analysis['n_overlaps']}")
        
        return report
    
    def _summarize_biochemical_coverage(self, assignments: List[Dict]) -> Dict[str, int]:
        """Count filters per biochemical region"""
        coverage = {}
        for a in assignments:
            bio = a['biochemical']
            coverage[bio] = coverage.get(bio, 0) + 1
        return coverage
    
    def _check_audit_pass(self, jitter_results: Dict, 
                         shuffle_results: Dict,
                         filter_analysis: Dict) -> bool:
        """Determine if model passes audit criteria"""
        checks = []
        
        # Robustness check
        checks.append(jitter_results['mean_robustness_score'] > 0.85)
        
        # Negative control checks
        if 'channel_shuffle' in shuffle_results:
            checks.append(shuffle_results['channel_shuffle']['mean_degradation'] > 0.2)
        
        if 'label_shuffle' in shuffle_results:
            checks.append(shuffle_results['label_shuffle']['mean_random_miou'] < 0.25)
        
        # Filter sanity checks
        checks.append(filter_analysis['mean_bandwidth'] < 150)  # Not too broad
        checks.append(filter_analysis['mean_bandwidth'] > 20)   # Not too narrow
        
        return all(checks)
    
    def plot_occlusion_importance(self, occlusion_results: Dict, save_path: str):
        """Plot spectral importance from occlusion analysis"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        centers = occlusion_results['wavenumber_centers']
        drops = occlusion_results['mean_miou_drops']
        
        ax.plot(centers, drops, 'b-', linewidth=2)
        ax.fill_between(centers, 0, drops, alpha=0.3)
        
        # Mark critical regions
        for center, drop in occlusion_results['critical_regions']:
            ax.axvline(center, color='r', linestyle='--', alpha=0.5)
            ax.text(center, drop, f'{center:.0f}', rotation=90, va='bottom')
        
        ax.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax.set_ylabel('mIoU Drop')
        ax.set_title('Spectral Importance via Occlusion Analysis')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved occlusion plot to {save_path}")


def run_full_audit(cfg: Dict, checkpoint_path: str, test_npz_paths: List[str]) -> Dict:
    """
    Run complete model audit
    
    Args:
        cfg: Configuration
        checkpoint_path: Path to model checkpoint
        test_npz_paths: List of test NPZ files
    
    Returns:
        Audit report dictionary
    """
    # Create auditor
    auditor = ModelAuditor(cfg, checkpoint_path)
    
    # Load z-score stats if available
    stats_path = Path(cfg['runtime_paths']['tables']) / 'zscore_stats.csv'
    if stats_path.exists():
        stats_df = pd.read_csv(stats_path)
        z_mean = stats_df['mean'].values.astype(np.float32)
        z_std = stats_df['std'].values.astype(np.float32)
    else:
        z_mean = None
        z_std = None
    
    # Create test dataset
    test_dataset = NpzCoreDataset(
        test_npz_paths,
        mode='test',
        augment=False,
        ignore_index=0,
        z_mean=z_mean,
        z_std=z_std
    )
    
    print("\n" + "="*60)
    print("RUNNING MODEL AUDIT")
    print("="*60)
    
    # Run audit components
    occlusion_results = auditor.spectral_occlusion_analysis(test_dataset)
    jitter_results = auditor.wavenumber_jitter_test(test_dataset)
    shuffle_results = auditor.shuffle_control_tests(test_dataset)
    filter_analysis = auditor.analyze_filter_usage()
    
    # Generate report
    output_dir = Path(cfg['runtime_paths']['tables']) / 'audit'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = auditor.generate_audit_report(
        occlusion_results,
        jitter_results,
        shuffle_results,
        filter_analysis,
        output_dir / 'audit_report.json'
    )
    
    # Generate plots
    fig_dir = Path(cfg['runtime_paths']['figures']) / 'audit'
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    auditor.plot_occlusion_importance(occlusion_results, 
                                     fig_dir / 'occlusion_importance.png')
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audit SSMU-Net model")
    parser.add_argument("--config", type=str, default="ssmu_net/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--npz", type=str, nargs='+', required=True)
    
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    report = run_full_audit(cfg, args.checkpoint, args.npz)
    
    print("\nAudit complete!")