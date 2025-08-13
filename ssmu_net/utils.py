"""
Utility functions for SSMU-Net project
"""

import os
import sys
import json
import random
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dirs(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Create all output directories and persist resolved config"""
    base_dir = Path(__file__).parent.parent
    
    # Initialize runtime_paths if not present
    cfg.setdefault('runtime_paths', {})
    
    # Create directories with absolute paths
    for folder in ['npz', 'models', 'figures', 'tables', 'logs']:
        path = base_dir / 'outputs' / folder
        path.mkdir(parents=True, exist_ok=True)
        cfg['runtime_paths'][folder] = str(path.resolve())  # Use absolute paths
    
    # Persist resolved config
    config_path = Path(cfg['runtime_paths']['logs']) / 'config_resolved.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    print(f"Created output directories and saved config to {config_path}")
    return cfg['runtime_paths']


def set_deterministic(seed: int, log_dir: str, deterministic: bool = True) -> None:
    """Set all seeds with full logging for reproducibility"""
    # Set environment variables for determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    
    # Set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Set deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Use warn_only mode to allow non-deterministic ops like 2D cross-entropy
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
    
    # Save seeds to JSON
    seeds = {
        'torch': seed,
        'numpy': seed,
        'random': seed,
        'cuda': seed,
        'python_hash_seed': seed,
        'cublas_workspace_config': ':16:8' if deterministic else 'not_set',
        'cudnn_deterministic': deterministic,
        'cudnn_benchmark': not deterministic,
        'use_deterministic_algorithms': deterministic,
        'timestamp': str(np.datetime64('now'))
    }
    
    seeds_path = Path(log_dir) / 'seeds.json'
    with open(seeds_path, 'w') as f:
        json.dump(seeds, f, indent=2)
    
    mode = "Deterministic" if deterministic else "Non-deterministic (faster)"
    print(f"{mode} mode enabled with seed {seed}")
    print(f"Seeds saved to {seeds_path}")


def enforce_npz_only() -> None:
    """Assert no SpecML import during training/eval"""
    assert 'specml' not in sys.modules, "SpecML must not be imported during training/eval"
    print("✓ NPZ-only mode enforced (no SpecML imports)")


def check_banned_terms(project_dir: str = '.') -> None:
    """Check for banned terms in code and documentation"""
    banned = ['Beer-Lambert', 'Beer Lambert', 'RMieS', 'EMSC', 'Beer–Lambert']
    violations = []
    
    for root, dirs, files in os.walk(project_dir):
        # Skip outputs directory
        if 'outputs' in root or '__pycache__' in root or '.git' in root:
            continue
            
        for file in files:
            if file.endswith(('.py', '.tex', '.md', '.yaml', '.txt')):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for term in banned:
                            if term.lower() in content.lower():
                                violations.append((path, term))
                except Exception as e:
                    print(f"Warning: Could not read {path}: {e}")
    
    if violations:
        print("ERROR: Banned terms found:")
        for path, term in violations:
            print(f"  - '{term}' in {path}")
        raise ValueError(f"Found {len(violations)} banned term violations")
    
    print("✓ No banned terms found")


def save_json(data: Any, path: str) -> None:
    """Save data to JSON file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    """Load data from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count