#!/usr/bin/env python
"""
Preprocessing script to convert raw QCL data to NPZ format
"""

import os
import sys
import json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ssmu_net.utils import load_config, ensure_dirs, set_deterministic
from ssmu_net.preprocess import preprocess_core, save_label_map


def enumerate_cores(metadata_excel: str, raw_root: str):
    """Enumerate cores from metadata Excel file"""
    df = pd.read_excel(metadata_excel)
    cores = []
    
    for _, row in df.iterrows():
        pos = row['Position'].strip()
        # Filter cores C2 to M16 as in original preprocessing
        letter, num = pos[0], int(pos[1:])
        if letter < 'C' or letter > 'M' or num < 2 or num > 16:
            continue
            
        grid = int(row['qcl_grid'])
        if grid < 0:
            continue
            
        # Build path to zarr data
        core_dir = Path(raw_root) / 'Isolated Cores' / row['qcl_folder'] / 'zarr_data' / f'core {grid}'
        
        if core_dir.exists():
            metadata = {
                'core_id': f'core_{pos}',
                'position': pos,
                'slide_id': row.get('slide_id', 'BR2082'),
                'block_id': row.get('block_id', 'H260'),
                'session_id': row.get('session_id', '1'),
                'qcl_folder': row['qcl_folder'],
                'qcl_grid': grid
            }
            cores.append((f'core_{pos}', core_dir, metadata))
    
    return cores


def find_annotation(core_id: str, ann_root: Path):
    """Find annotation PNG for a core"""
    # Extract position from core_id (e.g., 'core_C10' -> 'C10')
    pos = core_id.replace('core_', '')
    
    # Try different naming patterns
    patterns = [
        f"{pos} anno.png",
        f"{pos}_anno.png",
        f"{pos}.png"
    ]
    
    for pattern in patterns:
        p = ann_root / pattern
        if p.exists():
            return p
    
    raise FileNotFoundError(f"Annotation missing for {core_id} in {ann_root}")


def main():
    """Main preprocessing pipeline"""
    # Load configuration
    cfg = load_config("ssmu_net/config.yaml")
    paths = ensure_dirs(cfg)
    set_deterministic(cfg["data"]["seed"], paths["logs"])
    
    # Setup paths
    metadata_excel = cfg["data"]["metadata_excel"]
    raw_root = cfg["data"]["raw_root"]
    ann_root = Path(cfg["data"]["annotations_dir"])
    npz_dir = Path(paths["npz"])
    tables_dir = Path(paths["tables"])
    logs_dir = Path(paths["logs"])
    
    # Save label mapping
    save_label_map(str(npz_dir))
    
    # Enumerate cores
    print(f"Reading metadata from {metadata_excel}")
    cores = enumerate_cores(metadata_excel, raw_root)
    print(f"Found {len(cores)} cores to process")
    
    # Process each core
    manifest = []
    failed = []
    
    for core_id, core_path, metadata in tqdm(cores, desc="Processing cores"):
        try:
            # Find annotation
            ann_path = find_annotation(core_id, ann_root)
            
            # Preprocess core
            data, mask, wn, delta_cm1, qc_mask, meta = preprocess_core(
                str(core_path), str(ann_path), cfg, logs_dir=str(logs_dir)
            )
            
            # Update metadata
            meta.update(metadata)
            
            # Save NPZ
            out_path = npz_dir / f"{core_id}.npz"
            np.savez_compressed(
                out_path,
                X=data.astype(np.float32),
                y=mask.astype(np.uint8),
                wn=wn.astype(np.float32),
                tissue_mask=qc_mask.astype(np.uint8),
                delta_cm1=np.float32(delta_cm1),
                meta=json.dumps(meta)
            )
            
            # Add to manifest
            manifest.append({
                "core_id": core_id,
                "npz": str(out_path),
                "H": int(data.shape[0]),
                "W": int(data.shape[1]),
                "C": int(data.shape[2]),
                "delta_cm1": float(delta_cm1),
                "slide_id": metadata.get("slide_id"),
                "block_id": metadata.get("block_id"),
                "session_id": metadata.get("session_id")
            })
            
            print(f"✓ Processed {core_id}: {data.shape}")
            
        except Exception as e:
            print(f"✗ Failed {core_id}: {e}")
            failed.append({"core_id": core_id, "error": str(e)})
    
    # Save manifest
    if manifest:
        manifest_df = pd.DataFrame(manifest)
        manifest_df.to_csv(tables_dir / "npz_manifest.csv", index=False)
        print(f"\nSaved manifest with {len(manifest)} cores")
    
    # Save failed cores list
    if failed:
        failed_df = pd.DataFrame(failed)
        failed_df.to_csv(tables_dir / "failed_cores.csv", index=False)
        print(f"Failed to process {len(failed)} cores - see failed_cores.csv")
    
    # Smoke test on first NPZ
    if manifest:
        print("\nRunning smoke test on first NPZ...")
        test = np.load(manifest[0]["npz"])
        
        # Check wavenumber uniformity
        wn = test["wn"]
        delta = float(np.diff(wn).mean())
        assert np.allclose(np.diff(wn), delta, rtol=1e-6), "Non-uniform wavenumber grid"
        
        # Check data types
        assert test["X"].dtype == np.float32, f"X dtype is {test['X'].dtype}, expected float32"
        assert test["y"].dtype == np.uint8, f"y dtype is {test['y'].dtype}, expected uint8"
        assert test["tissue_mask"].dtype == np.uint8, f"tissue_mask dtype is {test['tissue_mask'].dtype}, expected uint8"
        
        # Check label range
        assert test["y"].min() >= 0 and test["y"].max() <= 7, \
            f"Label values out of range: [{test['y'].min()}, {test['y'].max()}]"
        
        print("✓ Smoke test passed!")
        print(f"  Shape: X={test['X'].shape}, y={test['y'].shape}")
        print(f"  Delta: {delta:.3f} cm⁻¹")
        print(f"  Labels: {np.unique(test['y'])}")
    
    print("\nPreprocessing complete!")
    print(f"Outputs saved to: {npz_dir}")


if __name__ == "__main__":
    main()