#!/usr/bin/env python3
"""
Project cleanup and organization script
- Removes macOS metadata files
- Archives old/superseded scripts
- Organizes project structure
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime

def remove_macos_files(root_dir):
    """Remove all macOS metadata files (._ and .DS_Store)"""
    removed_count = 0
    patterns = ['._*', '.DS_Store']
    
    for pattern in patterns:
        for file_path in Path(root_dir).rglob(pattern):
            try:
                file_path.unlink()
                removed_count += 1
                print(f"  Removed: {file_path.relative_to(root_dir)}")
            except Exception as e:
                print(f"  Error removing {file_path}: {e}")
    
    return removed_count

def identify_files_to_archive():
    """Identify superseded/old versions of files"""
    
    archive_candidates = {
        'configs': [
            'ssmu_net/config_fixed.yaml',  # Superseded by config_optimized.yaml
            'ssmu_net/config_benchmark.yaml',  # Can keep in archive
            'ssmu_net/config_test.yaml',  # Test config, can archive
            'ssmu_net/my_config.yaml',  # Old user config
        ],
        'scripts': [
            'scripts/validate_data.py',  # Superseded by visualize_training_data.py
            'scripts/test_optimized_preprocessing.py',  # Test script, can archive
            'scripts/test_small_subset.py',  # Old test script
            'scripts/audit_shapes.py',  # Debug script, can archive
        ],
        'preprocessing': [
            'ssmu_net/preprocess.py',  # Old preprocessing, superseded by preprocess_optimized.py
        ],
        'training': [
            'ssmu_net/train_ssmunet.py',  # Old training script
        ],
        'temp_files': [
            'metadata_test_subset.xlsx',  # Temporary test file
        ]
    }
    
    return archive_candidates

def create_archive_structure(root_dir):
    """Create organized archive structure"""
    archive_dir = Path(root_dir) / 'archive'
    
    subdirs = [
        'configs',
        'scripts',
        'preprocessing',
        'training',
        'notebooks',
        'temp_files',
        'old_outputs'
    ]
    
    for subdir in subdirs:
        (archive_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Create archive README
    readme_content = f"""# Archive Directory

Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This directory contains superseded, old, or temporary files from the project.
These files are kept for reference but are no longer actively used.

## Structure:
- `configs/`: Old configuration files
- `scripts/`: Superseded or test scripts  
- `preprocessing/`: Old preprocessing implementations
- `training/`: Previous training scripts
- `notebooks/`: Experimental notebooks
- `temp_files/`: Temporary files from testing
- `old_outputs/`: Previous output directories

## Active Files:
The current active implementations are in the main directories:
- Main config: `ssmu_net/config.yaml` (for SSMU-Net)
- Optimized config: `ssmu_net/config_optimized.yaml` (with new preprocessing)
- Preprocessing: `ssmu_net/preprocess_optimized.py`
- Training: `scripts/train_benchmark.py` (for benchmarks)
- Data visualization: `scripts/visualize_training_data.py`
"""
    
    with open(archive_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    return archive_dir

def archive_files(root_dir, files_to_archive, archive_dir):
    """Move files to archive with logging"""
    archived = []
    
    for category, files in files_to_archive.items():
        for file_path in files:
            src = Path(root_dir) / file_path
            if src.exists():
                # Determine destination
                dest_dir = archive_dir / category
                dest = dest_dir / src.name
                
                # Handle duplicates
                if dest.exists():
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    dest = dest_dir / f"{src.stem}_{timestamp}{src.suffix}"
                
                try:
                    shutil.move(str(src), str(dest))
                    archived.append({
                        'original': str(file_path),
                        'archived_to': str(dest.relative_to(root_dir))
                    })
                    print(f"  Archived: {file_path} → archive/{category}/{dest.name}")
                except Exception as e:
                    print(f"  Error archiving {file_path}: {e}")
    
    # Save archive log
    if archived:
        log_path = archive_dir / 'archive_log.json'
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'files_archived': archived
        }
        
        # Append to existing log if it exists
        if log_path.exists():
            with open(log_path, 'r') as f:
                existing_log = json.load(f)
            if isinstance(existing_log, list):
                existing_log.append(log_data)
            else:
                existing_log = [existing_log, log_data]
            log_data = existing_log
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    return archived

def rename_clean_versions(root_dir):
    """Rename the 'optimized' versions to clean names"""
    renames = {
        # Keep preprocess_optimized.py as the main preprocessing
        # 'ssmu_net/preprocess_optimized.py': 'ssmu_net/preprocess.py',  # Actually keep both
        
        # Scripts that should have cleaner names
        'scripts/preprocess_optimized_with_stats.py': 'scripts/preprocess_data.py',
        'scripts/generate_preprocessing_report.py': 'scripts/analyze_preprocessing.py',
    }
    
    renamed = []
    for old_name, new_name in renames.items():
        old_path = Path(root_dir) / old_name
        new_path = Path(root_dir) / new_name
        
        if old_path.exists() and not new_path.exists():
            try:
                old_path.rename(new_path)
                renamed.append(f"{old_name} → {new_name}")
                print(f"  Renamed: {old_name} → {new_name}")
            except Exception as e:
                print(f"  Error renaming {old_name}: {e}")
    
    return renamed

def create_project_structure_doc(root_dir):
    """Create documentation of the clean project structure"""
    
    structure_doc = """# Project Structure (After Cleanup)

## Main Directories

### `/ssmu_net/` - Core Library
- `config.yaml` - Main SSMU-Net configuration
- `config_optimized.yaml` - Configuration with optimized preprocessing
- `preprocess_optimized.py` - Optimized preprocessing pipeline
- `models.py` - SSMU-Net model implementation
- `bench_models.py` - Benchmark model implementations
- `data.py` - Data loading and dataset classes
- `losses.py` - Loss functions
- `evaluation_metrics.py` - Evaluation metrics

### `/scripts/` - Executable Scripts
- `preprocess_data.py` - Main preprocessing script with statistics
- `train_benchmark.py` - Training script for benchmark models
- `visualize_training_data.py` - Data visualization with annotations
- `analyze_preprocessing.py` - Preprocessing comparison analysis
- `cleanup_project.py` - This cleanup script

### `/outputs/` - Generated Data
- `npz/` - Original preprocessed data (15% tissue)
- `npz_optimized/` - Optimized preprocessed data (60% tissue)
- `models/` - Trained model checkpoints
- `figures/` - Generated figures
- `logs/` - Training logs
- `training_visualization/` - Data visualization outputs

### `/archive/` - Archived Files
Contains superseded versions and experimental code for reference.

## Key Improvements Made
1. **Preprocessing**: 4x more training data (60% vs 15% tissue)
2. **Organization**: Clear separation of active vs archived code
3. **Naming**: Removed confusing "fixed", "new", "optimized" suffixes where possible
4. **Documentation**: Clear structure and purpose for each component
"""
    
    with open(Path(root_dir) / 'PROJECT_STRUCTURE.md', 'w') as f:
        f.write(structure_doc)
    
    print(f"  Created: PROJECT_STRUCTURE.md")

def main():
    root_dir = Path.cwd()
    
    print("=" * 60)
    print("PROJECT CLEANUP AND ORGANIZATION")
    print("=" * 60)
    
    # 1. Remove macOS files
    print("\n1. Removing macOS metadata files...")
    removed_count = remove_macos_files(root_dir)
    print(f"   Removed {removed_count} macOS files")
    
    # 2. Create archive structure
    print("\n2. Creating archive structure...")
    archive_dir = create_archive_structure(root_dir)
    print(f"   Created archive at: {archive_dir}")
    
    # 3. Identify and archive old files
    print("\n3. Archiving superseded files...")
    files_to_archive = identify_files_to_archive()
    archived = archive_files(root_dir, files_to_archive, archive_dir)
    print(f"   Archived {len(archived)} files")
    
    # 4. Rename files to cleaner names
    print("\n4. Renaming files to cleaner names...")
    renamed = rename_clean_versions(root_dir)
    print(f"   Renamed {len(renamed)} files")
    
    # 5. Create project structure documentation
    print("\n5. Creating project documentation...")
    create_project_structure_doc(root_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("CLEANUP COMPLETE")
    print("=" * 60)
    print(f"✅ Removed {removed_count} macOS files")
    print(f"📦 Archived {len(archived)} old/superseded files")
    print(f"✏️  Renamed {len(renamed)} files to cleaner names")
    print(f"📄 Created PROJECT_STRUCTURE.md documentation")
    
    print("\nProject is now organized and clean!")
    print("Run this script periodically to keep the project tidy.")
    
    # Create a simple cleanup script for regular use
    quick_cleanup = """#!/bin/bash
# Quick cleanup script for macOS files
find . -name "._*" -type f -delete
find . -name ".DS_Store" -type f -delete
echo "Removed macOS metadata files"
"""
    
    with open(root_dir / 'clean_macos.sh', 'w') as f:
        f.write(quick_cleanup)
    os.chmod(root_dir / 'clean_macos.sh', 0o755)
    print("\n💡 Tip: Use './clean_macos.sh' for quick macOS file cleanup")

if __name__ == "__main__":
    main()