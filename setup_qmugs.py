#!/usr/bin/env python3
"""
QMugs Dataset Setup Script
Run this on your vast.ai instance to download and prepare the QMugs dataset.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0 and check:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    
    return result

def main():
    print("Setting up QMugs dataset...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Check if files already exist
    structures_file = data_dir / "structures.tar.gz"
    summary_file = data_dir / "summary.csv"
    extracted_dir = data_dir / "extracted_structures"
    qmugs_dir = data_dir / "qmugs"
    
    # Download structures.tar.gz if not exists
    if structures_file.exists():
        print("\n1. structures.tar.gz already exists, skipping download")
    else:
        print("\n1. Downloading structures.tar.gz...")
        structures_cmd = (
            'wget --content-disposition -P data '
            '"https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download?path=%2F&files=structures.tar.gz"'
        )
        run_command(structures_cmd)
    
    # Download summary.csv if not exists
    if summary_file.exists():
        print("\n2. summary.csv already exists, skipping download")
    else:
        print("\n2. Downloading summary.csv...")
        summary_cmd = (
            'wget --content-disposition -P data '
            '"https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download?path=%2F&files=summary.csv"'
        )
        run_command(summary_cmd)
    
    # Extract structures if not already extracted
    if extracted_dir.exists() and any(extracted_dir.iterdir()):
        print("\n3. Structures already extracted, skipping extraction")
    else:
        print("\n3. Extracting structures...")
        run_command("python prepare_dataset.py --extract")
    
    # Create splits if not already created
    if qmugs_dir.exists() and (qmugs_dir / "train").exists():
        print("\n4. Train/val/test splits already exist, skipping split creation")
        print("Existing splits found:")
        for split in ['train', 'val', 'test']:
            split_dir = qmugs_dir / split
            if split_dir.exists():
                result = run_command(f"find {split_dir} -name '*.sdf' | wc -l", check=False)
                print(f"  {split}: {result.stdout.strip()} SDF files")
    else:
        print("\n4. Creating train/val/test splits...")
        run_command("python prepare_dataset.py")
    
    print("\nSetup complete!")
    print("Dataset structure:")
    run_command("find data/qmugs -type d", check=False)
    
    print("\nFile counts:")
    for split in ['train', 'val', 'test']:
        result = run_command(f"find data/qmugs/{split} -name '*.sdf' | wc -l", check=False)
        print(f"{split}: {result.stdout.strip()} SDF files")
    
    print("\nReady to train! Run: python train.py")

if __name__ == "__main__":
    main()