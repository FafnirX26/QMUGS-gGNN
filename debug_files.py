#!/usr/bin/env python3
"""
Debug script to understand QMugs file structure
"""

import os
import pandas as pd
from pathlib import Path

def main():
    print("Debugging QMugs file structure...")
    
    # Check extracted structures
    extracted_dir = Path("data/extracted_structures")
    print(f"\n1. Checking extracted structures in {extracted_dir}")
    
    if extracted_dir.exists():
        # Find all SDF files
        sdf_files = list(extracted_dir.rglob("*.sdf"))
        print(f"Found {len(sdf_files)} SDF files")
        
        if sdf_files:
            print("\nFirst 10 SDF filenames:")
            for i, sdf_file in enumerate(sdf_files[:10]):
                print(f"  {sdf_file.name}")
            
            print(f"\nDirectory structure in {extracted_dir}:")
            for root, dirs, files in os.walk(extracted_dir):
                level = root.replace(str(extracted_dir), '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                if level < 3:  # Don't go too deep
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:5]:  # Show first 5 files
                        print(f"{subindent}{file}")
                    if len(files) > 5:
                        print(f"{subindent}... and {len(files) - 5} more files")
    else:
        print(f"{extracted_dir} does not exist!")
    
    # Check summary.csv
    print("\n2. Checking summary.csv structure")
    summary_path = Path("data/summary.csv")
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        print(f"Summary has {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        print("\nFirst few rows:")
        print(df.head())
        
        # Check for unique values
        if 'chembl_id' in df.columns:
            print(f"\nUnique chembl_ids: {df['chembl_id'].nunique()}")
            print(f"Example chembl_ids: {df['chembl_id'].head().tolist()}")
        
        if 'conf_id' in df.columns:
            print(f"Unique conf_ids: {df['conf_id'].nunique()}")
            print(f"Example conf_ids: {df['conf_id'].head().tolist()}")
    else:
        print(f"{summary_path} does not exist!")
    
    # Check what splits were created
    print("\n3. Checking created splits")
    splits_dir = Path("data/qmugs")
    if splits_dir.exists():
        for split in ['train', 'val', 'test']:
            metadata_file = splits_dir / f"{split}_metadata.csv"
            if metadata_file.exists():
                split_df = pd.read_csv(metadata_file)
                print(f"\n{split} metadata: {len(split_df)} entries")
                if len(split_df) > 0:
                    print(f"Columns: {list(split_df.columns)}")
                    if 'chembl_id' in split_df.columns and 'conf_id' in split_df.columns:
                        print(f"Example filenames would be:")
                        for i in range(min(3, len(split_df))):
                            row = split_df.iloc[i]
                            print(f"  {row['chembl_id']}_{row['conf_id']}.sdf")

if __name__ == "__main__":
    main()