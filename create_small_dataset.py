#!/usr/bin/env python3
"""
Create a smaller, manageable subset of QMugs for faster training and iteration
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse

def create_small_dataset(sample_size=50000, seed=42):
    """Create a smaller dataset by sampling from the full QMugs dataset"""
    
    print(f"Creating small dataset with {sample_size} samples...")
    
    # Load full summary
    summary_path = Path("data/summary.csv")
    if not summary_path.exists():
        print("summary.csv not found!")
        return
    
    df = pd.read_csv(summary_path)
    print(f"Full dataset: {len(df)} conformers")
    
    # Sample strategically
    np.random.seed(seed)
    
    # Strategy 1: Random sampling
    if sample_size >= len(df):
        sampled_df = df.copy()
    else:
        sampled_df = df.sample(n=sample_size, random_state=seed)
    
    print(f"Sampled dataset: {len(sampled_df)} conformers")
    
    # Save sampled summary
    small_summary_path = Path("data/summary_small.csv")
    sampled_df.to_csv(small_summary_path, index=False)
    print(f"Saved small summary to {small_summary_path}")
    
    return sampled_df

def create_balanced_dataset(molecules_per_chembl=1, max_molecules=10000, seed=42):
    """Create a balanced dataset with limited conformers per molecule"""
    
    print(f"Creating balanced dataset with max {molecules_per_chembl} conformers per ChEMBL ID...")
    
    # Load full summary
    summary_path = Path("data/summary.csv")
    df = pd.read_csv(summary_path)
    
    # Group by chembl_id and sample
    np.random.seed(seed)
    sampled_groups = []
    
    unique_chembls = df['chembl_id'].unique()
    if max_molecules < len(unique_chembls):
        selected_chembls = np.random.choice(unique_chembls, max_molecules, replace=False)
    else:
        selected_chembls = unique_chembls
    
    for chembl_id in tqdm(selected_chembls, desc="Sampling molecules"):
        chembl_df = df[df['chembl_id'] == chembl_id]
        
        if len(chembl_df) <= molecules_per_chembl:
            sampled_groups.append(chembl_df)
        else:
            sampled = chembl_df.sample(n=molecules_per_chembl, random_state=seed)
            sampled_groups.append(sampled)
    
    sampled_df = pd.concat(sampled_groups, ignore_index=True)
    print(f"Balanced dataset: {len(sampled_df)} conformers from {len(selected_chembls)} molecules")
    
    # Save
    balanced_summary_path = Path("data/summary_balanced.csv")
    sampled_df.to_csv(balanced_summary_path, index=False)
    print(f"Saved balanced summary to {balanced_summary_path}")
    
    return sampled_df

def create_splits_fast(df, output_suffix="", test_size=0.2, val_size=0.1, seed=42):
    """Create splits faster with the smaller dataset"""
    from sklearn.model_selection import train_test_split
    
    print("Creating fast splits...")
    
    # Split by conformers (since we have a manageable dataset now)
    train_df, temp_df = train_test_split(df, test_size=(test_size + val_size), random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=(test_size / (test_size + val_size)), random_state=seed)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create output directory
    output_dir = Path(f"data/qmugs{output_suffix}")
    output_dir.mkdir(exist_ok=True)
    
    # Copy files efficiently
    structures_dir = Path("data/extracted_structures")
    
    splits = {
        'train': train_df,
        'val': val_df, 
        'test': test_df
    }
    
    for split_name, split_df in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        print(f"\nCopying {split_name} files...")
        copied = 0
        
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            chembl_id = row['chembl_id']
            conf_id = row['conf_id']
            
            # Try multiple possible paths for the source file
            possible_paths = [
                structures_dir / chembl_id / f"{conf_id}.sdf",
                structures_dir / "extracted_structures" / chembl_id / f"{conf_id}.sdf"
            ]
            
            # Also search recursively for the chembl_id directory
            for subdir in structures_dir.rglob(chembl_id):
                if subdir.is_dir():
                    possible_paths.append(subdir / f"{conf_id}.sdf")
            
            source_path = None
            for path in possible_paths:
                if path.exists():
                    source_path = path
                    break
            
            if source_path:
                dest_path = split_dir / f"{chembl_id}_{conf_id}.sdf"
                shutil.copy2(source_path, dest_path)
                copied += 1
        
        print(f"Copied {copied} files to {split_name}")
        
        # Save metadata
        metadata_path = output_dir / f"{split_name}_metadata.csv"
        split_df.to_csv(metadata_path, index=False)
    
    print(f"\nSmall dataset ready in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Create smaller QMugs dataset for faster training")
    parser.add_argument("--strategy", choices=["random", "balanced"], default="balanced",
                       help="Sampling strategy")
    parser.add_argument("--size", type=int, default=50000,
                       help="Sample size for random strategy")
    parser.add_argument("--molecules", type=int, default=10000,
                       help="Max molecules for balanced strategy")
    parser.add_argument("--conformers", type=int, default=1,
                       help="Conformers per molecule for balanced strategy")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    if args.strategy == "random":
        print(f"Using random sampling strategy with {args.size} samples")
        df = create_small_dataset(args.size, args.seed)
        suffix = f"_random_{args.size}"
    else:
        print(f"Using balanced sampling: {args.conformers} conformers from {args.molecules} molecules")
        df = create_balanced_dataset(args.conformers, args.molecules, args.seed)
        suffix = f"_balanced_{args.molecules}mol_{args.conformers}conf"
    
    # Create splits
    create_splits_fast(df, suffix, seed=args.seed)

if __name__ == "__main__":
    main()