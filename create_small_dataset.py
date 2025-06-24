#!/usr/bin/env python3
"""
Create smaller, more manageable datasets from the full QMugs dataset.
This script provides several strategies for creating training subsets that are
faster to process while maintaining statistical representativeness.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import argparse
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

class SmallDatasetCreator:
    def __init__(self, data_dir="data/qmugs", output_dir="data/qmugs_small"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
    def load_metadata(self):
        """Load metadata for all splits"""
        metadata = {}
        for split in ['train', 'val', 'test']:
            meta_file = self.data_dir / f"{split}_metadata.csv"
            if meta_file.exists():
                metadata[split] = pd.read_csv(meta_file)
                print(f"Loaded {len(metadata[split])} {split} samples")
            else:
                print(f"Warning: {meta_file} not found")
        return metadata
    
    def strategy_random_sample(self, metadata, train_size=10000, val_size=2000, test_size=5000):
        """Strategy 1: Random sampling from each split"""
        print(f"\n=== Random Sampling Strategy ===")
        print(f"Target sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        sampled = {}
        for split, size in [('train', train_size), ('val', val_size), ('test', test_size)]:
            if split in metadata and len(metadata[split]) >= size:
                sampled[split] = metadata[split].sample(n=size, random_state=42)
                print(f"Sampled {len(sampled[split])} from {split} split")
            else:
                print(f"Warning: Not enough data in {split} split")
                
        return sampled

    def strategy_stratified_molecular(self, metadata, train_size=10000, val_size=2000, test_size=5000):
        """Strategy 2: Stratified sampling by molecular diversity"""
        print(f"\n=== Stratified Molecular Diversity Strategy ===")
        
        # Combine all data to analyze molecular diversity
        all_data = []
        for split, df in metadata.items():
            df_copy = df.copy()
            df_copy['original_split'] = split
            all_data.append(df_copy)
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # Group by SMILES to understand molecular diversity
        if 'nonunique_smiles' in combined.columns:
            smiles_groups = combined.groupby('nonunique_smiles')
            print(f"Found {len(smiles_groups)} unique molecules")
            
            # Sample molecules (not conformers) proportionally
            unique_molecules = list(smiles_groups.groups.keys())
            total_target = train_size + val_size + test_size
            
            # Sample molecules
            n_molecules = min(len(unique_molecules), total_target // 3)  # Assume ~3 conformers per molecule on average
            sampled_molecules = random.sample(unique_molecules, n_molecules)
            
            # Get all conformers for sampled molecules
            sampled_data = combined[combined['nonunique_smiles'].isin(sampled_molecules)]
            
            # Re-split the sampled data
            train_frac = train_size / total_target
            val_frac = val_size / total_target
            test_frac = test_size / total_target
            
            # Split by molecules to avoid data leakage
            mol_train, mol_temp = train_test_split(sampled_molecules, 
                                                  test_size=(val_frac + test_frac), 
                                                  random_state=42)
            mol_val, mol_test = train_test_split(mol_temp, 
                                                test_size=(test_frac / (val_frac + test_frac)), 
                                                random_state=42)
            
            sampled = {
                'train': sampled_data[sampled_data['nonunique_smiles'].isin(mol_train)].sample(n=min(len(sampled_data[sampled_data['nonunique_smiles'].isin(mol_train)]), train_size), random_state=42),
                'val': sampled_data[sampled_data['nonunique_smiles'].isin(mol_val)].sample(n=min(len(sampled_data[sampled_data['nonunique_smiles'].isin(mol_val)]), val_size), random_state=42),
                'test': sampled_data[sampled_data['nonunique_smiles'].isin(mol_test)].sample(n=min(len(sampled_data[sampled_data['nonunique_smiles'].isin(mol_test)]), test_size), random_state=42)
            }
            
            for split in sampled:
                print(f"Stratified {split}: {len(sampled[split])} samples")
                
        else:
            print("No SMILES column found, falling back to random sampling")
            sampled = self.strategy_random_sample(metadata, train_size, val_size, test_size)
            
        return sampled

    def copy_sampled_files(self, sampled_metadata, strategy_name="small"):
        """Copy the sampled files to output directory"""
        output_strategy_dir = self.output_dir / strategy_name
        output_strategy_dir.mkdir(parents=True, exist_ok=True)
        
        total_files = sum(len(df) for df in sampled_metadata.values())
        print(f"\nCopying {total_files} files to {output_strategy_dir}")
        
        copied_counts = {}
        
        for split, df in sampled_metadata.items():
            split_dir = output_strategy_dir / split
            split_dir.mkdir(exist_ok=True)
            
            print(f"\nCopying {split} split ({len(df)} files)...")
            copied = 0
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split}"):
                if 'chembl_id' in row and 'conf_id' in row:
                    chembl_id = row['chembl_id']
                    conf_id = row['conf_id']
                    
                    # Source file in original split
                    source_file = self.data_dir / split / f"{chembl_id}_{conf_id}.sdf"
                    dest_file = split_dir / f"{chembl_id}_{conf_id}.sdf"
                    
                    if source_file.exists():
                        shutil.copy2(source_file, dest_file)
                        copied += 1
                    else:
                        print(f"Warning: Source file not found: {source_file}")
            
            copied_counts[split] = copied
            print(f"Copied {copied}/{len(df)} files for {split} split")
            
            # Save metadata
            metadata_file = output_strategy_dir / f"{split}_metadata.csv"
            df.to_csv(metadata_file, index=False)
            print(f"Saved metadata: {metadata_file}")
        
        # Create summary
        summary = {
            'strategy': strategy_name,
            'total_files': sum(copied_counts.values()),
            'splits': copied_counts,
            'source_dir': str(self.data_dir),
            'output_dir': str(output_strategy_dir)
        }
        
        print(f"\n=== Summary for {strategy_name} ===")
        print(f"Total files copied: {summary['total_files']}")
        for split, count in summary['splits'].items():
            print(f"{split}: {count} files")
        
        return summary
    
    def create_development_datasets(self):
        """Create multiple small datasets for development and testing"""
        print("Creating development datasets from full QMugs dataset...")
        
        # Load metadata
        metadata = self.load_metadata()
        if not metadata:
            print("Error: No metadata found. Make sure the full dataset is prepared.")
            return
        
        strategies = [
            # Ultra-small for quick testing
            {
                'name': 'tiny',
                'method': self.strategy_random_sample,
                'params': {'train_size': 1000, 'val_size': 200, 'test_size': 500},
                'description': 'Tiny dataset for rapid prototyping and debugging'
            },
            # Small for development
            {
                'name': 'small', 
                'method': self.strategy_random_sample,
                'params': {'train_size': 5000, 'val_size': 1000, 'test_size': 2000},
                'description': 'Small dataset for development and initial training'
            },
            # Medium for serious experiments
            {
                'name': 'medium',
                'method': self.strategy_random_sample, 
                'params': {'train_size': 25000, 'val_size': 5000, 'test_size': 10000},
                'description': 'Medium dataset for full experiments'
            },
            # Stratified version for molecular diversity
            {
                'name': 'stratified_small',
                'method': self.strategy_stratified_molecular,
                'params': {'train_size': 5000, 'val_size': 1000, 'test_size': 2000}, 
                'description': 'Small dataset with molecular diversity stratification'
            }
        ]
        
        results = {}
        
        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"Creating {strategy['name']} dataset")
            print(f"Description: {strategy['description']}")
            
            # Sample data using the specified strategy
            sampled = strategy['method'](metadata, **strategy['params'])
            
            if sampled:
                # Copy files
                summary = self.copy_sampled_files(sampled, strategy['name'])
                results[strategy['name']] = summary
            else:
                print(f"Failed to create {strategy['name']} dataset")
        
        # Print final summary
        print(f"\n{'='*60}")
        print("DATASET CREATION SUMMARY")
        print(f"{'='*60}")
        
        for name, summary in results.items():
            print(f"\n{name.upper()} ({summary['total_files']} files):")
            print(f"  Location: {summary['output_dir']}")
            for split, count in summary['splits'].items():
                print(f"  {split}: {count:,} files")
        
        print(f"\nAll datasets created in: {self.output_dir}")
        print("\nTo use a dataset, update your config.py:")
        print("  data_dir = 'data/qmugs_small/tiny'  # or small, medium, etc.")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Create smaller datasets from QMugs")
    parser.add_argument("--input-dir", default="data/qmugs", 
                       help="Input directory with full dataset")
    parser.add_argument("--output-dir", default="data/qmugs_small",
                       help="Output directory for small datasets")
    parser.add_argument("--strategy", choices=['tiny', 'small', 'medium', 'stratified', 'all'],
                       default='all', help="Which dataset size to create")
    
    args = parser.parse_args()
    
    creator = SmallDatasetCreator(args.input_dir, args.output_dir)
    
    if args.strategy == 'all':
        creator.create_development_datasets()
    else:
        # Create specific strategy
        metadata = creator.load_metadata()
        if args.strategy == 'tiny':
            sampled = creator.strategy_random_sample(metadata, 1000, 200, 500)
        elif args.strategy == 'small':
            sampled = creator.strategy_random_sample(metadata, 5000, 1000, 2000)
        elif args.strategy == 'medium':
            sampled = creator.strategy_random_sample(metadata, 25000, 5000, 10000)
        elif args.strategy == 'stratified':
            sampled = creator.strategy_stratified_molecular(metadata, 5000, 1000, 2000)
        
        if sampled:
            creator.copy_sampled_files(sampled, args.strategy)

if __name__ == "__main__":
    main()