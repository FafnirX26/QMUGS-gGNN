import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import tarfile
from tqdm import tqdm
import argparse
from pathlib import Path

class QMugsDatasetPreparator:
    def __init__(self, data_dir="data", output_dir="data/qmugs"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.structures_file = self.data_dir / "structures.tar.gz"
        self.summary_file = self.data_dir / "summary.csv"
        
    def download_data(self):
        """Download QMugs dataset files"""
        print("Downloading QMugs dataset...")
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download commands
        structures_cmd = (
            'wget --content-disposition -P {} '
            '"https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download?path=%2F&files=structures.tar.gz"'
        ).format(self.data_dir)
        
        summary_cmd = (
            'wget --content-disposition -P {} '
            '"https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download?path=%2F&files=summary.csv"'
        ).format(self.data_dir)
        
        print("Run these commands to download the data:")
        print(f"1. {structures_cmd}")
        print(f"2. {summary_cmd}")
        print("\nThen run this script again with --extract flag")
        
    def extract_structures(self):
        """Extract structures.tar.gz"""
        print("Extracting structures...")
        
        if not self.structures_file.exists():
            raise FileNotFoundError(f"structures.tar.gz not found at {self.structures_file}")
        
        # Extract to temporary directory
        extract_dir = self.data_dir / "extracted_structures"
        extract_dir.mkdir(exist_ok=True)
        
        with tarfile.open(self.structures_file, 'r:gz') as tar:
            tar.extractall(extract_dir)
        
        print(f"Structures extracted to {extract_dir}")
        return extract_dir
    
    def load_summary(self):
        """Load and analyze summary.csv"""
        print("Loading summary.csv...")
        
        if not self.summary_file.exists():
            raise FileNotFoundError(f"summary.csv not found at {self.summary_file}")
        
        df = pd.read_csv(self.summary_file)
        print(f"Loaded {len(df)} molecules from summary.csv")
        print(f"Columns: {list(df.columns)}")
        
        # Check for duplicate SMILES
        if 'nonunique_smiles' in df.columns:
            unique_smiles = df['nonunique_smiles'].nunique()
            print(f"Unique SMILES: {unique_smiles}")
            print(f"Total molecules: {len(df)}")
            print(f"Molecules with duplicate SMILES: {len(df) - unique_smiles}")
        
        return df
    
    def create_splits(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """Create train/val/test splits avoiding SMILES leakage"""
        print("Creating data splits...")
        
        # Check if we should split by SMILES or by individual conformers
        if 'nonunique_smiles' in df.columns or 'smiles' in df.columns:
            smiles_col = 'nonunique_smiles' if 'nonunique_smiles' in df.columns else 'smiles'
            unique_smiles = df[smiles_col].unique()
            
            print(f"Found {len(unique_smiles)} unique SMILES in {len(df)} total conformers")
            
            # If we have a reasonable number of unique SMILES (>10), split by SMILES
            if len(unique_smiles) > 10:
                print("Splitting by unique SMILES to avoid data leakage")
                
                # Split unique SMILES
                smiles_train, smiles_temp = train_test_split(
                    unique_smiles, test_size=(test_size + val_size), random_state=random_state
                )
                
                smiles_val, smiles_test = train_test_split(
                    smiles_temp, test_size=(test_size / (test_size + val_size)), random_state=random_state
                )
                
                # Get molecules for each split
                train_df = df[df[smiles_col].isin(smiles_train)]
                val_df = df[df[smiles_col].isin(smiles_val)]
                test_df = df[df[smiles_col].isin(smiles_test)]
                
            else:
                print(f"Too few unique SMILES ({len(unique_smiles)}), splitting by conformers instead")
                # Split by individual conformers/molecules
                train_df, temp_df = train_test_split(df, test_size=(test_size + val_size), random_state=random_state)
                val_df, test_df = train_test_split(temp_df, test_size=(test_size / (test_size + val_size)), random_state=random_state)
            
        else:
            print("No SMILES column found, splitting randomly by conformers")
            train_df, temp_df = train_test_split(df, test_size=(test_size + val_size), random_state=random_state)
            val_df, test_df = train_test_split(temp_df, test_size=(test_size / (test_size + val_size)), random_state=random_state)
        
        print(f"Train set: {len(train_df)} conformers")
        print(f"Val set: {len(val_df)} conformers") 
        print(f"Test set: {len(test_df)} conformers")
        
        return train_df, val_df, test_df
    
    def organize_files(self, train_df, val_df, test_df, structures_dir, max_files_per_split=None):
        """Organize SDF files into train/val/test directories"""
        print("Organizing files into splits...")
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
        
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        # Apply limits if specified
        if max_files_per_split:
            print(f"Limiting each split to {max_files_per_split} files for faster processing")
            for split_name in splits:
                if len(splits[split_name]) > max_files_per_split:
                    splits[split_name] = splits[split_name].sample(n=max_files_per_split, random_state=42)
                    print(f"Sampled {split_name} to {len(splits[split_name])} files")
        
        # Find all SDF files in extracted directory for verification
        sdf_files = list(structures_dir.rglob("*.sdf"))
        print(f"Found {len(sdf_files)} SDF files in extracted structures")
        
        # Copy files to appropriate splits with parallel processing
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def copy_file(args):
            row, split_dir, structures_dir = args
            
            if 'chembl_id' in row and 'conf_id' in row:
                chembl_id = row['chembl_id']
                conf_id = row['conf_id']
                
                source_path = structures_dir / chembl_id / f"{conf_id}.sdf"
                if source_path.exists():
                    dest_filename = f"{chembl_id}_{conf_id}.sdf"
                    dest_path = split_dir / dest_filename
                    shutil.copy2(source_path, dest_path)
                    return True
            return False
        
        for split_name, split_df in splits.items():
            split_dir = self.output_dir / split_name
            print(f"\nProcessing {split_name} split ({len(split_df)} files)...")
            
            # Prepare arguments for parallel processing
            args_list = [(row, split_dir, structures_dir) for _, row in split_df.iterrows()]
            
            # Use ThreadPoolExecutor for I/O bound operations
            copied_files = 0
            with ThreadPoolExecutor(max_workers=min(8, mp.cpu_count())) as executor:
                futures = [executor.submit(copy_file, args) for args in args_list]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Copying {split_name}"):
                    if future.result():
                        copied_files += 1
            
            print(f"Copied {copied_files} files to {split_name} split")
        
        # Save split metadata
        for split_name, split_df in splits.items():
            metadata_file = self.output_dir / f"{split_name}_metadata.csv"
            split_df.to_csv(metadata_file, index=False)
            print(f"Saved {split_name} metadata to {metadata_file}")
    
    def prepare_dataset(self, extract=False, create_splits=True):
        """Main function to prepare the dataset"""
        print("Preparing QMugs dataset...")
        
        if extract:
            # Extract structures
            structures_dir = self.extract_structures()
        else:
            structures_dir = self.data_dir / "extracted_structures"
            if not structures_dir.exists():
                print("Extracted structures not found. Run with --extract flag first.")
                return
        
        if create_splits:
            # Load summary
            df = self.load_summary()
            
            # Create splits
            train_df, val_df, test_df = self.create_splits(df)
            
            # Organize files
            self.organize_files(train_df, val_df, test_df, structures_dir)
            
            print(f"\nDataset preparation complete!")
            print(f"Output directory: {self.output_dir}")
            print("Directory structure:")
            print(f"  {self.output_dir}/train/")
            print(f"  {self.output_dir}/val/")
            print(f"  {self.output_dir}/test/")
            print(f"  {self.output_dir}/*_metadata.csv")

def main():
    parser = argparse.ArgumentParser(description="Prepare QMugs dataset")
    parser.add_argument("--data-dir", default="data", help="Directory containing downloaded files")
    parser.add_argument("--output-dir", default="data/qmugs", help="Output directory for organized dataset")
    parser.add_argument("--download", action="store_true", help="Show download commands")
    parser.add_argument("--extract", action="store_true", help="Extract structures.tar.gz")
    parser.add_argument("--no-splits", action="store_true", help="Skip creating splits")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (default: 0.2)")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation set size (default: 0.1)")
    
    args = parser.parse_args()
    
    preparator = QMugsDatasetPreparator(args.data_dir, args.output_dir)
    
    if args.download:
        preparator.download_data()
    else:
        preparator.prepare_dataset(
            extract=args.extract,
            create_splits=not args.no_splits
        )

if __name__ == "__main__":
    main()