import os
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from typing import List, Dict, Optional, Tuple
import json

class QMugsDataset(Dataset):
    def __init__(self, root: str, split: str = 'train', target_properties: List[str] = None, summary_df: pd.DataFrame = None):
        self.root = root
        self.split = split
        self.target_properties = target_properties or [
            "homo_energy", "lumo_energy", "gap_energy", "dipole_moment", 
            "polarizability", "electronic_energy", "zero_point_energy"
        ]
        
        # Load metadata
        self.metadata_path = os.path.join(root, f"{split}_metadata.csv")
        if os.path.exists(self.metadata_path):
            self.metadata = pd.read_csv(self.metadata_path)
        else:
            self.metadata = self._create_metadata()
        
        # Load summary data if provided (contains quantum properties)
        self.summary_df = summary_df
        if summary_df is not None:
            # Create mapping from molecule ID to properties
            self.property_mapping = self._create_property_mapping(summary_df)
        else:
            self.property_mapping = {}
            
        super().__init__(root)
    
    def _create_metadata(self):
        """Create metadata from SDF files if not exists"""
        sdf_dir = os.path.join(self.root, self.split)
        metadata = []
        
        for filename in os.listdir(sdf_dir):
            if filename.endswith('.sdf'):
                filepath = os.path.join(sdf_dir, filename)
                mol_id = filename.replace('.sdf', '')
                metadata.append({
                    'mol_id': mol_id,
                    'filepath': filepath
                })
        
        df = pd.DataFrame(metadata)
        df.to_csv(self.metadata_path, index=False)
        return df
    
    def len(self):
        return len(self.metadata)
    
    def get(self, idx):
        row = self.metadata.iloc[idx]
        mol_id = row['mol_id']
        filepath = row['filepath']
        
        # Load molecule from SDF
        mol = self._load_molecule(filepath)
        if mol is None:
            return None
            
        # Convert to graph
        data = self._mol_to_graph(mol)
        
        # Add target properties
        targets = self._load_targets(mol_id, mol)
        if targets is not None:
            data.y = torch.tensor(targets, dtype=torch.float)
        
        data.mol_id = mol_id
        return data
    
    def _load_molecule(self, filepath: str):
        """Load molecule from SDF file"""
        try:
            supplier = Chem.SDMolSupplier(filepath, removeHs=False)
            mol = next(supplier)
            return mol
        except:
            return None
    
    def _mol_to_graph(self, mol):
        """Convert RDKit molecule to PyTorch Geometric graph"""
        # Node features (atoms)
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
                atom.GetTotalValence(),
                atom.GetMass()
            ]
            atom_features.append(features)
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Edge features (bonds)
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions
            edge_indices.extend([[i, j], [j, i]])
            
            bond_features = [
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsAromatic()),
                int(bond.IsInRing()),
                bond.GetBondDir()
            ]
            edge_features.extend([bond_features, bond_features])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # 3D coordinates
        conformer = mol.GetConformer()
        pos = torch.tensor([[conformer.GetAtomPosition(i).x,
                           conformer.GetAtomPosition(i).y, 
                           conformer.GetAtomPosition(i).z] 
                          for i in range(mol.GetNumAtoms())], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    
    def _create_property_mapping(self, summary_df: pd.DataFrame) -> Dict:
        """Create mapping from molecule ID to quantum properties"""
        property_mapping = {}
        
        # Map QMugs property names to our target properties
        qmugs_property_map = {
            "homo_energy": ["GFN2_HOMO_ENERGY", "DFT_HOMO_ENERGY", "HOMO", "homo"],
            "lumo_energy": ["GFN2_LUMO_ENERGY", "DFT_LUMO_ENERGY", "LUMO", "lumo"], 
            "gap_energy": ["GFN2_HOMO_LUMO_GAP", "DFT_HOMO_LUMO_GAP", "gap", "HOMO_LUMO_gap"],
            "dipole_moment": ["GFN2_DIPOLE_TOT", "DFT_DIPOLE_TOT", "dipole", "dipole_moment"],
            "polarizability": ["GFN2_POLARIZABILITY_MOLECULAR", "alpha", "polarizability"],
            "electronic_energy": ["GFN2_TOTAL_ENERGY", "DFT_TOTAL_ENERGY", "E_tot", "total_energy"],
            "zero_point_energy": ["GFN2_FORMATION_ENERGY", "DFT_FORMATION_ENERGY", "ZPE", "zero_point_energy"]
        }
        
        for idx, row in summary_df.iterrows():
            mol_properties = {}
            
            for target_prop in self.target_properties:
                value = None
                
                # Try different possible column names
                if target_prop in qmugs_property_map:
                    for possible_name in qmugs_property_map[target_prop]:
                        if possible_name in row and pd.notna(row[possible_name]):
                            value = float(row[possible_name])
                            break
                
                # Try exact match
                if value is None and target_prop in row and pd.notna(row[target_prop]):
                    value = float(row[target_prop])
                
                mol_properties[target_prop] = value if value is not None else 0.0
            
            # Create molecule ID from QMugs columns
            mol_id = None
            if 'chembl_id' in row and 'conf_id' in row:
                mol_id = f"{row['chembl_id']}_{row['conf_id']}"
            else:
                # Use different possible ID columns
                for id_col in ['id', 'mol_id', 'molecule', 'name', 'chembl_id']:
                    if id_col in row and pd.notna(row[id_col]):
                        mol_id = str(row[id_col])
                        break
            
            if mol_id is None:
                mol_id = str(idx)
            
            property_mapping[mol_id] = mol_properties
        
        return property_mapping
    
    def _load_targets(self, mol_id: str, mol) -> Optional[List[float]]:
        """Load target quantum properties"""
        targets = []
        
        # First try to load from property mapping (summary.csv data)
        if mol_id in self.property_mapping:
            props = self.property_mapping[mol_id]
            for prop_name in self.target_properties:
                targets.append(props.get(prop_name, 0.0))
            return targets
        
        # Try to load from SDF properties
        try:
            props = mol.GetPropsAsDict()
            for prop_name in self.target_properties:
                if prop_name in props:
                    targets.append(float(props[prop_name]))
                else:
                    targets.append(0.0)
            return targets
        except:
            pass
        
        # Try to load from separate JSON files
        json_path = os.path.join(self.root, f"{mol_id}_properties.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                props = json.load(f)
                for prop_name in self.target_properties:
                    targets.append(props.get(prop_name, 0.0))
                return targets
        
        return None

def create_data_loaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders"""
    
    # Load summary data if available
    summary_df = None
    summary_path = os.path.join(config.data_dir, "summary.csv")
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        print(f"Loaded summary.csv with {len(summary_df)} molecules")
    
    train_dataset = QMugsDataset(
        root=config.data_dir,
        split='train',
        target_properties=config.target_properties,
        summary_df=summary_df
    )
    
    val_dataset = QMugsDataset(
        root=config.data_dir,
        split='val',
        target_properties=config.target_properties,
        summary_df=summary_df
    )
    
    test_dataset = QMugsDataset(
        root=config.data_dir,
        split='test',
        target_properties=config.target_properties,
        summary_df=summary_df
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader