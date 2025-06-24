#!/usr/bin/env python3
"""
Quick configuration templates for different dataset sizes and training scenarios.
This allows easy switching between full dataset and smaller development datasets.
"""

import torch

class BaseConfig:
    # Target properties (common across all configs)
    target_properties = [
        "homo_energy",
        "lumo_energy", 
        "gap_energy",
        "dipole_moment",
        "polarizability",
        "electronic_energy",
        "zero_point_energy"
    ]
    
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4
    
    # Logging
    log_dir = "logs"
    checkpoint_dir = "checkpoints"
    save_every = 10

class TinyConfig(BaseConfig):
    """Ultra-fast config for debugging and rapid prototyping"""
    # Dataset
    data_dir = "data/qmugs_small/tiny"  # 1K train, 200 val, 500 test
    
    # Model (smaller for speed)
    hidden_dim = 64
    num_layers = 3
    num_heads = 4
    dropout = 0.1
    
    # Training (fast iterations)
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 20
    weight_decay = 1e-5

class SmallConfig(BaseConfig):
    """Development config for initial experiments"""
    # Dataset
    data_dir = "data/qmugs_small/small"  # 5K train, 1K val, 2K test
    
    # Model
    hidden_dim = 96
    num_layers = 4
    num_heads = 6
    dropout = 0.1
    
    # Training
    batch_size = 48
    learning_rate = 5e-4
    num_epochs = 50
    weight_decay = 1e-5

class MediumConfig(BaseConfig):
    """Serious experimentation config"""
    # Dataset
    data_dir = "data/qmugs_small/medium"  # 25K train, 5K val, 10K test
    
    # Model
    hidden_dim = 128
    num_layers = 6
    num_heads = 8
    dropout = 0.1
    
    # Training
    batch_size = 64
    learning_rate = 2e-4
    num_epochs = 100
    weight_decay = 1e-5

class FullConfig(BaseConfig):
    """Full dataset config for final training"""
    # Dataset
    data_dir = "data/qmugs"  # Full dataset: 1.4M train, 200K val, 400K test
    
    # Model (full capacity)
    hidden_dim = 128
    num_layers = 6
    num_heads = 8
    dropout = 0.1
    
    # Training (conservative for large dataset)
    batch_size = 64
    learning_rate = 1e-4
    num_epochs = 200
    weight_decay = 1e-5

# Easy config selection
CONFIGS = {
    'tiny': TinyConfig,
    'small': SmallConfig,
    'medium': MediumConfig,
    'full': FullConfig
}

def get_config(size='small'):
    """Get configuration for specified dataset size
    
    Args:
        size: 'tiny', 'small', 'medium', or 'full'
    
    Returns:
        Config class instance
    """
    if size not in CONFIGS:
        print(f"Unknown config size '{size}'. Available: {list(CONFIGS.keys())}")
        print("Defaulting to 'small' config")
        size = 'small'
    
    config_class = CONFIGS[size]
    config = config_class()
    
    print(f"Using {size.upper()} configuration:")
    print(f"  Dataset: {config.data_dir}")
    print(f"  Model: {config.hidden_dim}D, {config.num_layers} layers")
    print(f"  Training: {config.batch_size} batch, {config.num_epochs} epochs")
    print(f"  Device: {config.device}")
    
    return config

# For backward compatibility, default to SmallConfig
Config = SmallConfig