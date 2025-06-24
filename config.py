import torch

class Config:
    # Dataset
    data_dir = "data/qmugs"  # Use the prepared dataset
    target_properties = [
        "homo_energy",
        "lumo_energy", 
        "gap_energy",
        "dipole_moment",
        "polarizability",
        "electronic_energy",
        "zero_point_energy"
    ]
    
    # Model
    hidden_dim = 128
    num_layers = 6
    num_heads = 8
    dropout = 0.1
    
    # Training
    batch_size = 64
    learning_rate = 1e-4
    num_epochs = 100
    weight_decay = 1e-5
    
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4
    
    # Logging
    log_dir = "logs"
    checkpoint_dir = "checkpoints"
    save_every = 10