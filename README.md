# Quantum GNN for QMugs Dataset

A Graph Neural Network implementation for predicting quantum mechanical properties of drug-like molecules using the QMugs dataset.

## Overview

This project implements a quantum-aware Graph Neural Network (GNN) that takes molecular conformer geometries (from SDF files) and predicts various quantum mechanical properties including:

- HOMO energy
- LUMO energy  
- HOMO-LUMO gap
- Dipole moment
- Polarizability
- Electronic energy
- Zero-point energy

## Features

- **3D-aware GNN**: Incorporates 3D positional information from conformer geometries
- **Transformer-based architecture**: Uses Graph Transformer layers for better long-range interactions
- **Multi-head attention**: Enhanced with self-attention mechanisms
- **Comprehensive evaluation**: Includes MAE, R² metrics for each property
- **Flexible data loading**: Handles SDF files with quantum properties
- **Training utilities**: Tensorboard logging, checkpointing, visualization

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Structure

Your QMugs dataset should be organized as follows:

```
data/qmugs/
├── train/
│   ├── molecule1.sdf
│   ├── molecule2.sdf
│   └── ...
├── val/
│   ├── molecule1.sdf
│   └── ...
└── test/
    ├── molecule1.sdf
    └── ...
```

Each SDF file should contain:
- 3D conformer geometry
- Quantum properties as SDF properties or separate JSON files

## Usage

### Training

```bash
python train.py
```

### Configuration

Modify `config.py` to adjust:
- Model architecture (hidden dimensions, layers, attention heads)
- Training parameters (batch size, learning rate, epochs)
- Target properties to predict
- Hardware settings

### Key Configuration Options

```python
# Model architecture
hidden_dim = 128          # Hidden layer dimensions
num_layers = 6           # Number of GNN layers
num_heads = 8            # Multi-head attention heads
dropout = 0.1            # Dropout rate

# Training
batch_size = 64          # Batch size
learning_rate = 1e-4     # Learning rate
num_epochs = 100         # Training epochs

# Target properties
target_properties = [
    "homo_energy",
    "lumo_energy", 
    "gap_energy",
    "dipole_moment",
    "polarizability",
    "electronic_energy",
    "zero_point_energy"
]
```

## Model Architecture

The model combines several advanced techniques:

1. **Node Embeddings**: Atomic features (atomic number, degree, charge, etc.)
2. **Edge Embeddings**: Bond features (bond type, aromaticity, etc.)  
3. **3D Positional Encoding**: Sinusoidal encoding of 3D coordinates
4. **Graph Transformer Layers**: Multi-head attention over molecular graphs
5. **Self-Attention**: Global molecular interactions
6. **Multi-scale Pooling**: Mean, max, and sum pooling for graph representation
7. **MLP Readout**: Final prediction layers

## Output

The training produces:
- Model checkpoints in `checkpoints/`
- Training logs in `logs/`
- Tensorboard visualizations
- Test results with detailed metrics
- Prediction analysis plots

## Monitoring

- **Tensorboard**: `tensorboard --logdir logs/`
- **Wandb**: Automatic logging if available
- **Metrics**: MAE and R² for each quantum property

## Files

- `train.py`: Main training script
- `model.py`: GNN model implementations
- `data_loader.py`: Dataset and data loading utilities
- `config.py`: Configuration settings
- `utils.py`: Analysis and visualization utilities
- `requirements.txt`: Python dependencies

## Tips for Training

1. **GPU Setup**: Ensure CUDA is available for faster training
2. **Memory**: Adjust batch size based on GPU memory
3. **Convergence**: Monitor validation loss for early stopping
4. **Hyperparameters**: Tune learning rate and architecture size
5. **Data Quality**: Ensure SDF files have proper 3D coordinates and properties

## Troubleshooting

- **Memory Issues**: Reduce batch size or model dimensions
- **Slow Training**: Increase num_workers in data loader
- **Poor Performance**: Check data quality and increase model capacity
- **NaN Losses**: Reduce learning rate or add gradient clipping

## Citation

If you use this code, please cite the QMugs dataset paper and this implementation.