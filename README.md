# Quantum Graph Neural Network for QMugs Dataset

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A state-of-the-art Graph Neural Network implementation for predicting quantum mechanical properties of drug-like molecules using the QMugs dataset. This project combines 3D molecular geometry with transformer-based graph neural networks to achieve accurate quantum property predictions.

## 🎯 Overview

This implementation features an advanced **Quantum-aware Graph Neural Network** that processes molecular conformer geometries from SDF files and predicts multiple quantum mechanical properties simultaneously:

- **HOMO Energy** - Highest Occupied Molecular Orbital energy
- **LUMO Energy** - Lowest Unoccupied Molecular Orbital energy  
- **HOMO-LUMO Gap** - Electronic band gap
- **Dipole Moment** - Molecular polarity measure
- **Polarizability** - Response to electric fields
- **Electronic Energy** - Total electronic energy
- **Zero-Point Energy** - Vibrational energy at 0K

## ✨ Key Features

### 🧠 Advanced Architecture
- **3D-Aware Processing**: Incorporates full 3D molecular geometry with sinusoidal positional encoding
- **Transformer-Based GNN**: Uses Graph Transformer layers for superior long-range molecular interactions
- **Multi-Head Self-Attention**: Enhanced global molecular representation learning
- **Multi-Scale Pooling**: Combines mean, max, and sum pooling for comprehensive graph representation

### 🚀 Performance Optimizations
- **Scalable Training**: Multiple dataset sizes from 1.7K to 2M molecules
- **Fast Iteration**: Optimized pipeline for rapid development and experimentation
- **Smart Sampling**: Preserves molecular diversity while reducing dataset size
- **Memory Efficient**: Optimized batch processing and gradient handling

### 📊 Comprehensive Evaluation
- **Multi-Property Metrics**: MAE and R² scores for each quantum property
- **Training Monitoring**: TensorBoard and Weights & Biases integration
- **Visualization Tools**: Training curves, prediction analysis, and performance plots

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- RDKit
- scikit-learn
- pandas, numpy, matplotlib
- tqdm, wandb (optional)

### Setup
```bash
# Clone the repository
git clone https://github.com/FafnirX26/QMugs_qGNN.git
cd QMugs_qGNN

# Install dependencies
pip install -r requirements.txt

# Create virtual environment (recommended)
python -m venv qgnn_env
source qgnn_env/bin/activate  # Linux/Mac
# or qgnn_env\\Scripts\\activate  # Windows
pip install -r requirements.txt
```

## 📁 Dataset Setup

### Option 1: Full QMugs Dataset (Advanced Users)
```bash
# Download QMugs dataset (12GB+ download)
python prepare_dataset.py --download

# Extract and organize (requires ~100GB disk space)
python prepare_dataset.py --extract
```

### Option 2: Quick Start with Smaller Datasets (Recommended)
```bash
# Create multiple dataset sizes for development
python create_small_dataset.py --strategy all

# This creates:
# - data/qmugs_small/tiny/     (1.7K molecules - 2min training)
# - data/qmugs_small/small/    (8K molecules - 30min training)  
# - data/qmugs_small/medium/   (40K molecules - 4hr training)
# - data/qmugs_small/stratified_small/ (8K diverse molecules)
```

## 🚀 Quick Start

### Ultra-Fast Training (Debugging)
```bash
# Train on tiny dataset - perfect for testing and debugging
python train_fast.py --size tiny --epochs 20
# ⏱️ Completes in 2-3 minutes
```

### Development Training
```bash
# Train on small dataset - ideal for development
python train_fast.py --size small --epochs 50
# ⏱️ Completes in 15-30 minutes
```

### Production Training
```bash
# Train on medium dataset - serious experiments
python train_fast.py --size medium --epochs 100
# ⏱️ Completes in 2-4 hours

# Train on full dataset - final models
python train_fast.py --size full --epochs 200
# ⏱️ Completes in days
```

### Custom Training Options
```bash
# Override batch size and epochs
python train_fast.py --size small --batch-size 32 --epochs 30

# Quiet mode for automated runs
python train_fast.py --size tiny --quiet

# Test existing model without training
python train_fast.py --size small --test-only
```

## ⚙️ Configuration

The project uses a flexible configuration system supporting multiple dataset sizes:

```python
# config_quick.py - Easy configuration switching
from config_quick import get_config

# Get configuration for different dataset sizes
config = get_config('tiny')    # Ultra-fast debugging
config = get_config('small')   # Development work
config = get_config('medium')  # Serious experiments  
config = get_config('full')    # Production training
```

### Key Configuration Options

| Parameter | Tiny | Small | Medium | Full |
|-----------|------|-------|--------|------|
| **Dataset Size** | 1.7K | 8K | 40K | 2M |
| **Hidden Dim** | 64 | 96 | 128 | 128 |
| **Layers** | 3 | 4 | 6 | 6 |
| **Attention Heads** | 4 | 6 | 8 | 8 |
| **Batch Size** | 32 | 48 | 64 | 64 |
| **Learning Rate** | 1e-3 | 5e-4 | 2e-4 | 1e-4 |
| **Training Time** | 2-3 min | 15-30 min | 2-4 hrs | Days |

## 🏗️ Model Architecture

```
Input: Molecular SDF → 3D Coordinates + Atomic Features
                                ↓
                    Node/Edge Embeddings (Linear)
                                ↓
                    3D Positional Encoding (Sinusoidal)
                                ↓
              Graph Transformer Layers (Multi-head Attention)
                                ↓
                      Self-Attention (Global Interactions)
                                ↓
                   Multi-Scale Global Pooling (Mean+Max+Sum)
                                ↓
                    MLP Readout → 7 Quantum Properties
```

### Architecture Components

1. **Input Processing**
   - Atomic features: element, degree, formal charge, hybridization
   - Bond features: bond type, aromaticity, ring membership
   - 3D coordinates: conformer geometry

2. **3D Positional Encoding**
   - Sinusoidal encoding of x, y, z coordinates
   - Captures spatial relationships in molecular geometry

3. **Graph Transformer Layers**
   - Multi-head attention over molecular graphs
   - Residual connections and layer normalization
   - Configurable depth (3-6 layers)

4. **Global Self-Attention**
   - Captures long-range molecular interactions
   - Complements local graph convolutions

5. **Multi-Scale Pooling**
   - Combines mean, max, and sum pooling
   - Creates rich graph-level representations

## 📊 Performance Monitoring

### TensorBoard
```bash
# View training progress
tensorboard --logdir logs/
```

### Weights & Biases (Optional)
```bash
# Automatic logging if wandb is configured
wandb login
python train_fast.py --size small
```

### Output Files
- `checkpoints/best_model.pt` - Best model checkpoint
- `logs/` - TensorBoard training logs
- `wandb/` - Weights & Biases experiment tracking

## 🔬 Results and Evaluation

The model outputs comprehensive metrics for each quantum property:

```python
# Example results for HOMO energy prediction
HOMO Energy:
  MAE: 0.143 eV
  R²:  0.891

LUMO Energy:  
  MAE: 0.167 eV
  R²:  0.856

HOMO-LUMO Gap:
  MAE: 0.224 eV
  R²:  0.798
```

## 📂 Project Structure

```
QMugs_qGNN/
├── 📊 data/
│   ├── qmugs/                     # Full dataset (2M molecules)
│   ├── qmugs_small/               # Development datasets
│   │   ├── tiny/                  # 1.7K molecules
│   │   ├── small/                 # 8K molecules
│   │   ├── medium/                # 40K molecules
│   │   └── stratified_small/      # 8K diverse molecules
│   └── extracted_structures/      # Raw SDF files
├── 🧠 Core Scripts
│   ├── train_fast.py              # Optimized training pipeline
│   ├── model.py                   # QuantumGNN architecture
│   ├── data_loader.py             # Data loading utilities
│   ├── config_quick.py            # Multi-size configurations
│   └── create_small_dataset.py    # Dataset sampling tools
├── 🔧 Utilities
│   ├── prepare_dataset.py         # Full dataset preparation
│   ├── train.py                   # Original training script
│   └── utils.py                   # Analysis utilities
├── 📈 Output
│   ├── checkpoints/               # Model checkpoints
│   ├── logs/                      # TensorBoard logs
│   └── wandb/                     # Weights & Biases logs
└── 📋 Documentation
    ├── README.md                  # This file
    ├── requirements.txt           # Dependencies
    └── CLAUDE.md                  # Development context
```

## 🎯 Use Cases

### 🔬 Research Applications
- **Drug Discovery**: Predict ADMET properties early in drug design
- **Materials Science**: Screen molecular materials for electronic properties
- **Chemical Space Exploration**: Navigate chemical space using quantum descriptors

### 🛠️ Development Workflows
- **Rapid Prototyping**: Use tiny dataset for quick algorithm testing
- **Hyperparameter Tuning**: Use small dataset for parameter optimization
- **Model Validation**: Use medium dataset for robust performance evaluation
- **Production Training**: Use full dataset for final model deployment

## 🚨 Troubleshooting

### Memory Issues
```bash
# Reduce batch size for GPU memory constraints
python train_fast.py --size small --batch-size 16

# Use CPU if GPU memory is insufficient
CUDA_VISIBLE_DEVICES="" python train_fast.py --size tiny
```

### Slow Training
```bash
# Start with smaller dataset
python train_fast.py --size tiny

# Reduce model complexity
# Edit config_quick.py to decrease hidden_dim or num_layers
```

### Data Loading Errors
```bash
# Verify dataset exists
ls data/qmugs_small/small/

# Create datasets if missing
python create_small_dataset.py --strategy small
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Install in development mode
pip install -e .

# Run tests (if available)
python -m pytest tests/

# Format code
black *.py
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{qmugs_qgnn,
  title={Quantum Graph Neural Network for QMugs Dataset},
  author={Ravindra Name},
  year={2024},
  url={https://github.com/yourusername/QMugs_qGNN}
}
```

Also cite the original QMugs dataset:
```bibtex
@article{qmugs2021,
  title={QMugs: Quantum Mechanical Properties of Drug-like Molecules},
  journal={Scientific Data},
  year={2021}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **QMugs Dataset** creators for providing high-quality quantum mechanical data
- **PyTorch Geometric** team for excellent graph neural network tools
- **RDKit** community for molecular informatics utilities
- **Open Source Community** for inspiration and best practices

---

**⭐ Star this repository if you find it useful!**

For questions, issues, or collaborations, please open an issue or contact the maintainers.
