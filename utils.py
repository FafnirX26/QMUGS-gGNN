import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
from typing import List, Dict, Tuple
import os

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_predictions(predictions: np.ndarray, 
                       targets: np.ndarray, 
                       property_names: List[str],
                       save_dir: str = None):
    """Analyze and visualize model predictions"""
    
    fig, axes = plt.subplots(2, len(property_names), figsize=(4*len(property_names), 8))
    if len(property_names) == 1:
        axes = axes.reshape(-1, 1)
    
    results = {}
    
    for i, prop_name in enumerate(property_names):
        pred_prop = predictions[:, i]
        target_prop = targets[:, i]
        
        # Calculate metrics
        mae = mean_absolute_error(target_prop, pred_prop)
        r2 = r2_score(target_prop, pred_prop)
        results[prop_name] = {'mae': mae, 'r2': r2}
        
        # Scatter plot
        axes[0, i].scatter(target_prop, pred_prop, alpha=0.6, s=10)
        axes[0, i].plot([target_prop.min(), target_prop.max()], 
                       [target_prop.min(), target_prop.max()], 'r--', lw=2)
        axes[0, i].set_xlabel(f'True {prop_name}')
        axes[0, i].set_ylabel(f'Predicted {prop_name}')
        axes[0, i].set_title(f'{prop_name}\nMAE: {mae:.4f}, R²: {r2:.4f}')
        
        # Residuals plot
        residuals = pred_prop - target_prop
        axes[1, i].scatter(target_prop, residuals, alpha=0.6, s=10)
        axes[1, i].axhline(y=0, color='r', linestyle='--')
        axes[1, i].set_xlabel(f'True {prop_name}')
        axes[1, i].set_ylabel('Residuals')
        axes[1, i].set_title(f'Residuals - {prop_name}')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return results

def plot_training_history(train_losses: List[float], 
                         val_losses: List[float],
                         save_dir: str = None):
    """Plot training history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Linear scale
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(True)
    
    # Log scale
    ax2.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax2.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_yscale('log')
    ax2.set_title('Training History (Log Scale)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def create_summary_report(metrics: Dict[str, Dict[str, float]], 
                         model_info: Dict,
                         save_path: str = None):
    """Create a summary report of model performance"""
    
    report = []
    report.append("=" * 60)
    report.append("QUANTUM GNN MODEL PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Model information
    report.append("MODEL INFORMATION:")
    report.append("-" * 30)
    for key, value in model_info.items():
        report.append(f"{key}: {value}")
    report.append("")
    
    # Performance metrics
    report.append("PERFORMANCE METRICS:")
    report.append("-" * 30)
    
    property_names = []
    mae_values = []
    r2_values = []
    
    for prop_name, prop_metrics in metrics.items():
        if prop_name.startswith('overall'):
            continue
        property_names.append(prop_name.replace('_mae', '').replace('_r2', ''))
        
    # Remove duplicates and sort
    property_names = sorted(list(set(property_names)))
    
    for prop_name in property_names:
        mae_key = f"{prop_name}_mae"
        r2_key = f"{prop_name}_r2"
        
        if mae_key in metrics and r2_key in metrics:
            mae = metrics[mae_key]
            r2 = metrics[r2_key]
            report.append(f"{prop_name}:")
            report.append(f"  MAE: {mae:.6f}")
            report.append(f"  R²:  {r2:.6f}")
            report.append("")
    
    # Overall metrics
    if 'overall_mae' in metrics:
        report.append("OVERALL PERFORMANCE:")
        report.append("-" * 30)
        report.append(f"Overall MAE: {metrics['overall_mae']:.6f}")
        report.append(f"Overall R²:  {metrics['overall_r2']:.6f}")
    
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    print(report_text)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text

def save_model_predictions(model, dataloader, device, save_path: str):
    """Save model predictions for analysis"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_mol_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            if batch.y is None:
                continue
            
            predictions = model(batch)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
            
            # Extract molecule IDs if available
            if hasattr(batch, 'mol_id'):
                all_mol_ids.extend(batch.mol_id)
    
    if all_predictions:
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Create DataFrame
        data = {
            'mol_id': all_mol_ids if all_mol_ids else range(len(predictions))
        }
        
        # Add predictions and targets for each property
        property_names = ['homo_energy', 'lumo_energy', 'gap_energy', 'dipole_moment', 
                         'polarizability', 'electronic_energy', 'zero_point_energy']
        
        for i, prop_name in enumerate(property_names[:predictions.shape[1]]):
            data[f'{prop_name}_pred'] = predictions[:, i]
            data[f'{prop_name}_true'] = targets[:, i]
            data[f'{prop_name}_error'] = predictions[:, i] - targets[:, i]
        
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        
        return df
    
    return None

def check_data_distribution(dataloader, property_names: List[str]):
    """Analyze the distribution of target properties"""
    all_targets = []
    
    for batch in dataloader:
        if batch.y is not None:
            all_targets.append(batch.y.cpu().numpy())
    
    if not all_targets:
        print("No target data found!")
        return
    
    targets = np.concatenate(all_targets, axis=0)
    
    fig, axes = plt.subplots(2, len(property_names), figsize=(4*len(property_names), 8))
    if len(property_names) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, prop_name in enumerate(property_names):
        if i >= targets.shape[1]:
            break
            
        prop_values = targets[:, i]
        
        # Histogram
        axes[0, i].hist(prop_values, bins=50, alpha=0.7)
        axes[0, i].set_title(f'{prop_name} Distribution')
        axes[0, i].set_xlabel(prop_name)
        axes[0, i].set_ylabel('Frequency')
        
        # Box plot
        axes[1, i].boxplot(prop_values)
        axes[1, i].set_title(f'{prop_name} Box Plot')
        axes[1, i].set_ylabel(prop_name)
        
        # Print statistics
        print(f"\n{prop_name} Statistics:")
        print(f"  Mean: {np.mean(prop_values):.6f}")
        print(f"  Std:  {np.std(prop_values):.6f}")
        print(f"  Min:  {np.min(prop_values):.6f}")
        print(f"  Max:  {np.max(prop_values):.6f}")
    
    plt.tight_layout()
    plt.show()