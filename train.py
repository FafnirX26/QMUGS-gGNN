import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime

from config import Config
from model import QuantumGNN, QuantumGNNWithAttention
from data_loader import create_data_loaders

class QuantumGNNTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        
        # Create directories
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
        # Load data
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(config)
        
        # Initialize model
        self.model = self.create_model()
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
    def setup_logging(self):
        """Setup tensorboard and wandb logging"""
        timestamp = datetime.now().strftime("%Y%m %d_%H%M%S")
        
        # Tensorboard
        self.tb_writer = SummaryWriter(
            log_dir=os.path.join(self.config.log_dir, f"run_{timestamp}")
        )
        
        # Wandb (optional)
        try:
            wandb.init(
                project="quantum-gnn",
                config=vars(self.config),
                name=f"qgnn_{timestamp}"
            )
            self.use_wandb = True
        except:
            self.use_wandb = False
            print("Warning: wandb not available, using only tensorboard")
    
    def create_model(self):
        """Create model based on config"""
        # Get input dimensions from first batch
        sample_batch = next(iter(self.train_loader))
        input_dim = sample_batch.x.shape[1]
        edge_dim = sample_batch.edge_attr.shape[1] if sample_batch.edge_attr is not None else 4
        output_dim = len(self.config.target_properties)
        
        model = QuantumGNNWithAttention(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=output_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            edge_dim=edge_dim
        )
        
        return model
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Skip if no targets
            if batch.y is None:
                continue
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred = self.model(batch)
            
            # Reshape targets to match prediction shape
            batch_size = len(torch.unique(batch.batch))
            num_properties = len(self.config.target_properties)
            targets = batch.y.view(batch_size, num_properties)
            
            loss = self.criterion(pred, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, epoch: int):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch = batch.to(self.device)
                
                if batch.y is None:
                    continue
                
                pred = self.model(batch)
                
                # Reshape targets to match prediction shape
                batch_size = len(torch.unique(batch.batch))
                num_properties = len(self.config.target_properties)
                targets = batch.y.view(batch_size, num_properties)
                
                loss = self.criterion(pred, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        self.val_losses.append(avg_loss)
        
        # Calculate additional metrics
        if all_preds:
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            
            metrics = self.calculate_metrics(all_preds, all_targets)
            
            # Log metrics
            self.log_metrics(epoch, avg_loss, metrics, split='val')
        
        return avg_loss
    
    def calculate_metrics(self, preds, targets):
        """Calculate evaluation metrics"""
        metrics = {}
        
        for i, prop_name in enumerate(self.config.target_properties):
            pred_prop = preds[:, i]
            target_prop = targets[:, i]
            
            mae = mean_absolute_error(target_prop, pred_prop)
            r2 = r2_score(target_prop, pred_prop)
            
            metrics[f'{prop_name}_mae'] = mae
            metrics[f'{prop_name}_r2'] = r2
        
        # Overall metrics
        metrics['overall_mae'] = mean_absolute_error(targets, preds)
        metrics['overall_r2'] = r2_score(targets.flatten(), preds.flatten())
        
        return metrics
    
    def log_metrics(self, epoch, loss, metrics, split='train'):
        """Log metrics to tensorboard and wandb"""
        # Tensorboard
        self.tb_writer.add_scalar(f'{split}/loss', loss, epoch)
        
        for metric_name, value in metrics.items():
            self.tb_writer.add_scalar(f'{split}/{metric_name}', value, epoch)
        
        # Wandb
        if self.use_wandb:
            log_dict = {f'{split}_loss': loss}
            log_dict.update({f'{split}_{k}': v for k, v in metrics.items()})
            wandb.log(log_dict, step=epoch)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config)
        }
        
        # Save regular checkpoint
        if epoch % self.config.save_every == 0:
            checkpoint_path = os.path.join(
                self.config.checkpoint_dir,
                f'checkpoint_epoch_{epoch}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
    
    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate(epoch)
            
            # Scheduler step
            self.scheduler.step()
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"Best Val Loss: {self.best_val_loss:.6f} (Epoch {self.best_epoch+1})")
            print("-" * 50)
        
        print("Training completed!")
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Curves')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.yscale('log')
        plt.legend()
        plt.title('Training Curves (Log Scale)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, 'training_curves.png'))
        plt.show()
    
    def test(self):
        """Test the best model"""
        # Load best model
        best_model_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
        
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = batch.to(self.device)
                
                if batch.y is None:
                    continue
                
                pred = self.model(batch)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        if all_preds:
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            
            metrics = self.calculate_metrics(all_preds, all_targets)
            
            print("\nTest Results:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
            
            # Save results
            results = {
                'test_metrics': metrics,
                'predictions': all_preds.tolist(),
                'targets': all_targets.tolist()
            }
            
            with open(os.path.join(self.config.log_dir, 'test_results.json'), 'w') as f:
                json.dump(results, f, indent=2)

def main():
    config = Config()
    trainer = QuantumGNNTrainer(config)
    
    # Train model
    trainer.train()
    
    # Test model
    trainer.test()

if __name__ == "__main__":
    main()