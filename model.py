import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, TransformerConv, 
    global_mean_pool, global_max_pool, global_add_pool,
    BatchNorm, LayerNorm
)
from torch_geometric.data import Batch
import math

class QuantumGNN(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 edge_dim: int = 4):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Use TransformerConv for better performance on molecular data
            conv = TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                edge_dim=hidden_dim,
                beta=True
            )
            self.conv_layers.append(conv)
            self.norm_layers.append(LayerNorm(hidden_dim))
        
        # 3D position encoding
        self.pos_encoder = PositionalEncoder3D(hidden_dim)
        
        # Global pooling and readout
        self.pool_layers = nn.ModuleList([
            global_mean_pool,
            global_max_pool,
            global_add_pool
        ])
        
        # Output layers
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, data):
        x, edge_index, edge_attr, pos, batch = (
            data.x, data.edge_index, data.edge_attr, data.pos, data.batch
        )
        
        # Initial embeddings
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # Add 3D positional encoding
        pos_enc = self.pos_encoder(pos)
        x = x + pos_enc
        
        # Graph convolution layers with residual connections
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            x_residual = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if i > 0:
                x = x + x_residual
        
        # Global pooling
        pooled_features = []
        for pool_fn in self.pool_layers:
            pooled_features.append(pool_fn(x, batch))
        
        graph_repr = torch.cat(pooled_features, dim=1)
        
        # Final prediction
        out = self.readout(graph_repr)
        
        return out

class PositionalEncoder3D(nn.Module):
    def __init__(self, hidden_dim: int, max_freq: float = 10.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_freq = max_freq
        
        # Create frequency bands
        freq_bands = torch.linspace(0, max_freq, hidden_dim // 6)
        self.register_buffer('freq_bands', freq_bands)
        
        self.linear = nn.Linear(hidden_dim // 6 * 6, hidden_dim)
    
    def forward(self, pos):
        # pos: [N, 3] - 3D coordinates
        batch_size, _ = pos.shape
        
        # Apply sinusoidal encoding to each dimension
        pos_enc = []
        for i in range(3):  # x, y, z coordinates
            coord = pos[:, i:i+1]  # [N, 1]
            
            # Apply frequency encoding
            angles = coord * self.freq_bands.unsqueeze(0)  # [N, num_freqs]
            sin_enc = torch.sin(angles)
            cos_enc = torch.cos(angles)
            
            pos_enc.extend([sin_enc, cos_enc])
        
        pos_enc = torch.cat(pos_enc, dim=1)  # [N, hidden_dim//6 * 6]
        pos_enc = self.linear(pos_enc)
        
        return pos_enc

class QuantumGNNWithAttention(QuantumGNN):
    """Enhanced version with self-attention for better long-range interactions"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Self-attention layers
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
        
        self.attention_norm = LayerNorm(self.hidden_dim)
    
    def forward(self, data):
        x, edge_index, edge_attr, pos, batch = (
            data.x, data.edge_index, data.edge_attr, data.pos, data.batch
        )
        
        # Initial embeddings
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # Add 3D positional encoding
        pos_enc = self.pos_encoder(pos)
        x = x + pos_enc
        
        # Graph convolution layers
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            x_residual = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if i > 0:
                x = x + x_residual
        
        # Self-attention for global interactions
        # Group by batch for attention
        x_att_list = []
        for batch_idx in torch.unique(batch):
            mask = batch == batch_idx
            x_batch = x[mask].unsqueeze(0)  # [1, N, hidden_dim]
            
            x_att, _ = self.self_attention(x_batch, x_batch, x_batch)
            x_att_list.append(x_att.squeeze(0))
        
        x_att = torch.cat(x_att_list, dim=0)
        x = self.attention_norm(x + x_att)
        
        # Global pooling
        pooled_features = []
        for pool_fn in self.pool_layers:
            pooled_features.append(pool_fn(x, batch))
        
        graph_repr = torch.cat(pooled_features, dim=1)
        
        # Final prediction
        out = self.readout(graph_repr)
        
        return out