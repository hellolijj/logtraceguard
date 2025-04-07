# models/encoder.py (patch with ablation control)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

class LogGNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, use_gat=True):
        super().__init__()
        self.edge_encoder = nn.Linear(edge_dim, node_dim)
        if use_gat:
            self.conv = GATConv(node_dim, node_dim)
        else:
            self.conv = GCNConv(node_dim, node_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_feat = self.edge_encoder(edge_attr)
        # In full use, edge_feat could influence x update via attention weights
        return self.conv(x, edge_index)


class LogGNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_lag = getattr(cfg.model, 'use_lag', True)
        self.use_gat = getattr(cfg.model, 'use_gat', True)

        self.node_encoder = nn.Linear(cfg.input_dim, cfg.node_dim)
        self.edge_encoder = nn.Linear(cfg.edge_input_dim, cfg.edge_dim)

        self.layers = nn.ModuleList([
            LogGNNLayer(cfg.node_dim, cfg.edge_dim, use_gat=self.use_gat) for _ in range(cfg.gnn_layers)
        ])

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr if self.use_lag else torch.zeros_like(edge_attr))

        return x