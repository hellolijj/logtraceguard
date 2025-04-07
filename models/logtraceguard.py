import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lag import LogGNNLayer
from models.pu_classifier import PUClassifier

class LogTraceGuard(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers, dropout=0.1):
        super(LogTraceGuard, self).__init__()
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            LogGNNLayer(node_dim if i == 0 else hidden_dim,
                       edge_dim,
                       hidden_dim,
                       dropout)
            for i in range(num_layers)
        ])
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # PU classifier
        self.pu_classifier = PUClassifier(hidden_dim, dropout)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # GNN forward pass
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
        
        # Global mean pooling
        graph_repr = torch.mean(x, dim=0)
        
        # Projection
        z = self.projection(graph_repr)
        
        return z, graph_repr
    
    def compute_loss(self, view1, view2, labels):
        # Get representations
        z1, _ = self(view1)
        z2, _ = self(view2)
        
        # Normalize representations
        z1 = F.normalize(z1, dim=0)
        z2 = F.normalize(z2, dim=0)
        
        # Compute contrastive loss
        sim = torch.mm(z1, z2.t())
        labels = torch.arange(sim.size(0)).to(sim.device)
        loss = F.cross_entropy(sim, labels)
        
        # Add PU loss
        pu_loss = self.pu_classifier.loss(z1, labels)
        loss += pu_loss
        
        return loss
