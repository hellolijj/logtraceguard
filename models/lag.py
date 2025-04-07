import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class LogGNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.1):
        super(LogGNNLayer, self).__init__(aggr='mean')
        
        # Node feature transformation
        self.node_lin = nn.Linear(node_dim, hidden_dim)
        
        # Edge feature transformation
        self.edge_lin = nn.Linear(edge_dim, hidden_dim)
        
        # Message transformation
        self.msg_lin = nn.Linear(2 * hidden_dim, hidden_dim)
        
        # Update transformation
        self.update_lin = nn.Linear(2 * hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index, edge_attr):
        # Ensure correct dimensions
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        
        # Transform node features
        x = self.node_lin(x)
        
        # Transform edge features
        edge_attr = self.edge_lin(edge_attr)
        
        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Update node features
        out = self.update_lin(torch.cat([x, out], dim=-1))
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        return out
    
    def message(self, x_j, edge_attr):
        # Ensure correct dimensions
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        
        # Combine node and edge features
        msg = torch.cat([x_j, edge_attr], dim=-1)
        msg = self.msg_lin(msg)
        return msg

class LogAwareGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config['num_layers']
        self.node_dim = config['node_dim']
        self.edge_dim = config['edge_dim']
        self.dropout = config.get('dropout', 0.1)
        
        # Node and edge encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(config['input_dim'], self.node_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(config['edge_input_dim'], self.edge_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # GNN layers
        self.layers = nn.ModuleList([
            LogGNNLayer(self.node_dim, self.edge_dim, self.node_dim, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # Graph pooling
        self.pool = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
    def forward(self, data):
        # Encode node and edge features
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)
        
        # Apply GNN layers
        for layer in self.layers:
            x = layer(x, data.edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Compute edge representations
        edge_repr = edge_attr
        
        # Compute graph representation
        graph_repr = self.pool(x.mean(dim=0, keepdim=True))
        
        return x, edge_repr, graph_repr