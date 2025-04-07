import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import random
import yaml
from models.logtraceguard import LogTraceGuard
from utils.evaluation import evaluate_model
from train.augmentations import generate_augmented_views
import torch_geometric.data as data

def load_mock_dataset(cfg):
    """Generate a mock dataset for testing"""
    num_nodes = cfg['data']['num_nodes']
    num_edges = cfg['data']['num_edges']
    feature_dim = cfg['model']['node_dim']
    
    # Generate random node features
    x = torch.randn(num_nodes, feature_dim)
    
    # Generate random edges (avoid self-loops)
    edge_index = []
    for _ in range(num_edges):
        src = random.randint(0, num_nodes - 1)
        dst = random.randint(0, num_nodes - 1)
        while src == dst:  # Avoid self-loops
            dst = random.randint(0, num_nodes - 1)
        edge_index.append([src, dst])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Generate edge features
    edge_attr = torch.randn(num_edges, cfg['model']['edge_dim'])
    
    # Generate random labels
    y = torch.randint(0, 2, (num_nodes,))
    
    # Create a single graph
    graph_data = data.Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )
    
    # Split into train/val/test
    train_size = int(cfg['data']['train_ratio'] * num_nodes)
    val_size = int(cfg['data']['val_ratio'] * num_nodes)
    
    indices = list(range(num_nodes))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = graph_data.subgraph(torch.tensor(train_indices))
    val_data = graph_data.subgraph(torch.tensor(val_indices))
    test_data = graph_data.subgraph(torch.tensor(test_indices))
    
    return train_data, val_data, test_data

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_data, val_data, test_data = load_mock_dataset(cfg)
    
    # Create data loaders
    train_loader = DataLoader([train_data], batch_size=1, shuffle=True)
    val_loader = DataLoader([val_data], batch_size=1, shuffle=False)
    test_loader = DataLoader([test_data], batch_size=1, shuffle=False)
    
    # Initialize model
    model = LogTraceGuard(
        node_dim=cfg['model']['node_dim'],
        edge_dim=cfg['model']['edge_dim'],
        hidden_dim=cfg['model']['hidden_dim'],
        num_layers=cfg['model']['num_layers'],
        dropout=cfg['model']['dropout']
    ).to(device)
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=cfg['training']['lr'])
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=cfg['training']['patience'],
        min_delta=cfg['training']['min_delta']
    )
    
    # Training loop
    for epoch in range(cfg['training']['num_epochs']):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            batch = batch.to(device)
            
            # Generate augmented views
            view1, view2 = generate_augmented_views(batch, cfg)
            view1 = view1.to(device)
            view2 = view2.to(device)
            
            # Forward pass
            loss = model.compute_loss(view1, view2, batch.y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        if (epoch + 1) % cfg['training']['eval_interval'] == 0:
            val_metrics = evaluate_model(model, val_loader, device, cfg)
            early_stopping(val_metrics['Loss'])
            
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
    
    # Final evaluation
    test_metrics = evaluate_model(model, test_loader, device, cfg, final=True)
    return test_metrics

def run_experiment(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    metrics = train(cfg)
    return metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()
    
    run_experiment(args.config)
