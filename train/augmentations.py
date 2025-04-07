import torch
import random

def generate_augmented_views(data, cfg):
    """Generate two augmented views of the input graph"""
    # First view: node feature masking
    view1 = mask_node_features(data, cfg['augmentation']['feature_drop_prob'])
    
    # Second view: edge dropping
    view2 = drop_edges(data, cfg['augmentation']['edge_drop_prob'])
    
    return view1, view2

def mask_node_features(data, drop_prob):
    """Randomly mask node features with zeros"""
    if drop_prob == 0:
        return data
    
    # Create mask
    mask = torch.rand(data.x.size(0), 1) > drop_prob
    mask = mask.to(data.x.device)
    
    # Apply mask
    x_masked = data.x * mask
    
    # Create new data object
    view = data.clone()
    view.x = x_masked
    
    return view

def drop_edges(data, drop_prob):
    """Randomly drop edges from the graph"""
    if drop_prob == 0 or data.edge_index.size(1) == 0:
        return data
    
    # Create mask
    mask = torch.rand(data.edge_index.size(1)) > drop_prob
    mask = mask.to(data.edge_index.device)
    
    # Apply mask
    edge_index = data.edge_index[:, mask]
    edge_attr = data.edge_attr[mask]
    
    # Create new data object
    view = data.clone()
    view.edge_index = edge_index
    view.edge_attr = edge_attr
    
    return view 