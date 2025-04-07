import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_sim(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return torch.matmul(a, b.T)

class ContrastiveLossModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temperature = config['temperature']
        self.node_weight = config.get('node_weight', 1.0)
        self.edge_weight = config.get('edge_weight', 1.0)
        self.graph_weight = config.get('graph_weight', 1.0)
        self.pu_weight = config.get('pu_weight', 1.0)
        
    def node_loss(self, z1, z2):
        # z1, z2: [N, D] node embeddings
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        sim = torch.matmul(z1, z2.T) / self.temperature
        positives = torch.diag(sim)
        loss = -torch.log(positives / (sim.sum(dim=1) + 1e-8))
        return loss.mean()
        
    def edge_loss(self, h1, h2):
        # h1, h2: [E, D] edge embeddings
        h1 = F.normalize(h1, dim=-1)
        h2 = F.normalize(h2, dim=-1)
        
        sim = torch.matmul(h1, h2.T) / self.temperature
        positives = torch.diag(sim)
        loss = -torch.log(positives / (sim.sum(dim=1) + 1e-8))
        return loss.mean()
        
    def graph_loss(self, g1, g2):
        # g1, g2: [B, D] graph embeddings
        g1 = F.normalize(g1, dim=-1)
        g2 = F.normalize(g2, dim=-1)
        
        sim = torch.matmul(g1, g2.T) / self.temperature
        positives = torch.diag(sim)
        loss = -torch.log(positives / (sim.sum(dim=1) + 1e-8))
        return loss.mean()

class GraphContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # z1, z2: [B, D] global graph representations
        sim = cosine_sim(z1, z2) / self.temperature
        positives = torch.diag(sim)
        loss = -torch.log(positives / (sim.sum(dim=1) + 1e-8))
        return loss.mean()

class NodeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # z1, z2: [N, D] node embeddings (same graph in two views)
        sim = cosine_sim(z1, z2) / self.temperature
        positives = torch.diag(sim)
        loss = -torch.log(positives / (sim.sum(dim=1) + 1e-8))
        return loss.mean()

class EdgeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, h1, h2):
        # h1, h2: [E, D] edge embeddings in two views
        sim = cosine_sim(h1, h2) / self.temperature
        positives = torch.diag(sim)
        loss = -torch.log(positives / (sim.sum(dim=1) + 1e-8))
        return loss.mean()