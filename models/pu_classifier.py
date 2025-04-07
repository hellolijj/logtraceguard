import torch
import torch.nn as nn
import torch.nn.functional as F

class PUClassifier(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super(PUClassifier, self).__init__()
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Prior probability of positive class
        self.prior = 0.5  # Can be adjusted based on domain knowledge
        
    def forward(self, x):
        # Ensure correct dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        return self.classifier(x)
    
    def loss(self, x, labels):
        # Ensure correct dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Get predictions
        pred = self.classifier(x)
        
        # Compute risks
        pos_risk = torch.mean(F.binary_cross_entropy(pred[labels == 1], torch.ones_like(pred[labels == 1])))
        unlabeled_risk = torch.mean(F.binary_cross_entropy(pred[labels == 0], torch.zeros_like(pred[labels == 0])))
        
        # Compute PU loss
        loss = pos_risk + (unlabeled_risk - self.prior * pos_risk) / (1 - self.prior)
        
        return loss
