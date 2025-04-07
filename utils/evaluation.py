# utils/evaluation.py
# Evaluation metrics for binary anomaly detection

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from train.augmentations import generate_augmented_views

def evaluate_model(model, dataloader, device, cfg, final=False):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # Generate augmented views for evaluation
            view1, view2 = generate_augmented_views(batch, cfg)
            view1 = view1.to(device)
            view2 = view2.to(device)
            
            # Forward pass
            loss = model.compute_loss(view1, view2, batch.y)
            total_loss += loss.item()
            
            # Get predictions
            _, graph_repr = model(batch)
            pred = model.pu_classifier(graph_repr)
            
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_bin = (y_pred > 0.5).astype(int)
    
    # Compute metrics
    metrics = {
        "Loss": total_loss / len(dataloader),
        "Precision": precision_score(y_true, y_pred_bin, zero_division=0),
        "Recall": recall_score(y_true, y_pred_bin, zero_division=0),
        "F1": f1_score(y_true, y_pred_bin, zero_division=0),
        "AUC": roc_auc_score(y_true, y_pred)
    }
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_bin)
    metrics["Confusion Matrix"] = cm
    
    # Print results
    if final:
        print("\nFinal Evaluation Metrics:")
    else:
        print("\nEvaluation Metrics:")
        
    for metric_name, value in metrics.items():
        if metric_name == "Confusion Matrix":
            print(f"\n{metric_name}:")
            print(value)
        else:
            print(f"{metric_name}: {value:.4f}")
        
    return metrics
