import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

def pearson_loss(pred, true):
    """Pearson correlation loss"""
    vx = pred - torch.mean(pred)
    vy = true - torch.mean(true)
    return 1 - torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-8)

def calculate_metrics(true, pred):
    """Calculate various evaluation metrics"""
    true = true.cpu().numpy()
    pred = pred.cpu().numpy()
    
    metrics = {
        'r2': r2_score(true, pred),
        'pearson': pearsonr(true, pred)[0],
        'mae': mean_absolute_error(true, pred),
        'mse': mean_squared_error(true, pred),
        'rmse': np.sqrt(mean_squared_error(true, pred))
    }
    
    return metrics