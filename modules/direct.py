import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
from utils.data_loader import SNPDataset
from models import SNPEncoderCNN


def train(snp, phe, target_trait, train_idx, val_idx, test_idx, args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    Y = phe[target_trait].values
    Y = np.nan_to_num(Y, nan=np.nanmean(Y))
    scaler_y = MinMaxScaler()
    Y_scaled = scaler_y.fit_transform(Y.reshape(-1, 1)).flatten()
    
    
    dataset = SNPDataset(snp, Y_scaled)
    
    
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)
    
    
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False)
    
    
    model = SNPEncoderCNN(
        input_len=args['input_len'],
        conv_channels=args['conv_channels'],
        kernel_size=args['kernel_size'],
        stride=args['stride'],
        hidden_dim=args['hidden_dim']
    ).to(device)
    
    head = nn.Sequential(
        nn.Linear(args['hidden_dim'], 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()),
        lr=args['lr']
    )
    criterion = nn.MSELoss()
    
    best_score = -1
    best_model_state = None
    best_head_state = None
    
    print(f"[Direct Baseline] Training on {len(train_idx)} samples, validating on {len(val_idx)} samples...")

    
    epochs = args.get('direct_epoch', args.get('direct_epochs', 300)) 
    
    for epoch in range(epochs):
        model.train()
        head.train()
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            features = model(batch_x)
            pred = head(features).squeeze()
            
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        head.eval()
        val_preds, val_true = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                features = model(batch_x)
                pred = head(features).squeeze()
                
                val_preds.extend(pred.cpu().numpy())
                val_true.extend(batch_y.numpy())
            
            corr = pearsonr(val_preds, val_true)[0]
            
           
            if corr > best_score:
                best_score = corr
                
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_head_state = {k: v.cpu() for k, v in head.state_dict().items()}
    
    print(f"[Direct Baseline] Best Val Pearson: {best_score:.4f}")
    
    
    model.load_state_dict(best_model_state)
    head.load_state_dict(best_head_state)
    model.eval()
    head.eval()
    test_preds_s, test_true_s = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            features = model(batch_x)
            pred = head(features).squeeze()
            
            test_preds_s.extend(pred.cpu().numpy())
            test_true_s.extend(batch_y.numpy())
            
    
    test_preds = scaler_y.inverse_transform(np.array(test_preds_s).reshape(-1, 1)).flatten()
    test_true = scaler_y.inverse_transform(np.array(test_true_s).reshape(-1, 1)).flatten()
    
    
    test_pearson = pearsonr(test_preds, test_true)[0]
    test_r2 = r2_score(test_true, test_preds)
    test_mae = mean_absolute_error(test_true, test_preds)
    test_rmse = np.sqrt(mean_squared_error(test_true, test_preds))
    
    print(f"[Direct Baseline] Test Metrics - Pearson: {test_pearson:.4f}, R2: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    
    os.makedirs(args['output'], exist_ok=True)
    torch.save({
        'model_state_dict': best_model_state,
        'head_state_dict': best_head_state
    }, os.path.join(args['output'], f'direct_{target_trait.replace(" ", "_")}.pth'))
    
    
    res_df = pd.DataFrame([{
        'Task': target_trait,
        'Model': 'Direct',
        'Pearson': test_pearson,
        'R2': test_r2,
        'MAE': test_mae,
        'RMSE': test_rmse
    }])
    res_df.to_csv(os.path.join(args['output'], 'direct_res.csv'), index=False)
