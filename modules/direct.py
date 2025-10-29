import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import pandas as pd
from utils.data_loader import SNPDataset
from models import SNPEncoderCNN

def train(snp, phe, target_trait, args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
   
    Y = phe[target_trait].values
    Y = np.nan_to_num(Y, nan=np.nanmean(Y))
    Y = MinMaxScaler().fit_transform(Y.reshape(-1, 1)).flatten()
    
   
    dataset = SNPDataset(snp, Y)
    
   
    kfold = KFold(n_splits=args.get('cv_folds', 5), 
                 shuffle=True, 
                 random_state=args['seed'])
    
  
    results = []
    best_model = None
    best_score = -1
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n=== Fold {fold+1} ===")
        
       
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_set, 
                                batch_size=args['batch_size'], 
                                shuffle=True)
        val_loader = DataLoader(val_set, 
                               batch_size=args['batch_size'])
        
        
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
        
       
        for epoch in range(args['direct_epoch']):
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
                print(f"Epoch {epoch+1}: Pearson = {corr:.4f}")
                
                if corr > best_score:
                    best_score = corr
                    best_model = {
                        'model': model.state_dict(),
                        'head': head.state_dict()
                    }
        
        
        results.append({
            'fold': fold+1,
            'pearson': corr,
            'epoch': epoch+1
        })
    
    
    os.makedirs(args['output'], exist_ok=True)
    torch.save(best_model, 
              os.path.join(args['output'], f'direct_{target_trait}.pth'))
    
   
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(args['output'], 'direct_res.csv'), index=False)
    
    print(f"\nBest Pearson Correlation: {best_score:.4f}")