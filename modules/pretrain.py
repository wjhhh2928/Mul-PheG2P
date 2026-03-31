import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_loader import SNPDataset
from utils.metrics import pearson_loss
from models import SNPEncoderCNN

import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ===================== Utility functions =====================

def _select_X_rows(X, rows):
    """Select X by row: Compatible pandas.DataFrame / np.ndarray"""
    if hasattr(X, "iloc"):  # pandas
        return X.iloc[rows, :]
    else:                   # numpy
        return X[rows]

def _select_y_rows_col(Y, rows, col_idx, col_name=None):
    """Select rows and columns for Y: prioritize using column names, otherwise use positions; compatible pandas / numpy"""
    if hasattr(Y, "iloc"):  # pandas
        if col_name is not None and hasattr(Y, "columns") and (col_name in list(Y.columns)):
            y = Y.loc[rows, col_name]
        else:
            y = Y.iloc[rows, col_idx]
    else:                   # numpy
        y = Y[rows, col_idx]
    
    return np.asarray(y, dtype=np.float32).reshape(-1)

def _safe_filename(name: str) -> str:
    return str(name).replace(" ", "_").replace("/", "_")

# =================================================================================

def train(snp, phe, source_traits, train_idx, val_idx, args):
    
   
    lr             = args['lr']
    batch_size     = args['batch_size']
    pretrain_epoch = args['pretrain_epoch']
    
    input_len      = args['input_len']
    conv_channels  = args['conv_channels']
    kernel_size    = args['kernel_size']
    stride         = args['stride']
    hidden_dim     = args['hidden_dim']
    
    out_dir        = args.get('output', './outputs/pretrain')
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[Pretrain] Using strictly provided train_idx ({len(train_idx)}) and val_idx ({len(val_idx)})")
    
   
    os.makedirs(out_dir, exist_ok=True)
    
  
    for trait_idx, trait_name in enumerate(source_traits):
        print(f"\n---> Pretraining for trait: {trait_name}")

        model = SNPEncoderCNN(input_len, conv_channels, kernel_size, stride, hidden_dim).to(device)
        head  = nn.Linear(hidden_dim, 1).to(device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr)
        mse = nn.MSELoss()

       
        X_train = _select_X_rows(snp, train_idx)
        if hasattr(X_train, "to_numpy"): X_train = X_train.to_numpy(copy=True)
        X_train = np.asarray(X_train, dtype=np.float32, order="C")
        
       
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

        y_train = _select_y_rows_col(phe, train_idx, trait_idx, col_name=trait_name)
        if np.isnan(y_train).any() or np.isinf(y_train).any():
            med = float(np.nanmedian(y_train))
            y_train = np.where(np.isfinite(y_train), y_train, med).astype(np.float32)

        train_set = SNPDataset(X_train, y_train)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        
       
        X_val = _select_X_rows(snp, val_idx)
        if hasattr(X_val, "to_numpy"): X_val = X_val.to_numpy(copy=True)
        X_val = np.asarray(X_val, dtype=np.float32, order="C")
        
        if np.isnan(X_val).any() or np.isinf(X_val).any():
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        y_val = _select_y_rows_col(phe, val_idx, trait_idx, col_name=trait_name)
        if np.isnan(y_val).any() or np.isinf(y_val).any():
            med_val = float(np.nanmedian(y_val))
            y_val = np.where(np.isfinite(y_val), y_val, med_val).astype(np.float32)

        val_set = SNPDataset(X_val, y_val)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

       
        best_val_loss = float('inf')
        best_model_state = None
        best_head_state = None

        for epoch in range(pretrain_epoch):
            model.train()
            head.train()
            train_loss_accum = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device, non_blocking=False).contiguous()
                batch_y = batch_y.to(device, non_blocking=False).contiguous()

                feat = model(batch_x)
                pred = head(feat).squeeze(1)
                loss = mse(pred, batch_y) + 0.2 * pearson_loss(pred, batch_y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                train_loss_accum += loss.item()

            avg_train_loss = train_loss_accum / max(1, len(train_loader))

            model.eval()
            head.eval()
            val_loss_accum = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device, non_blocking=False).contiguous()
                    batch_y = batch_y.to(device, non_blocking=False).contiguous()
                    
                    feat = model(batch_x)
                    pred = head(feat).squeeze(1)
                    v_loss = mse(pred, batch_y) + 0.2 * pearson_loss(pred, batch_y)
                    val_loss_accum += v_loss.item()
                    
            avg_val_loss = val_loss_accum / max(1, len(val_loader))

            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_head_state = {k: v.cpu() for k, v in head.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"  [Epoch {epoch+1:03d}/{pretrain_epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        
        save_name = _safe_filename(trait_name)
        save_path = os.path.join(out_dir, f'pretrain_{save_name}.pth')
        
        torch.save(
            {
                'model_state_dict': best_model_state,
                'head_state_dict' : best_head_state,
                'config': {
                    'input_len': input_len,
                    'conv_channels': conv_channels,
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'hidden_dim': hidden_dim,
                },
                'train_idx': np.asarray(train_idx, dtype=np.int64),
                'val_idx': np.asarray(val_idx, dtype=np.int64) 
            },
            save_path
        )
        print(f"  --> Saved best model to: {save_path}")
