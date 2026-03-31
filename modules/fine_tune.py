import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_loader import SNPDataset
from utils.metrics import pearson_loss
from models import SNPEncoderCNN

# ===================== Utility functions =====================
def _select_X_rows(X, rows):
    if hasattr(X, "iloc"):   # pandas
        return X.iloc[rows, :]
    return X[rows]           # numpy

def _select_y_rows_col(Y, rows, col_idx=None, col_name=None):
    if hasattr(Y, "iloc"):   # pandas
        if (col_name is not None) and hasattr(Y, "columns") and (col_name in list(Y.columns)):
            y = Y.loc[rows, col_name]
        else:
            if col_idx is None:
                raise ValueError("Please provide col_idx.")
            y = Y.iloc[rows, int(col_idx)]
    else:                    # numpy
        if col_idx is None:
            raise ValueError("Please provide col_idx.")
        y = Y[rows, int(col_idx)]
    return np.asarray(y, dtype=np.float32).reshape(-1)

def _safe_filename(name: str) -> str:
    return str(name).replace(" ", "_").replace("/", "_")
# =============================================================


def train(snp, phe, target_trait, source_traits, train_idx, val_idx, test_idx, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    lr            = args['lr']
    batch_size    = args['batch_size']
    ft_epoch      = args.get('ft_epoch', args.get('finetune_epochs', 300))
    input_len     = args['input_len']
    conv_channels = args['conv_channels']
    kernel_size   = args['kernel_size']
    stride        = args['stride']
    hidden_dim    = args['hidden_dim']
    out_dir       = args.get('output', './outputs/pretrain')

    print(f"\n[Fine-Tune] Using strictly provided train_idx ({len(train_idx)}) and val_idx ({len(val_idx)})")

    trait_models = []
    for trait_name in source_traits:
        ckpt_path = os.path.join(out_dir, f'pretrain_{_safe_filename(trait_name)}.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[Error] Pre-trained weights not found: {ckpt_path}. Did pretrain.py run successfully?")
        
       
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model = SNPEncoderCNN(input_len, conv_channels, kernel_size, stride, hidden_dim).to(device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
        
        for p in model.parameters():
            p.requires_grad = False
        model.eval()  
        trait_models.append(model)

    print(f"[Fine-Tune] Successfully loaded {len(trait_models)} frozen pre-trained encoders.")

    
    head = nn.Linear(hidden_dim, 1).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    mse = nn.MSELoss()

   
    target_col_idx  = target_trait if isinstance(target_trait, (int, np.integer)) else None
    target_col_name = target_trait if isinstance(target_trait, str) else None

   
    X_train = _select_X_rows(snp, train_idx)
    if hasattr(X_train, "to_numpy"): X_train = X_train.to_numpy(copy=True)
    X_train = np.asarray(X_train, dtype=np.float32, order="C")
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    y_train = _select_y_rows_col(phe, train_idx, col_idx=target_col_idx, col_name=target_col_name)
    if np.isnan(y_train).any() or np.isinf(y_train).any():
        med = float(np.nanmedian(y_train))
        y_train = np.where(np.isfinite(y_train), y_train, med).astype(np.float32)

    
    X_val = _select_X_rows(snp, val_idx)
    if hasattr(X_val, "to_numpy"): X_val = X_val.to_numpy(copy=True)
    X_val = np.asarray(X_val, dtype=np.float32, order="C")
    if np.isnan(X_val).any() or np.isinf(X_val).any():
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    y_val = _select_y_rows_col(phe, val_idx, col_idx=target_col_idx, col_name=target_col_name)
    if np.isnan(y_val).any() or np.isinf(y_val).any():
        med_val = float(np.nanmedian(y_val))
        y_val = np.where(np.isfinite(y_val), y_val, med_val).astype(np.float32)

   
    train_set = SNPDataset(X_train, y_train)
    val_set   = SNPDataset(X_val,   y_val)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

   
    best_val_loss = float('inf') 
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'fine_tuned_{_safe_filename(str(target_trait))}.pth')

    for epoch in range(ft_epoch):
        head.train()
        train_loss_accum = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=False).contiguous(), batch_y.to(device, non_blocking=False).contiguous()

            with torch.no_grad():
                
                feats = [m(batch_x) for m in trait_models]     # list of [B, hidden_dim]
                
                feat  = torch.mean(torch.stack(feats, dim=0), dim=0)  # [B, hidden_dim]

           
            pred = head(feat).squeeze(1)  # [B]
            loss = mse(pred, batch_y) + 0.2 * pearson_loss(pred, batch_y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / max(1, len(train_loader))

        
        head.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device, non_blocking=False).contiguous(), batch_y.to(device, non_blocking=False).contiguous()
                
                feats = [m(batch_x) for m in trait_models]
                feat  = torch.mean(torch.stack(feats, dim=0), dim=0)
                pred  = head(feat).squeeze(1)
                
                v_loss = mse(pred, batch_y) + 0.2 * pearson_loss(pred, batch_y)
                val_loss_accum += v_loss.item()
                
        avg_val_loss = val_loss_accum / max(1, len(val_loader))

        if (epoch + 1) % 10 == 0:
            print(f"  [Epoch {epoch+1:03d}/{ft_epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

       
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    
                    'model_state_dicts': [m.state_dict() for m in trait_models],
                    
                    'head_state_dict': {k: v.cpu() for k, v in head.state_dict().items()},
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
    
    print(f"  --> Saved best fine-tuned model to: {save_path}")

   
    print("\n  [Fine-Tune] Evaluating on independent Test Set...")
    from scipy.stats import pearsonr
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import pandas as pd

   
    best_ckpt = torch.load(save_path, map_location=device)
    head.load_state_dict(best_ckpt['head_state_dict'])
    head.eval()

   
    X_test = _select_X_rows(snp, test_idx)
    if hasattr(X_test, "to_numpy"): X_test = X_test.to_numpy(copy=True)
    X_test = np.asarray(X_test, dtype=np.float32, order="C")
    if np.isnan(X_test).any() or np.isinf(X_test).any():
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    y_test = _select_y_rows_col(phe, test_idx, col_idx=target_col_idx, col_name=target_col_name)
    if np.isnan(y_test).any() or np.isinf(y_test).any():
        med_test = float(np.nanmedian(y_test)) 
        y_test = np.where(np.isfinite(y_test), y_test, med_test).astype(np.float32)

    test_set = SNPDataset(X_test, y_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    test_preds, test_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device, non_blocking=False).contiguous()
            feats = [m(batch_x) for m in trait_models]
            feat  = torch.mean(torch.stack(feats, dim=0), dim=0)
            pred  = head(feat).squeeze(1)

            test_preds.extend(pred.cpu().numpy())
            test_trues.extend(batch_y.numpy())

  
    pearson = float(pearsonr(test_trues, test_preds)[0])
    r2 = float(r2_score(test_trues, test_preds))
    mae = float(mean_absolute_error(test_trues, test_preds))
    rmse = float(np.sqrt(mean_squared_error(test_trues, test_preds)))

    print(f"  [Fine-Tune Test Metrics] Pearson: {pearson:.4f} | R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

   
    res_df = pd.DataFrame([{
        'Task': target_trait,
        'Model': 'Fine-Tuned',
        'Pearson': pearson,
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse
    }])
   
    res_df.to_csv(os.path.join(out_dir, 'fine_tuned_res.csv'), index=False)
