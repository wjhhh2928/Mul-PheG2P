
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils.data_loader import SNPDataset
from utils.metrics import pearson_loss
from models import SNPEncoderCNN


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
                raise ValueError("Please provide the column name or column position to extract the target traits from phe.")
            y = Y.iloc[rows, int(col_idx)]
    else:                    # numpy
        if col_idx is None:
            raise ValueError("When phe is an ndarray, the column position col_idx must be provided.")
        y = Y[rows, int(col_idx)]
    return np.asarray(y, dtype=np.float32).reshape(-1)

def _safe_filename(name: str) -> str:
    return str(name).replace(" ", "_").replace("/", "_")

# ---------- 训练主函数 ----------
def train(snp, phe, target_trait, source_traits, args):
    """
    snp: DataFrame or ndarray，shape [N, ...]
    phe: DataFrame or ndarray，shape [N, T]
    target_trait
    source_traits
    args: dict 
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    seed          = args['seed']
    lr            = args['lr']
    batch_size    = args['batch_size']
    ft_epoch      = args['ft_epoch']
    input_len     = args['input_len']
    conv_channels = args['conv_channels']
    kernel_size   = args['kernel_size']
    stride        = args['stride']
    hidden_dim    = args['hidden_dim']
    out_dir       = args.get('output', './outputs/pretrain')

    
    total_idx = np.arange(len(snp))
    train_pool, test_idx = train_test_split(total_idx, test_size=0.3, random_state=seed)
    train_idx,  val_idx  = train_test_split(train_pool, test_size=0.3, random_state=seed + 1)
    # now：train ≈ 49%，val ≈ 21%，test = 30%

    
    trait_models = []
    for trait_name in source_traits:
        ckpt_path = os.path.join(out_dir, f'pretrain_{_safe_filename(trait_name)}.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Pre-trained weights not found：{ckpt_path}")

        # checkpoint = torch.load(ckpt_path, map_location='cpu')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)


        model = SNPEncoderCNN(input_len, conv_channels, kernel_size, stride, hidden_dim).to(device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)

        
        for p in model.parameters():
            p.requires_grad = False
        model.eval()  
        trait_models.append(model)

    # ---------- predict head----------
    head = nn.Linear(hidden_dim, 1).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    mse = nn.MSELoss()

    
    target_col_idx  = target_trait if isinstance(target_trait, (int, np.integer)) else None
    target_col_name = target_trait if isinstance(target_trait, str) else None

    X_train = _select_X_rows(snp, train_idx)
    y_train = _select_y_rows_col(phe, train_idx, col_idx=target_col_idx, col_name=target_col_name)
    X_val   = _select_X_rows(snp, val_idx)
    y_val   = _select_y_rows_col(phe, val_idx,   col_idx=target_col_idx, col_name=target_col_name)

    train_set = SNPDataset(X_train, y_train)
    val_set   = SNPDataset(X_val,   y_val)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

   
    best_val_loss = float('inf')
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(ft_epoch):
        head.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            with torch.no_grad():
                feats = [m(batch_x) for m in trait_models]     # list of [B, hidden_dim]
                feat  = torch.mean(torch.stack(feats, dim=0), dim=0)  # [B, hidden_dim]

            pred = head(feat).squeeze(1)  # [B]
            loss = mse(pred, batch_y) + 0.2 * pearson_loss(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                feats = [m(batch_x) for m in trait_models]
                feat  = torch.mean(torch.stack(feats, dim=0), dim=0)
                pred  = head(feat).squeeze(1)
                val_loss += mse(pred, batch_y).item()
        val_loss /= max(1, len(val_loader))

        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    'model_state_dicts': [m.state_dict() for m in trait_models],
                    'head_state_dict': head.state_dict(),
                    'config': {
                        'input_len': input_len,
                        'conv_channels': conv_channels,
                        'kernel_size': kernel_size,
                        'stride': stride,
                        'hidden_dim': hidden_dim,
                    }
                },
                os.path.join(out_dir, f'fine_tuned_{_safe_filename(str(target_trait))}.pth')
            )
