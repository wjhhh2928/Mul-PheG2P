import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold


from utils.data_loader import SNPDataset
from utils.metrics import pearson_loss
from models import SNPEncoderCNN  


import os as _os, torch as _torch
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

_os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
try:
    import torch.backends.mkldnn as _mkldnn
    _mkldnn.enabled = False
except Exception:
    pass
import torch.backends.cudnn as _cudnn
_cudnn.benchmark = False
_cudnn.deterministic = True
_torch.set_num_threads(1)


# ===================== Utility function =====================

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


# ===================== Training: Multi-trait Pretraining =====================

def train(snp, phe, source_traits, args):

    
    seed           = args['seed']
    lr             = args['lr']
    batch_size     = args['batch_size']
    pretrain_epoch = args['pretrain_epoch']
    input_len      = args['input_len']
    conv_channels  = args['conv_channels']
    kernel_size    = args['kernel_size']
    stride         = args['stride']
    hidden_dim     = args['hidden_dim']
    out_dir        = args.get('output', './outputs/pretrain')
    cv_folds       = args.get('cv_folds', 5)

    print(f"\n[Pretrain] Receive parameters: seed={seed}, lr={lr}")
    
    _ = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    
    device_req = str(args.get('device', 'auto')).lower()
    if device_req == 'cpu':
        device = torch.device("cpu")
    elif device_req == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== Dimension self-check ======
    if hasattr(snp, "shape"):
        snp_cols = snp.shape[1]
        print(f"[Check] input_len={input_len}, SNP cols={snp_cols}")
        if snp_cols != input_len:
            print(f"[Warn] input_len is inconsistent with the number of data columns：{input_len} vs {snp_cols}，it will be based on the number of data columns.")
            input_len = snp_cols  

    
    total_idx = np.arange(len(snp))
    pretrain_idx, _ = train_test_split(total_idx, test_size=0.3, random_state=seed)

    for trait_idx, trait_name in enumerate(source_traits):
        print(f"Pretraining for trait: {trait_name}")

        # ========= Model and Optimizer =========
        model = SNPEncoderCNN(input_len, conv_channels, kernel_size, stride, hidden_dim).to(device)
        head  = nn.Linear(hidden_dim, 1).to(device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr)

        # ========= Data preparation (forced float32, contiguous memory) =========
        X_train = _select_X_rows(snp, pretrain_idx)
        if hasattr(X_train, "to_numpy"):
            X_train = X_train.to_numpy(copy=True)  # pandas -> numpy
        X_train = np.asarray(X_train, dtype=np.float32, order="C")  

        y_train = _select_y_rows_col(phe, pretrain_idx, trait_idx, col_name=trait_name)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)

        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"[ShapeError] The number of rows in X_train {X_train.shape[0]} does not match y_train {y_train.shape[0]}.")  
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            print("[Warn] NaN/Inf detected in X_train, will be filled with 0.")  
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isnan(y_train).any() or np.isinf(y_train).any():
            print("[Warn] NaN/Inf detected in y_train, will be filled with the median.")
            med = float(np.nanmedian(y_train))
            y_train = np.where(np.isfinite(y_train), y_train, med).astype(np.float32)

        train_set = SNPDataset(X_train, y_train)

        # ========= DataLoader =========
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,            
            pin_memory=False,         
            persistent_workers=False
        )

        # ========= DryRun =========
        try:
            bx, by = next(iter(train_loader))
            print(f"[DryRun] First batch X={tuple(bx.shape)} Y={tuple(by.shape)} dtype={bx.dtype}")
            bx = bx.to(device, non_blocking=False).contiguous()
            by = by.to(device, non_blocking=False).contiguous()
            with torch.no_grad():
                _ = head(model(bx))
        except Exception as e:
            raise RuntimeError(f"[DryRunError] First batch forward failed: {e}")

        # ========= Training loop (add CUDA synchronization points to locate crashes) =========
        mse = nn.MSELoss()
        for epoch in range(pretrain_epoch):
            model.train()
            head.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                try:
                    batch_x = batch_x.to(device, non_blocking=False).contiguous()
                    batch_y = batch_y.to(device, non_blocking=False).contiguous()

                    feat = model(batch_x)                 # [B, hidden_dim]
                    pred = head(feat).squeeze(1)          # [B]
                    loss = mse(pred, batch_y) + 0.2 * pearson_loss(pred, batch_y)

                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    optimizer.step()

                    if device.type == "cuda" and (i % 100 == 0):
                        torch.cuda.synchronize()

                except Exception as e:
                    print(f"[CrashHint] epoch={epoch} iter={i}")
                    raise

            if (epoch + 1) % 10 == 0:
                print(f"[Epoch {epoch+1}/{pretrain_epoch}] loss={float(loss):.6f}")

        # ========= Save =========
        os.makedirs(out_dir, exist_ok=True)
        save_name   = _safe_filename(trait_name)
        heldout_idx = np.array([i for i in total_idx if i not in set(pretrain_idx)], dtype=np.int64)

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'head_state_dict' : head.state_dict(),
                'config': {
                    'input_len': input_len,
                    'conv_channels': conv_channels,
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'hidden_dim': hidden_dim,
                },
                'train_idx': np.asarray(pretrain_idx, dtype=np.int64),
                'heldout_idx': heldout_idx
            },
            os.path.join(out_dir, f'pretrain_{save_name}.pth')
        )
        print(f"[Save] -> {os.path.join(out_dir, f'pretrain_{save_name}.pth')}")


# ===================== Prediction: Zero-shot (directly using pre-trained weights) =====================

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import pandas as pd

def predict_with_pretrained_target(snp, phe, target_trait, args):
    """
   Directly use the pre-trained weights of [target trait] for inference and evaluation (without any fine-tuning training).
   - If the checkpoint contains heldout_idx, evaluate only on the held-out set to avoid leakage.
   - Export two files to args['output']:
   1) pretrained_target_res.csv (overall metrics)
   2) pretrained_target_true_vs_pred.csv (sample-level true values and predictions)
    """
   
    device_req = str(args.get('device', 'auto')).lower()
    if device_req == 'cpu':
        device = torch.device("cpu")
    elif device_req == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir     = args.get('output', './outputs/pretrain')
    target_name = str(target_trait)

    
    ckpt_path = os.path.join(out_dir, f"pretrain_{target_name.replace(' ', '_').replace('/', '_')}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[ZeroShot] Pretrained weights not found：{ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    
    cfg   = ckpt['config']
    model = SNPEncoderCNN(
        cfg['input_len'], cfg['conv_channels'], cfg['kernel_size'], cfg['stride'], cfg['hidden_dim']
    ).to(device)
    head  = nn.Linear(cfg['hidden_dim'], 1).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    head.load_state_dict(ckpt['head_state_dict'])
    model.eval(); head.eval()

    
    if 'heldout_idx' in ckpt and ckpt['heldout_idx'] is not None and len(ckpt['heldout_idx']) > 0:
        eval_idx = np.asarray(ckpt['heldout_idx'], dtype=np.int64)
        print(f"[ZeroShot] Number of held-out samples used：{len(eval_idx)}")
    else:
        eval_idx = np.arange(len(snp))
        print(f"[ZeroShot] Held-out index not found, defaulting to evaluation on all {len(eval_idx)} samples.")

    # 4) （forced float32 + C-order）
    if hasattr(snp, "iloc"):
        X = snp.iloc[eval_idx, :].to_numpy(copy=True)
    else:
        X = snp[eval_idx]
    X = np.asarray(X, dtype=np.float32, order="C")


    if hasattr(phe, "iloc"):
        y_true = phe.loc[eval_idx, target_trait].to_numpy(dtype=np.float32).reshape(-1)
    else:
    
        y_true = np.asarray(phe[eval_idx, int(target_trait)], dtype=np.float32).reshape(-1)

    # 5) Forward reasoning
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32, device=device).contiguous()
        feat = model(xt)
        y_pred = head(feat).squeeze(1).detach().cpu().numpy().astype(np.float64)

    # 6) Indicator
    r2   = r2_score(y_true, y_pred)
    p    = pearsonr(y_true, y_pred)[0]
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[ZeroShot] R²={r2:.4f} | Pearson={p:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f}")

    # 7) Export
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({
        "Task": [target_name],
        "Model": ["Pretrained-Target(ZeroShot)"],
        "R2": [r2], "Pearson": [p], "MAE": [mae], "RMSE": [rmse]
    }).to_csv(os.path.join(out_dir, "pretrained_target_res.csv"), index=False)

    pd.DataFrame({
        "sample_index": eval_idx, "y_true": y_true, "y_pred": y_pred
    }).to_csv(os.path.join(out_dir, "pretrained_target_true_vs_pred.csv"), index=False)

    print(f"[ZeroShot] Metric has been saved: {os.path.join(out_dir,'pretrained_target_res.csv')}")
    print(f"[ZeroShot] Results for each sample have been saved: {os.path.join(out_dir,'pretrained_target_true_vs_pred.csv')}")
