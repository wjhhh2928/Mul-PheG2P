import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
import pandas as pd

from models import SNPEncoderCNN  

# ================================ Utility function ================================

def _safe_filename(name: str) -> str:
    return str(name).replace(" ", "_").replace("/", "_")

def _to_input_tensor(X_np, device):
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    if X.ndim == 2:
        X = X.unsqueeze(1)
    elif X.ndim == 3:
        pass
    elif X.ndim == 4 and X.shape[1] == 1 and X.shape[2] == 1:
        X = X.squeeze(2)
    else:
        raise ValueError(f"Expected [B,L]/[B,1,L]/[B,1,1,L], got {tuple(X.shape)}")
    assert X.ndim == 3 and X.shape[1] == 1
    return X

def _safe_load_checkpoint(path):
    import warnings
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.load(path, map_location="cpu", weights_only=False)
    # except TypeError:
    #     return torch.load(path, map_location="cpu")

def _metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

def _nan_fix(y, fallback=None):
    if np.isnan(y).any():
        if fallback is None:
            fallback = np.nanmean(y)
        y = np.nan_to_num(y, nan=float(fallback))
    return y

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _load_encoders_from_pretrain(source_traits: List[str],
                                 input_len: int, conv_channels: int, kernel_size: int,
                                 stride: int, hidden_dim: int, out_dir: str, device) -> List[nn.Module]:
    encoders = []
    for trait in source_traits:
        ckpt = os.path.join(out_dir, f"pretrain_{_safe_filename(trait)}.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Pre-trained weights not found: {ckpt}")
        ckp = _safe_load_checkpoint(ckpt)
        enc = SNPEncoderCNN(input_len, conv_channels, kernel_size, stride, hidden_dim).to(device)
        enc.load_state_dict(ckp["model_state_dict"], strict=True)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False  
        encoders.append(enc)
    return encoders

def _build_embeddings(encoders: List[nn.Module], snp: np.ndarray, idx: np.ndarray,
                      device) -> torch.Tensor:
    X = _to_input_tensor(snp[idx], device)
    with torch.no_grad():
        feats = [enc(X) for enc in encoders]  # each: [B, H]
    emb = torch.stack(feats, dim=1)           # [B, T, H]
    return emb

# ================================ Embedding Fusion Model ================================

class EmbeddingAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_traits: int, dropout: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_traits = num_traits
        self.ln = nn.LayerNorm(hidden_dim)

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, emb: torch.Tensor):
        emb = self.ln(emb)
        scores = self.scorer(emb)               # [B, T, 1]
        weights = torch.softmax(scores, dim=1)  # [B, T, 1]
        fused = (emb * weights).sum(dim=1)      # [B, H]
        pred = self.reg_head(fused).squeeze(-1) # [B]
        return pred, weights.squeeze(-1)        # [B, T]

# ================================ 1. Embedding Fusion Strategy ================================

def train_embedding_fusion_grouped(snp, phe, target_trait, source_traits, train_idx, val_idx, test_idx, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract parameters
    lr            = float(args.get("lr", 1e-3))
    fu_epoch      = int(args.get("fu_epoch", 500))
    patience      = int(args.get("patience", 50))
    input_len     = int(args["input_len"])
    conv_channels = int(args["conv_channels"])
    kernel_size   = int(args["kernel_size"])
    stride        = int(args["stride"])
    hidden_dim    = int(args["hidden_dim"])
    out_dir       = args.get("output", "./outputs/pretrain")
    dropout       = float(args.get("fusion_dropout", 0.5))
    weight_decay  = float(args.get("weight_decay", 5e-4))
    attn_entropy_lambda = float(args.get("attn_entropy_lambda", 1e-3))
    apply_calibration = bool(args.get("apply_calibration", True))

    _ensure_dir(out_dir)

    y_all = phe[target_trait].values.astype(np.float32)
    y_all = _nan_fix(y_all)

    print(f"\n[Embedding Fusion] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Load frozen encoders and build embeddings
    encoders = _load_encoders_from_pretrain(source_traits, input_len, conv_channels,
                                            kernel_size, stride, hidden_dim, out_dir, device)
    emb_train = _build_embeddings(encoders, snp, train_idx, device)  # [Bt, T, H]
    emb_val   = _build_embeddings(encoders, snp, val_idx, device)    # [Bv, T, H]
    emb_test  = _build_embeddings(encoders, snp, test_idx, device)   # [Be, T, H]

    # Data scaling
    y_train_raw = y_all[train_idx]
    y_val_raw   = y_all[val_idx]
    y_test_raw  = y_all[test_idx]
    
    scaler_y = MinMaxScaler().fit(y_train_raw.reshape(-1, 1))
    y_train_s = torch.tensor(scaler_y.transform(y_train_raw.reshape(-1, 1)).ravel(), dtype=torch.float32, device=device)
    y_val_s   = torch.tensor(scaler_y.transform(y_val_raw.reshape(-1, 1)).ravel(), dtype=torch.float32, device=device)

    # Fusion model and optimizer
    fusion = EmbeddingAttentionFusion(hidden_dim=hidden_dim, num_traits=len(source_traits), dropout=dropout).to(device)
    optimizer = torch.optim.Adam(fusion.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss(beta=0.5)

    best = {"pearson": -1.0, "r2": -1.0, "mae": 1e9, "rmse": 1e9, "epoch": -1}
    best_state = None
    best_calib = {"a": 1.0, "b": 0.0}  
    wait = 0

    for epoch in range(fu_epoch):
        fusion.train()
        pred_tr, weights_tr = fusion(emb_train)
        loss = criterion(pred_tr, y_train_s)

        if attn_entropy_lambda > 0:
            attn_reg = (weights_tr * torch.log(weights_tr + 1e-8)).sum(dim=1).mean()
            loss = loss + attn_entropy_lambda * attn_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        fusion.eval()
        with torch.no_grad():
            pred_val_s, weights_val = fusion(emb_val)
            y_pred_val = scaler_y.inverse_transform(pred_val_s.detach().cpu().numpy().reshape(-1, 1)).ravel()
            y_true_val = y_val_raw

        m = _metrics(y_true_val, y_pred_val)

        if m["pearson"] > best["pearson"]:
            best.update(m); best["epoch"] = epoch + 1
            best_state = {k: v.cpu() for k, v in fusion.state_dict().items()}
            torch.save(best_state, os.path.join(out_dir, "fusion_model_embedding.pth"))
            wait = 0

            # Linear calibration mapping
            try:
                a, b = np.polyfit(y_pred_val, y_true_val, 1)
                best_calib = {"a": float(a), "b": float(b)}
            except Exception:
                best_calib = {"a": 1.0, "b": 0.0}

            # Save attention weights
            w_mean = weights_val.mean(dim=0).detach().cpu().numpy()
            pd.DataFrame({"Trait": source_traits, "Weight": w_mean}).to_csv(os.path.join(out_dir, "fusion_attention_weights.csv"), index=False)
        else:
            wait += 1
            if wait >= patience:
                print(f"[EarlyStopping] epoch={epoch+1}, best_pearson={best['pearson']:.4f}")
                break

    # Final Test Evaluation
    if best_state is not None:
        fusion.load_state_dict(best_state)
    fusion.eval()
    with torch.no_grad():
        pred_test_s, _ = fusion(emb_test)
        
    y_pred_test = scaler_y.inverse_transform(pred_test_s.detach().cpu().numpy().reshape(-1, 1)).ravel()
    y_true_test = y_test_raw

    test_raw = _metrics(y_true_test, y_pred_test)

    if apply_calibration and np.isfinite(best_calib["a"]) and np.isfinite(best_calib["b"]):
        y_pred_test_cal = best_calib["a"] * y_pred_test + best_calib["b"]
        test_cal = _metrics(y_true_test, y_pred_test_cal)
    else:
        y_pred_test_cal = y_pred_test.copy()
        test_cal = test_raw

    print(f"[Embedding Fusion] Test Metrics - Pearson: {test_cal['pearson']:.4f}, R2: {test_cal['r2']:.4f}, MAE: {test_cal['mae']:.4f}, RMSE: {test_cal['rmse']:.4f}")

    # Save final results
    pd.DataFrame([{
        "best_epoch": best["epoch"],
        "val_pearson": best["pearson"], "val_r2": best["r2"],
        "val_mae": best["mae"], "val_rmse": best["rmse"],
        "test_pearson_raw": test_raw["pearson"], "test_r2_raw": test_raw["r2"],
        "test_mae_raw": test_raw["mae"], "test_rmse_raw": test_raw["rmse"],
        "test_pearson_cal": test_cal["pearson"], "test_r2_cal": test_cal["r2"],
        "test_mae_cal": test_cal["mae"], "test_rmse_cal": test_cal["rmse"],
    }]).to_csv(os.path.join(out_dir, "fusion_val_best.csv"), index=False)

    return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx, "best_val": best, "test_raw": test_raw, "test_cal": test_cal}


# ================================ 2. OOF Stacking Strategy ================================

class _HeadRegressor(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        return self.fc(x).squeeze(-1)

def _fit_base_model_for_trait(enc_init, head_init, X_idx, y_raw, snp, device, args):
    lr_head = float(args.get("oof_ft_lr_head", 1e-3))
    wd      = float(args.get("oof_ft_weight_decay", 5e-4))
    epochs  = int(args.get("oof_ft_epoch", 50))
    batch_size = int(args.get("oof_ft_batch_size", 256))

    X_t = _to_input_tensor(snp[X_idx], device)
    y_raw = y_raw.astype(np.float32)
    y_raw = _nan_fix(y_raw, fallback=np.nanmean(y_raw))
    scaler_y = MinMaxScaler().fit(y_raw.reshape(-1, 1))
    y_s = torch.tensor(scaler_y.transform(y_raw.reshape(-1, 1)).ravel(), dtype=torch.float32, device=device)

    enc = SNPEncoderCNN(**enc_init["cfg"]).to(device)
    enc.load_state_dict(enc_init["state"])
    head = _HeadRegressor(enc_init["cfg"]["hidden_dim"]).to(device)
    if head_init is not None: head.load_state_dict(head_init)

    enc.eval()
    for p in enc.parameters(): p.requires_grad = False
    opt = torch.optim.Adam(head.parameters(), lr=lr_head, weight_decay=wd)
    criterion = nn.SmoothL1Loss(beta=0.5)

    n = X_t.shape[0]
    idx_all = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx_all)
        for s in range(0, n, batch_size):
            sl = idx_all[s:s+batch_size]
            x_b = X_t[sl]
            with torch.no_grad():
                feat = enc(x_b)
            pred = head(feat)
            loss = criterion(pred, y_s[sl])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return enc, head, scaler_y

def _predict_with_base(enc, head, snp, idx, device):
    X = _to_input_tensor(snp[idx], device)
    with torch.no_grad():
        pred = head(enc(X))
    return pred.detach().cpu().numpy().ravel()

def build_oof_features(snp, phe, source_traits, train_idx, val_idx, test_idx, args, device, out_dir):
    kfold = int(args.get("oof_folds", 5))
    seed  = int(args.get("seed", 42))

    enc_inits = []
    head_states = []
    for trait in source_traits:
        ckpt = os.path.join(out_dir, f"pretrain_{_safe_filename(trait)}.pth")
        ckp = _safe_load_checkpoint(ckpt)
        cfg = dict(input_len=int(args["input_len"]), conv_channels=int(args["conv_channels"]),
                   kernel_size=int(args["kernel_size"]), stride=int(args["stride"]), hidden_dim=int(args["hidden_dim"]))
        enc_inits.append({"cfg": cfg, "state": ckp["model_state_dict"]})
        head_states.append(ckp.get("head_state_dict", None))

    n_train = len(train_idx)
    X_train_oof = np.zeros((n_train, len(source_traits)), dtype=np.float32)
    X_val_pred  = np.zeros((len(val_idx),  len(source_traits)), dtype=np.float32)
    X_test_pred = np.zeros((len(test_idx), len(source_traits)), dtype=np.float32)

    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)

    for j, trait in enumerate(source_traits):
        y_full = _nan_fix(phe[trait].values.astype(np.float32))

        # Nested CV strictly inside train_idx
        for tr_rel, va_rel in kf.split(train_idx):
            tr_idx = train_idx[tr_rel]
            va_idx = train_idx[va_rel]
            enc, head, _ = _fit_base_model_for_trait(enc_inits[j], head_states[j], tr_idx, y_full[tr_idx], snp, device, args)
            X_train_oof[va_rel, j] = _predict_with_base(enc, head, snp, va_idx, device).astype(np.float32)

        # Full fit on train_idx for Val and Test
        enc_full, head_full, _ = _fit_base_model_for_trait(enc_inits[j], head_states[j], train_idx, y_full[train_idx], snp, device, args)
        X_val_pred[:, j] = _predict_with_base(enc_full, head_full, snp, val_idx, device).astype(np.float32)
        X_test_pred[:, j] = _predict_with_base(enc_full, head_full, snp, test_idx, device).astype(np.float32)

    return X_train_oof, X_val_pred, X_test_pred

def train_oof_stacking_grouped(snp, phe, target_trait, source_traits, train_idx, val_idx, test_idx, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.get("output", "./outputs/pretrain")
    _ensure_dir(out_dir)

    print(f"\n[OOF Stacking] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    X_train_oof, X_val_pred, X_test_pred = build_oof_features(
        snp=snp, phe=phe, source_traits=source_traits, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        args=args, device=device, out_dir=out_dir
    )

    scaler_x = StandardScaler().fit(X_train_oof)
    X_train_n = scaler_x.transform(X_train_oof)
    X_val_n   = scaler_x.transform(X_val_pred)
    X_test_n  = scaler_x.transform(X_test_pred)

    y_all = _nan_fix(phe[target_trait].values.astype(np.float32))
    y_train_raw = y_all[train_idx]
    y_val_raw   = y_all[val_idx]
    y_test_raw  = y_all[test_idx]
    
    scaler_y = MinMaxScaler().fit(y_train_raw.reshape(-1, 1))
    y_train_s = scaler_y.transform(y_train_raw.reshape(-1, 1)).ravel()

    ridge = RidgeCV(alphas=np.logspace(-4, 2, 16), fit_intercept=True, cv=5)
    ridge.fit(X_train_n, y_train_s)

    y_pred_val = scaler_y.inverse_transform(ridge.predict(X_val_n).reshape(-1, 1)).ravel()
    m_val = _metrics(y_val_raw, y_pred_val)

    y_pred_test = scaler_y.inverse_transform(ridge.predict(X_test_n).reshape(-1, 1)).ravel()
    m_test = _metrics(y_test_raw, y_pred_test)

    print(f"[OOF Stacking] Test Metrics - Pearson: {m_test['pearson']:.4f}, R2: {m_test['r2']:.4f}, MAE: {m_test['mae']:.4f}, RMSE: {m_test['rmse']:.4f}")

    pd.DataFrame([{
        "val_pearson": m_val["pearson"], "val_r2": m_val["r2"],
        "val_mae": m_val["mae"], "val_rmse": m_val["rmse"],
        "test_pearson": m_test["pearson"], "test_r2": m_test["r2"],
        "test_mae": m_test["mae"], "test_rmse": m_test["rmse"]
    }]).to_csv(os.path.join(out_dir, "fusion_val_best_oof.csv"), index=False)

    return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx, "val": m_val, "test": m_test}


# ================================ 3. Main Entry ================================

def train(snp, phe, target_trait, source_traits, train_idx, val_idx, test_idx, args):
    mode = str(args.get("fusion_mode", "embedding")).lower()
    if mode == "embedding":
        return train_embedding_fusion_grouped(snp, phe, target_trait, source_traits, train_idx, val_idx, test_idx, args)
    elif mode == "oof":
        return train_oof_stacking_grouped(snp, phe, target_trait, source_traits, train_idx, val_idx, test_idx, args)
    else:
        raise ValueError(f"Unknown fusion_mode: {mode}. Use 'embedding' or 'oof'.")
