import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict

from sklearn.model_selection import GroupShuffleSplit, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

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

# ================================ Segmentation by groups ================================

def build_groups(phe: pd.DataFrame, snp: np.ndarray, args) -> np.ndarray:

    if args.get("group_col") and args["group_col"] in phe.columns:
        g = phe[args["group_col"]].values
        
        _, groups = np.unique(g, return_inverse=True)
        return groups.astype(int)

    
    n_groups = int(args.get("n_groups", 10))
    n_pca = int(args.get("n_pca", 16))
    rnd = int(args.get("seed", 42))

    
    pca = PCA(n_components=min(n_pca, snp.shape[1]), random_state=rnd)
    Z = pca.fit_transform(snp)
    km = KMeans(n_clusters=n_groups, n_init=10, random_state=rnd)
    groups = km.fit_predict(Z)
    return groups.astype(int)

def group_split_indices(snp: np.ndarray, groups: np.ndarray, seed=42,
                        test_size=0.3, val_size=0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_pool_idx, test_idx = next(gss.split(snp, groups=groups))

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + 1)
    train_idx_rel, val_idx_rel = next(gss_val.split(snp[train_pool_idx], groups=groups[train_pool_idx]))
    train_idx = train_pool_idx[train_idx_rel]
    val_idx = train_pool_idx[val_idx_rel]
    return train_idx, val_idx, test_idx

# ================================ Embedding  ================================

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
        """
        emb: [B, T, H]
        return: pred [B], weights [B, T]
        """
        # NEW: LayerNorm
        emb = self.ln(emb)

        scores = self.scorer(emb)               # [B, T, 1]
        weights = torch.softmax(scores, dim=1)  # [B, T, 1]
        fused = (emb * weights).sum(dim=1)      # [B, H]
        pred = self.reg_head(fused).squeeze(-1) # [B]
        return pred, weights.squeeze(-1)        # [B, T]


def _load_encoders_from_pretrain(source_traits: List[str],
                                 input_len: int, conv_channels: int, kernel_size: int,
                                 stride: int, hidden_dim: int, out_dir: str, device) -> List[nn.Module]:
    encoders = []
    for trait in source_traits:
        ckpt = os.path.join(out_dir, f"pretrain_{_safe_filename(trait)}.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Pre-trained weights not found：{ckpt}")
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

def train_embedding_fusion_grouped(snp, phe, target_trait, source_traits, args):
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed          = int(args.get("seed", 42))
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
    test_size     = float(args.get("test_size", 0.3))
    val_size      = float(args.get("val_size", 0.3))
    
    attn_entropy_lambda = float(args.get("attn_entropy_lambda", 1e-3))
  
    apply_calibration = bool(args.get("apply_calibration", True))

    _ensure_dir(out_dir)


    y_all = phe[target_trait].values.astype(np.float32)
    y_all = _nan_fix(y_all)

   
    groups = build_groups(phe, snp, args)
    train_idx, val_idx, test_idx = group_split_indices(snp, groups, seed=seed,
                                                       test_size=test_size, val_size=val_size)

    
    encoders = _load_encoders_from_pretrain(source_traits, input_len, conv_channels,
                                            kernel_size, stride, hidden_dim, out_dir, device)
    emb_train = _build_embeddings(encoders, snp, train_idx, device)  # [Bt, T, H]
    emb_val   = _build_embeddings(encoders, snp, val_idx, device)    # [Bv, T, H]
    emb_test  = _build_embeddings(encoders, snp, test_idx, device)   # [Be, T, H]

    
    y_train_raw = y_all[train_idx]
    y_val_raw   = y_all[val_idx]
    y_test_raw  = y_all[test_idx]
    scaler_y = MinMaxScaler().fit(y_train_raw.reshape(-1, 1))
    y_train_s = torch.tensor(scaler_y.transform(y_train_raw.reshape(-1, 1)).ravel(),
                             dtype=torch.float32, device=device)
    y_val_s   = torch.tensor(scaler_y.transform(y_val_raw.reshape(-1, 1)).ravel(),
                             dtype=torch.float32, device=device)

    # fusion model
    fusion = EmbeddingAttentionFusion(hidden_dim=hidden_dim, num_traits=len(source_traits),
                                      dropout=dropout).to(device)
    optimizer = torch.optim.Adam(fusion.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss(beta=0.5)  # Huber

    best = {"pearson": -1.0, "r2": -1.0, "mae": 1e9, "rmse": 1e9, "epoch": -1}
    best_state = None
    best_calib = {"a": 1.0, "b": 0.0}  
    wait = 0

    for epoch in range(fu_epoch):
        fusion.train()
        pred_tr, weights_tr = fusion(emb_train)  # [B], [B,T]
        loss = criterion(pred_tr, y_train_s)

       
        if attn_entropy_lambda > 0:
            attn_reg = (weights_tr * torch.log(weights_tr + 1e-8)).sum(dim=1).mean()
            loss = loss + attn_entropy_lambda * attn_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validation
        fusion.eval()
        with torch.no_grad():
            pred_val_s, weights_val = fusion(emb_val)
            
            y_pred_val = scaler_y.inverse_transform(
                pred_val_s.detach().cpu().numpy().reshape(-1, 1)
            ).ravel()
            y_true_val = y_val_raw

        m = _metrics(y_true_val, y_pred_val)

        print(f"Epoch {epoch+1}: Val Pearson={m['pearson']:.4f} | R2={m['r2']:.4f} | "
              f"MAE={m['mae']:.4f} | RMSE={m['rmse']:.4f} | Loss={loss.item():.6f}")

    
        if m["pearson"] > best["pearson"]:
            best.update(m); best["epoch"] = epoch + 1
            best_state = {k: v.cpu() for k, v in fusion.state_dict().items()}
            torch.save(best_state, os.path.join(out_dir, "fusion_model_embedding.pth"))
            wait = 0

        
            try:
                a, b = np.polyfit(y_pred_val, y_true_val, 1)
                best_calib = {"a": float(a), "b": float(b)}
                pd.DataFrame([best_calib]).to_csv(
                    os.path.join(out_dir, "fusion_calibration.csv"), index=False
                )
            except Exception:
                
                best_calib = {"a": 1.0, "b": 0.0}

        
            w_mean = weights_val.mean(dim=0).detach().cpu().numpy()  # [T]
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(w_mean))
            bars = ax.bar(x, w_mean)
            ax.set_xticks(x)
            ax.set_xticklabels(source_traits, rotation=45, ha='right')
            ax.set_title("Embedding Fusion Attention Weights (with LayerNorm + Entropy Reg.)")
            ax.set_xlabel("Source Traits"); ax.set_ylabel("Weight")
            ymax = float(w_mean.max()) if w_mean.size > 0 else 1.0
            ax.set_ylim(0, ymax * 1.15)
            labels = [f"{v:.3f}" for v in w_mean]
            try:
                ax.bar_label(bars, labels=labels, padding=2)
            except Exception:
                for rect, lab in zip(bars, labels):
                    h = rect.get_height()
                    ax.text(rect.get_x()+rect.get_width()/2.0, h + ymax*0.01,
                            lab, ha='center', va='bottom', clip_on=False)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "fusion_attention_weights.png"),
                        dpi=200, bbox_inches='tight')
            plt.close(fig)

            pd.DataFrame({"Trait": source_traits, "Weight": w_mean}) \
              .to_csv(os.path.join(out_dir, "fusion_attention_weights.csv"), index=False)

        else:
            wait += 1
            if wait >= patience:
                print(f"[EarlyStopping] epoch={epoch+1}, best_pearson={best['pearson']:.4f}")
                break

   
    if best_state is not None:
        fusion.load_state_dict(best_state)
    fusion.eval()
    with torch.no_grad():
        pred_test_s, _ = fusion(emb_test)
    y_pred_test = scaler_y.inverse_transform(
        pred_test_s.detach().cpu().numpy().reshape(-1, 1)
    ).ravel()
    y_true_test = y_test_raw

   
    test_raw = _metrics(y_true_test, y_pred_test)

    if apply_calibration and np.isfinite(best_calib["a"]) and np.isfinite(best_calib["b"]):
        y_pred_test_cal = best_calib["a"] * y_pred_test + best_calib["b"]
        test_cal = _metrics(y_true_test, y_pred_test_cal)
    else:
        y_pred_test_cal = y_pred_test.copy()
        test_cal = test_raw

   
    print("\nBest Fusion Metrics (on Val):")
    print(f"R²: {best['r2']:.4f}\nPearson: {best['pearson']:.4f}\nMAE: {best['mae']:.4f}\nRMSE: {best['rmse']:.4f}")

    print("\nTest Metrics (on Held-out Test) — BEFORE calibration:")
    print(f"R²: {test_raw['r2']:.4f}\nPearson: {test_raw['pearson']:.4f}\nMAE: {test_raw['mae']:.4f}\nRMSE: {test_raw['rmse']:.4f}")

    print("\nTest Metrics (on Held-out Test) — AFTER calibration:")
    print(f"R²: {test_cal['r2']:.4f}\nPearson: {test_cal['pearson']:.4f}\nMAE: {test_cal['mae']:.4f}\nRMSE: {test_cal['rmse']:.4f}")

    _ensure_dir(out_dir)
  
    pd.DataFrame({
        "sample_index": test_idx,
        "y_true": y_true_test.astype(np.float64),
        "y_pred": y_pred_test.astype(np.float64),
        "y_pred_cal": y_pred_test_cal.astype(np.float64)
    }).to_csv(os.path.join(out_dir, "test_true_vs_pred.csv"), index=False)

 
    pd.DataFrame([{
        "best_epoch": best["epoch"],
        "val_pearson": best["pearson"],
        "val_r2": best["r2"],
        "val_mae": best["mae"],
        "val_rmse": best["rmse"],
        "test_pearson_raw": test_raw["pearson"],
        "test_r2_raw": test_raw["r2"],
        "test_mae_raw": test_raw["mae"],
        "test_rmse_raw": test_raw["rmse"],
        "test_pearson_cal": test_cal["pearson"],
        "test_r2_cal": test_cal["r2"],
        "test_mae_cal": test_cal["mae"],
        "test_rmse_cal": test_cal["rmse"],
        "cal_a": best_calib["a"],
        "cal_b": best_calib["b"]
    }]).to_csv(os.path.join(out_dir, "fusion_val_best.csv"), index=False)

    return {
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "best_val": best, "test_raw": test_raw, "test_cal": test_cal,
        "calibration": best_calib
    }

# ================================================================

class _HeadRegressor(nn.Module):
   
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)

def _fit_base_model_for_trait(enc_init: nn.Module, head_init: nn.Module,
                              X_idx: np.ndarray, y_raw: np.ndarray,
                              snp: np.ndarray, device, args) -> Tuple[nn.Module, nn.Module, MinMaxScaler]:
    
    lr_enc  = float(args.get("oof_ft_lr_enc", 1e-4))
    lr_head = float(args.get("oof_ft_lr_head", 1e-3))
    wd      = float(args.get("oof_ft_weight_decay", 5e-4))
    epochs  = int(args.get("oof_ft_epoch", 50))
    unfreeze_encoder = bool(args.get("oof_ft_unfreeze_encoder", False))
    batch_size = int(args.get("oof_ft_batch_size", 256))

    
    X_t = _to_input_tensor(snp[X_idx], device)
    y_raw = y_raw.astype(np.float32)
    y_raw = _nan_fix(y_raw, fallback=np.nanmean(y_raw))
    scaler_y = MinMaxScaler().fit(y_raw.reshape(-1, 1))
    y_s = torch.tensor(scaler_y.transform(y_raw.reshape(-1, 1)).ravel(),
                       dtype=torch.float32, device=device)

   
    enc = SNPEncoderCNN(**enc_init["cfg"]).to(device)
    enc.load_state_dict(enc_init["state"])
    head = _HeadRegressor(enc_init["cfg"]["hidden_dim"]).to(device)
    if head_init is not None:
        head.load_state_dict(head_init)

    if not unfreeze_encoder:
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
        params = list(head.parameters())
        lr = lr_head
    else:
        enc.train()
        for p in enc.parameters():
            p.requires_grad = True
        params = list(enc.parameters()) + list(head.parameters())
        lr = lr_enc

    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    criterion = nn.SmoothL1Loss(beta=0.5)

    
    n = X_t.shape[0]
    idx_all = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx_all)
        for s in range(0, n, batch_size):
            sl = idx_all[s:s+batch_size]
            x_b = X_t[sl]
            with torch.no_grad():
                feat = enc(x_b)          # [b, H]
            if unfreeze_encoder:
                
                feat = enc(x_b)
            pred = head(feat)
            loss = criterion(pred, y_s[sl])

            opt.zero_grad()
            loss.backward()
            opt.step()
    return enc, head, scaler_y

def _predict_with_base(enc: nn.Module, head: nn.Module,
                       snp: np.ndarray, idx: np.ndarray,
                       device) -> np.ndarray:
    X = _to_input_tensor(snp[idx], device)
    with torch.no_grad():
        feat = enc(X)
        pred = head(feat)
    return pred.detach().cpu().numpy().ravel()

def build_oof_features(snp, phe, source_traits: List[str],
                       train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
                       args, device, out_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    - X_train_oof  [len(train_idx), T]
    - X_val_pred   [len(val_idx),   T]   
    - X_test_pred  [len(test_idx),  T]   
    """
    kfold = int(args.get("oof_folds", 5))
    seed  = int(args.get("seed", 42))

    
    enc_inits = []
    head_states = []
    for trait in source_traits:
        ckpt = os.path.join(out_dir, f"pretrain_{_safe_filename(trait)}.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Pre-trained weights not found: {ckpt}")
        ckp = _safe_load_checkpoint(ckpt)

        cfg = dict(
            input_len=int(args["input_len"]),
            conv_channels=int(args["conv_channels"]),
            kernel_size=int(args["kernel_size"]),
            stride=int(args["stride"]),
            hidden_dim=int(args["hidden_dim"])
        )
        enc_inits.append({"cfg": cfg, "state": ckp["model_state_dict"]})
        head_states.append(ckp.get("head_state_dict", None))

    n_train = len(train_idx)
    X_train_oof = np.zeros((n_train, len(source_traits)), dtype=np.float32)
    X_val_pred  = np.zeros((len(val_idx),  len(source_traits)), dtype=np.float32)
    X_test_pred = np.zeros((len(test_idx), len(source_traits)), dtype=np.float32)

    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)


    for j, trait in enumerate(source_traits):
        print(f"[OOF] Building for source trait: {trait}")
        y_full = phe[trait].values.astype(np.float32)
        y_full = _nan_fix(y_full)

       
        for tr_rel, va_rel in kf.split(train_idx):
            tr_idx = train_idx[tr_rel]
            va_idx = train_idx[va_rel]

           
            enc, head, scaler_yj = _fit_base_model_for_trait(
                enc_init=enc_inits[j], head_init=head_states[j],
                X_idx=tr_idx, y_raw=y_full[tr_idx],
                snp=snp, device=device, args=args
            )
           
            pred_oof = _predict_with_base(enc, head, snp, va_idx, device)
            X_train_oof[va_rel, j] = pred_oof.astype(np.float32)

        
        enc_full, head_full, scaler_yj_full = _fit_base_model_for_trait(
            enc_init=enc_inits[j], head_init=head_states[j],
            X_idx=train_idx, y_raw=y_full[train_idx],
            snp=snp, device=device, args=args
        )
        X_val_pred[:,  j] = _predict_with_base(enc_full, head_full, snp, val_idx,  device).astype(np.float32)
        X_test_pred[:, j] = _predict_with_base(enc_full, head_full, snp, test_idx, device).astype(np.float32)

    return X_train_oof, X_val_pred, X_test_pred

def train_oof_stacking_grouped(snp, phe, target_trait, source_traits, args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed          = int(args.get("seed", 42))
    out_dir       = args.get("output", "./outputs/pretrain")
    test_size     = float(args.get("test_size", 0.3))
    val_size      = float(args.get("val_size", 0.3))
    _ensure_dir(out_dir)

    
    groups = build_groups(phe, snp, args)
    train_idx, val_idx, test_idx = group_split_indices(snp, groups, seed=seed,
                                                       test_size=test_size, val_size=val_size)

   
    X_train_oof, X_val_pred, X_test_pred = build_oof_features(
        snp=snp, phe=phe, source_traits=source_traits,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        args=args, device=device, out_dir=out_dir
    )

   
    scaler_x = StandardScaler().fit(X_train_oof)
    X_train_n = scaler_x.transform(X_train_oof)
    X_val_n   = scaler_x.transform(X_val_pred)
    X_test_n  = scaler_x.transform(X_test_pred)

    # train-only fit
    y_all = phe[target_trait].values.astype(np.float32)
    y_all = _nan_fix(y_all)
    y_train_raw = y_all[train_idx]
    y_val_raw   = y_all[val_idx]
    y_test_raw  = y_all[test_idx]
    scaler_y = MinMaxScaler().fit(y_train_raw.reshape(-1, 1))
    y_train_s = scaler_y.transform(y_train_raw.reshape(-1, 1)).ravel()
    y_val_s   = scaler_y.transform(y_val_raw.reshape(-1, 1)).ravel()

    
    ridge = RidgeCV(alphas=np.logspace(-4, 2, 16), fit_intercept=True, cv=5)
    ridge.fit(X_train_n, y_train_s)

   
    y_pred_val = scaler_y.inverse_transform(ridge.predict(X_val_n).reshape(-1, 1)).ravel()
    m_val = _metrics(y_val_raw, y_pred_val)

   
    y_pred_test = scaler_y.inverse_transform(ridge.predict(X_test_n).reshape(-1, 1)).ravel()
    m_test = _metrics(y_test_raw, y_pred_test)

   
    print("\n[OOF-Stacking] Best Fusion Metrics (on Val):")
    print(f"R²: {m_val['r2']:.4f}\nPearson: {m_val['pearson']:.4f}\nMAE: {m_val['mae']:.4f}\nRMSE: {m_val['rmse']:.4f}")
    print("\n[OOF-Stacking] Test Metrics (on Held-out Test):")
    print(f"R²: {m_test['r2']:.4f}\nPearson: {m_test['pearson']:.4f}\nMAE: {m_test['mae']:.4f}\nRMSE: {m_test['rmse']:.4f}")

    pd.DataFrame({
        "sample_index": test_idx,
        "y_true": y_test_raw.astype(np.float64),
        "y_pred": y_pred_test.astype(np.float64)
    }).to_csv(os.path.join(out_dir, "test_true_vs_pred_oof.csv"), index=False)

    pd.DataFrame([{
        "val_pearson": m_val["pearson"], "val_r2": m_val["r2"],
        "val_mae": m_val["mae"], "val_rmse": m_val["rmse"],
        "test_pearson": m_test["pearson"], "test_r2": m_test["r2"],
        "test_mae": m_test["mae"], "test_rmse": m_test["rmse"]
    }]).to_csv(os.path.join(out_dir, "fusion_val_best_oof.csv"), index=False)

    return {
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "val": m_val, "test": m_test
    }

# ================================================================

def train(snp, phe, target_trait, source_traits, args):
    """
      seed, output, input_len, conv_channels, kernel_size, stride, hidden_dim
      test_size=0.3, val_size=0.3, group_col(optional) or n_groups/n_pca
      （embedding mode）lr, fu_epoch, patience, fusion_dropout, weight_decay,
                       attn_entropy_lambda(1e-3~1e-2 suggestion), apply_calibration(True/False)
      （oof mode）oof_folds, oof_ft_epoch, oof_ft_unfreeze_encoder,
                 oof_ft_lr_head, oof_ft_lr_enc, oof_ft_weight_decay, oof_ft_batch_size
    """
    mode = str(args.get("fusion_mode", "embedding")).lower()
    if mode == "embedding":
        return train_embedding_fusion_grouped(snp, phe, target_trait, source_traits, args)
    elif mode == "oof":
        return train_oof_stacking_grouped(snp, phe, target_trait, source_traits, args)
    else:
        raise ValueError(f"Unknown fusion_mode: {mode}. Use 'embedding' or 'oof'.")
