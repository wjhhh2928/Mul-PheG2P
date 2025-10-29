import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# ====== Import internal modules ======
from utils.data_loader import SNPDataset, load_data
from utils.metrics import pearson_loss
from models import SNPEncoderCNN, MetaTraitFusion


# ============================================================
# ============================================================
def _safe_filename(name: str) -> str:
    
    return str(name).replace(" ", "_").replace("/", "_")


def _select_X_rows(X, rows):
    return X.iloc[rows, :] if hasattr(X, "iloc") else X[rows]


def _select_y_rows_col(Y, rows, col_idx=None, col_name=None):
    if hasattr(Y, "iloc"):
        if (col_name is not None) and (col_name in list(Y.columns)):
            y = Y.loc[rows, col_name]
        else:
            y = Y.iloc[rows, int(col_idx)]
    else:
        y = Y[rows, int(col_idx)]
    return np.asarray(y, dtype=np.float32).reshape(-1)


def visualize_attention(weights, source_traits, save_path):
    """Draw an attention weight bar chart"""
    weights = weights.mean(dim=0).cpu().numpy()  # 
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(weights)), weights)
    plt.xticks(range(len(weights)), source_traits, rotation=45, ha='right')
    plt.title("Meta Attention Weights Across Source Traits")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"save: {save_path}")


# ============================================================
# key：Fine-tune + Predict
# ============================================================
def fine_tune_and_predict(snp, phe, target_trait, source_traits, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Parameter Extraction =====
    seed          = args.get('seed', 42)
    lr            = args.get('lr', 1e-3)
    batch_size    = args.get('batch_size', 32)
    ft_epoch      = args.get('ft_epoch', 200)
    input_len     = args.get('input_len', 512)
    conv_channels = args.get('conv_channels', 32)
    kernel_size   = args.get('kernel_size', 7)
    stride        = args.get('stride', 2)
    hidden_dim    = args.get('hidden_dim', 128)

    # 
    out_dir = args.get('output', './results')     
    pre_dir = args.get('pretrain_dir', out_dir)    
    os.makedirs(out_dir, exist_ok=True)

    print("\n[Dirs]")
    print(f"  pretrain_dir = {pre_dir}")
    print(f"  output_dir   = {out_dir}")

   
    total_idx = np.arange(len(snp))

    
    pretrain_pool, remainder = train_test_split(
        total_idx, train_size=0.7, random_state=seed, shuffle=True
    )
   
    ft_train_idx, ft_val_idx = train_test_split(
        remainder, train_size=0.7, random_state=seed + 1, shuffle=True
    )
    train_idx, val_idx = ft_train_idx, ft_val_idx

    print(f"[Split] pretrain_pool={len(pretrain_pool)} ({len(pretrain_pool)/len(total_idx):.1%}), "
          f"finetune_train={len(train_idx)} ({len(train_idx)/len(total_idx):.1%}), "
          f"finetune_val={len(val_idx)} ({len(val_idx)/len(total_idx):.1%})")

    
    pretrained_models = []
    loaded_traits     = []
    for trait in source_traits:
        ckpt_path = os.path.join(pre_dir, f'pretrain_{_safe_filename(trait)}.pth')
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Skip: Not found {ckpt_path}")
            continue
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model = SNPEncoderCNN(input_len, conv_channels, kernel_size, stride, hidden_dim).to(device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        for p in model.parameters():
            p.requires_grad = False  # Freeze
        model.eval()
        pretrained_models.append(model)
        loaded_traits.append(trait)
        print(f" Loaded pretrain: {ckpt_path}")

    if len(pretrained_models) == 0:
        raise RuntimeError("No pre-trained models were found, please check pretrain_dir and file naming（pretrain_{Trait}.pth）。")

    num_traits = len(pretrained_models)
    print(f" {num_traits} pre-trained models have been loaded for fusion：{', '.join(_safe_filename(t) for t in loaded_traits)}")

    # ==========
    fusion = MetaTraitFusion(num_traits=num_traits, hidden_dim=64).to(device)
    head = nn.Sequential(
        nn.Linear(1, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)

    optimizer = torch.optim.Adam(list(fusion.parameters()) + list(head.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-5
    )
    mse = nn.MSELoss()

    # ===== Building a dataset (NaN cleaning and fallback handling) =====
    col_idx = target_trait if isinstance(target_trait, int) else None
    col_name = target_trait if isinstance(target_trait, str) else None

    def _to_numpy(x):
        return x.values if hasattr(x, "values") else x

    X_train = _select_X_rows(snp, train_idx)
    y_train = _select_y_rows_col(phe, train_idx, col_idx=col_idx, col_name=col_name)
    X_val   = _select_X_rows(snp, val_idx)
    y_val   = _select_y_rows_col(phe, val_idx, col_idx=col_idx, col_name=col_name)

    
    X_train = _to_numpy(X_train).astype(np.float32)
    X_val   = _to_numpy(X_val).astype(np.float32)

    
    mask_tr = np.isfinite(y_train)
    mask_va = np.isfinite(y_val)
    if mask_tr.sum() < len(mask_tr):
        print(f"⚠️ The training set removed {len(mask_tr) - mask_tr.sum()} samples with y=NaN")
    if mask_va.sum() < len(mask_va):
        print(f"⚠️ The validation set removed {len(mask_va) - mask_va.sum()} samples with y=NaN")
    X_train, y_train = X_train[mask_tr], y_train[mask_tr]
    X_val,   y_val   = X_val[mask_va],   y_val[mask_va]

    # NaN/Inf in X are defaulted to 0 (can also be changed to mean/median)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val   = np.nan_to_num(X_val,   nan=0.0, posinf=0.0, neginf=0.0)

    train_loader = DataLoader(SNPDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(SNPDataset(X_val,   y_val),   batch_size=batch_size)

    # ============================================================
    # Training Phase (Fine-tuning)
    # ============================================================
    best_val = float('inf')
    best_model = None
    best_ckpt_path = os.path.join(out_dir, f"fine_tuned_{_safe_filename(str(target_trait))}.pth")

    print("\n===== （Attention + MLP） =====")
    for epoch in range(ft_epoch):
        fusion.train()
        head.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            with torch.no_grad():
                feats = [m(batch_x) for m in pretrained_models]  # [B, hidden_dim] × num_traits
                feats = torch.stack(feats, dim=1)                # [B, num_traits, hidden_dim]
                feats = torch.nan_to_num(feats, nan=0.0)
                trait_preds = feats.mean(dim=2)                   # [B, num_traits]
                trait_preds = torch.nan_to_num(trait_preds, nan=0.0)

            pred_fused, weights = fusion(trait_preds)             # [B], [B, num_traits]
            pred_fused = torch.nan_to_num(pred_fused, nan=0.0)
            output = head(pred_fused.unsqueeze(-1)).squeeze(1)    # [B]
            output = torch.nan_to_num(output, nan=0.0)

            loss = mse(output, batch_y) + 0.2 * pearson_loss(output, batch_y)
            if torch.isnan(loss):
                print(" Detected training loss=NaN, skipping this batch")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(fusion.parameters()) + list(head.parameters()), max_norm=5.0)
            optimizer.step()

        # ===== validation =====
        fusion.eval()
        head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                feats = [m(batch_x) for m in pretrained_models]
                feats = torch.stack(feats, dim=1)
                feats = torch.nan_to_num(feats, nan=0.0)
                trait_preds = feats.mean(dim=2)
                trait_preds = torch.nan_to_num(trait_preds, nan=0.0)
                pred_fused, _ = fusion(trait_preds)
                pred_fused = torch.nan_to_num(pred_fused, nan=0.0)
                output = head(pred_fused.unsqueeze(-1)).squeeze(1)
                output = torch.nan_to_num(output, nan=0.0)
                loss = mse(output, batch_y)
                if torch.isnan(loss):
                    print(" Detected validation loss=NaN, skipping this batch")
                    continue
                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))
        print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        if np.isnan(val_loss):
            print(" Validation loss is NaN, early stopping, rolling back to the current best.")
            break

        
        if val_loss < best_val:
            best_val = val_loss
            best_model = {
                'fusion': fusion.state_dict(),
                'head': head.state_dict(),
                'used_traits': loaded_traits
            }
            torch.save(best_model, best_ckpt_path)
            print(f" The current best fine-tuned model has been saved to: {best_ckpt_path}")

    
    if best_model is None:
        best_model = {
            'fusion': fusion.state_dict(),
            'head': head.state_dict(),
            'used_traits': loaded_traits
        }

    # ============================================================

    # ============================================================
    print("\n===== Evaluation (using 0.09 of the finetune validation set as the test) =====")
    fusion.load_state_dict(best_model['fusion'])
    head.load_state_dict(best_model['head'])
    fusion.eval()
    head.eval()

    preds, trues, all_weights = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            feats = [m(batch_x) for m in pretrained_models]
            feats = torch.stack(feats, dim=1)
            feats = torch.nan_to_num(feats, nan=0.0)
            trait_preds = feats.mean(dim=2)
            trait_preds = torch.nan_to_num(trait_preds, nan=0.0)
            pred_fused, weights = fusion(trait_preds)
            pred_fused = torch.nan_to_num(pred_fused, nan=0.0)
            output = head(pred_fused.unsqueeze(-1)).squeeze(1)
            output = torch.nan_to_num(output, nan=0.0)

            preds.extend(output.cpu().numpy())
            trues.extend(batch_y.cpu().numpy())
            all_weights.append(weights)

    preds = np.array(preds, dtype=np.float64)
    trues = np.array(trues, dtype=np.float64)

   
    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    trues = np.nan_to_num(trues, nan=0.0, posinf=0.0, neginf=0.0)
    if np.isnan(preds).any() or np.isnan(trues).any():
        raise ValueError("NaN still exists before evaluation. Please check data cleaning or model output.")

    r2 = r2_score(trues, preds)
    pearson = float(pearsonr(trues, preds)[0])
    mae = mean_absolute_error(trues, preds)
    rmse = float(np.sqrt(mean_squared_error(trues, preds)))

    print("\n=== Validation set (used as the final test) results ===")
    print(f"R² = {r2:.4f} | Pearson = {pearson:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}")

    # ===== Save predictions and metrics =====
    save_pred = os.path.join(out_dir, f"predict_{_safe_filename(str(target_trait))}.csv")
    np.savetxt(save_pred, np.column_stack([trues, preds]),
               delimiter=",", header="y_true,y_pred", comments="")
    print(f"The prediction results have been saved to: {save_pred}")

    metrics_path = os.path.join(out_dir, f"metrics_{_safe_filename(str(target_trait))}.csv")
    with open(metrics_path, "w") as f:
        f.write("R2,Pearson,MAE,RMSE\n")
        f.write(f"{r2:.6f},{pearson:.6f},{mae:.6f},{rmse:.6f}\n")
    print(f"The indicator has been saved to: {metrics_path}")

    # ===== Attention Visualization =====
    all_weights = torch.cat(all_weights, dim=0)
    save_attn = os.path.join(out_dir, f"attention_{_safe_filename(str(target_trait))}.png")
    visualize_attention(all_weights, loaded_traits, save_attn)


# ============================================================

# ============================================================
if __name__ == "__main__":
    print("=== Fine-tune & Predict (Attention Fusion) ===")

    # ====== Data path ======
    geno_path   = "/home/zcy/zyq/GE_T/new_data/bjut/gene_new.npy"
    pheno_path  = "/home/zcy/zyq/GE_T/new_data/bjut/phe_new_wusample.csv"

    
    pretrain_dir = "/mnt/mydisk/zyq/GE_T/trans_learning2/out_puts"  #
    output_dir   = "/mnt/mydisk/zyq/GE_T/trans_learning2/results"   # 

    # ====== Load Data ======
    snp, phe = load_data(geno_path, pheno_path)

    # ====== Target traits and pre-trained source traits ======
    target_trait = "Yield"
    source_traits = [
        "Day to tasseling", "Day to siking", "Day to maturity", "Plant height", "Ear height",
        "100 kernel weight", "Ear length", "Kernel number per row", "Ear diameter", "Row number per ear"
    ]

    # ====== Parameter Settings ======
    args = {
        'input_len'    : 512,
        'hidden_dim'   : 128,
        'conv_channels': 32,
        'kernel_size'  : 7,
        'stride'       : 2,
        'ft_epoch'     : 200,
        'batch_size'   : 32,
        'lr'           : 1e-3,
        'seed'         : 42,
        'output'       : output_dir,   
        'pretrain_dir' : pretrain_dir, 
    }

    # ====== Run fine-tuning   Prediction ======
    fine_tune_and_predict(snp, phe, target_trait, source_traits, args)
