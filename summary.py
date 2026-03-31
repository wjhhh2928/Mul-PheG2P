import pandas as pd
import numpy as np
import os

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    df.columns = df.columns.str.strip()
    rename_map = {}
    cols_lower = {c.lower(): c for c in df.columns}

   
    if 'task' in cols_lower:          rename_map[cols_lower['task']] = 'Task'
    if 'trait' in cols_lower:         rename_map[cols_lower['trait']] = 'Task'
    if 'target_trait' in cols_lower:  rename_map[cols_lower['target_trait']] = 'Task'
    if 'model' in cols_lower:         rename_map[cols_lower['model']] = 'Model'
    if 'model_name' in cols_lower:    rename_map[cols_lower['model_name']] = 'Model'

    
    for lo, std in [('r2', 'R2'), ('pearson', 'Pearson'), ('mae', 'MAE'), ('rmse', 'RMSE')]:
        if lo in cols_lower:
            rename_map[cols_lower[lo]] = std

    return df.rename(columns=rename_map)

def summary(output_dir, cv_folds=5):
    
    dfs = []
    
   
    target_files = {
        '1. Baseline (Direct)': 'direct_res.csv',
        '2. Pretrain (Zero-Shot)': 'pretrained_target_res.csv', 
        '3. Fine-Tuned': 'fine_tuned_res.csv', 
        '4. Fusion (Embedding)': 'fusion_val_best.csv',
        '5. Fusion (OOF)': 'fusion_val_best_oof.csv'
    }

    print(f"\n[Summary] Scanning {cv_folds} folds in {output_dir}...")

    
    for fold in range(cv_folds):
        
        fold_dir = os.path.join(output_dir, f'fold_{fold}')
        if not os.path.exists(fold_dir):
            fold_dir = os.path.join(output_dir, f'fold_{fold+1}')
            
        if not os.path.exists(fold_dir):
            print(f"[Warn] Directory {fold_dir} not found. Skipping...")
            continue
        
        for model_name, file_name in target_files.items():
            file_path = os.path.join(fold_dir, file_name)
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    df = _normalize_columns(df)
                    
                    if 'Fusion' in model_name:
                        
                        rename_fusion = {
                            'test_pearson_cal': 'Pearson', 'test_pearson': 'Pearson',
                            'test_r2_cal': 'R2', 'test_r2': 'R2',
                            'test_mae_cal': 'MAE', 'test_mae': 'MAE',
                            'test_rmse_cal': 'RMSE', 'test_rmse': 'RMSE'
                        }
                        df = df.rename(columns=rename_fusion)
                        
                        
                        df = df.loc[:, ~df.columns.duplicated()]
                    
                    metrics = ['R2', 'Pearson', 'MAE', 'RMSE']
                    available_metrics = [m for m in metrics if m in df.columns]
                    
                    if available_metrics:
                        if len(df) > 1 and 'Pearson' in available_metrics:
                            best_row = df.loc[df['Pearson'].idxmax()].to_frame().T
                        else:
                            best_row = df.head(1)
                            
                        df_sub = best_row[available_metrics].copy()
                        df_sub['Model'] = model_name
                        df_sub['Fold'] = fold
                        dfs.append(df_sub)
                except Exception as e:
                    print(f"[Error] Failed to parse {file_path}: {e}")

    if not dfs:
        print("[Error] No valid result files found across all folds.")
        return

    
    res_all = pd.concat(dfs, ignore_index=True)
    
   
    metrics = ['R2', 'Pearson', 'MAE', 'RMSE']
    available_metrics = [m for m in metrics if m in res_all.columns]
    
    grouped = res_all.groupby('Model')[available_metrics].agg(['mean', 'std'])
    
   
    summary_table = pd.DataFrame()
    for metric in available_metrics:
        mean_col = grouped[metric]['mean'].fillna(0)
        std_col = grouped[metric]['std'].fillna(0)
        summary_table[metric] = mean_col.apply(lambda x: f"{x:.4f}") + " ± " + std_col.apply(lambda x: f"{x:.4f}")
    
    summary_table.reset_index(inplace=True)
    summary_table = summary_table.sort_values(by='Model').reset_index(drop=True)

   
    out_csv = os.path.join(output_dir, 'global_5fold_summary.csv')
    summary_table.to_csv(out_csv, index=False)
    
    raw_csv = os.path.join(output_dir, 'global_5fold_raw_metrics.csv')
    res_all.to_csv(raw_csv, index=False)

   
    print("\n" + "="*70)
    print("=== Final Global 5-Fold Cross Validation Summary (Mean ± SD) ===")
    print("="*70)
    print(summary_table.to_string(index=False, justify='center'))
    print("="*70)
    print(f"\n[Save] Publication-ready summary table saved to: {out_csv}")
    print(f"[Save] Raw metrics per fold (for boxplots) saved to: {raw_csv}")

if __name__ == "__main__":
    
    summary("./results", cv_folds=5)
