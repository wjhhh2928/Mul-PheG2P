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

    
    for lo, std in [('r2','R2'), ('pearson','Pearson'), ('mae','MAE'), ('rmse','RMSE')]:
        if lo in cols_lower:
            rename_map[cols_lower[lo]] = std

    return df.rename(columns=rename_map)

def _ensure_task_model(df: pd.DataFrame, task_name: str) -> pd.DataFrame:
  
    if 'Task' not in df.columns:
        df['Task'] = task_name
    if 'Model' not in df.columns:
        
        df['Model'] = 'UnknownModel'
    return df

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(x) for x in tup if x != '']).strip('_') for tup in df.columns.values]
    return df

def summary(output_dir):
    files = {
        'Direct':    'direct_res.csv',
        'Pretrain':  'pretraining_res.csv',
        'FineTune':  'fine_tuning_res.csv',
        'Fusion':    'fusion_res.csv',
    }

    dfs = []
    for task_name, file in files.items():
        path = os.path.join(output_dir, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = _normalize_columns(df)
            df = _ensure_task_model(df, task_name)
            dfs.append(df)

    if not dfs:
        print("No result files found for summary")
        return

    res_all = pd.concat(dfs, ignore_index=True)

    
    wanted = ['R2', 'Pearson', 'MAE', 'RMSE']
    metrics = [c for c in wanted if c in res_all.columns]
    if not metrics:
        
        print("Columns found:", list(res_all.columns))
        raise ValueError("No metric columns (R2/Pearson/MAE/RMSE) found to summarize.")

    
    grouped = (res_all
               .groupby(['Task', 'Model'])[metrics]
               .agg(['mean', 'std'])
               .reset_index())

    grouped = _flatten_columns(grouped)

    # save
    out_csv = os.path.join(output_dir, 'results_summary.csv')
    grouped.to_csv(out_csv, index=False)

    print("\n=== Results Summary ===")
    print(grouped)
    print(f"\nSummary saved to {out_csv}")
