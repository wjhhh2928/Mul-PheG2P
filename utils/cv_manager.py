import numpy as np
from sklearn.model_selection import KFold, train_test_split
import os
import pandas as pd

class GlobalCVManager:
    
    def __init__(self, n_samples: int, n_splits: int = 5, seed: int = 42, val_ratio: float = 0.2):
        self.n_samples = n_samples
        self.n_splits = n_splits
        self.seed = seed
        self.val_ratio = val_ratio
        self.indices = np.arange(n_samples)
        
        
        self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        self.folds_data = self._generate_folds()

        
        self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        self.folds_data = self._generate_folds()

    def _generate_folds(self):
       
        folds = []
       
        for fold, (train_val_pool_idx, test_idx) in enumerate(self.kf.split(self.indices)):
            
            train_idx, val_idx = train_test_split(
                train_val_pool_idx, 
                test_size=self.val_ratio, 
                random_state=self.seed + fold
            )
            
            folds.append({
                'fold': fold + 1,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx
            })
        return folds

    def get_fold(self, fold_idx: int):
        
        return self.folds_data[fold_idx]

    def save_indices(self, output_dir: str):

        os.makedirs(output_dir, exist_ok=True)
        for fold_data in self.folds_data:
            fold = fold_data['fold']
            np.save(os.path.join(output_dir, f'fold_{fold}_train_idx.npy'), fold_data['train_idx'])
            np.save(os.path.join(output_dir, f'fold_{fold}_val_idx.npy'), fold_data['val_idx'])
            np.save(os.path.join(output_dir, f'fold_{fold}_test_idx.npy'), fold_data['test_idx'])
        print(f"Global CV indices saved to {output_dir}")
