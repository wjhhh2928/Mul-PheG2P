import numpy as np
from sklearn.model_selection import KFold, train_test_split
import os
import pandas as pd

class GlobalCVManager:
    """
    Nature-level 全局交叉验证管理器
    确保所有多阶段模型和基线模型使用完全相同的数据划分，杜绝数据泄露。
    """
    def __init__(self, n_samples: int, n_splits: int = 5, seed: int = 42, val_ratio: float = 0.2):
        self.n_samples = n_samples
        self.n_splits = n_splits
        self.seed = seed
        self.val_ratio = val_ratio
        self.indices = np.arange(n_samples)
        
        # 核心：使用同一颗随机种子初始化 KFold
        self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        self.folds_data = self._generate_folds()
    # def __init__(self, n_samples: int, n_splits: int = 11, seed: int = 42, val_ratio: float = 0.231):
    #     self.n_samples = n_samples
    #     self.n_splits = n_splits
    #     self.seed = seed
    #     self.val_ratio = val_ratio
    #     self.indices = np.arange(n_samples)
        
        # 核心：使用同一颗随机种子初始化 KFold
        self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        self.folds_data = self._generate_folds()

    def _generate_folds(self):
        """生成并固化每一折的 train, val, test 索引"""
        folds = []
        # KFold 将总体分为: 剩余部分 (Train+Val Pool) 和 测试集 (Test)
        for fold, (train_val_pool_idx, test_idx) in enumerate(self.kf.split(self.indices)):
            
            # 在 Train+Val Pool 中，再次严格划分出 Train 和 Val
            # 这里的 random_state=self.seed + fold 确保每一折内部划分是确定且可复现的
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
        """获取特定折的数据索引"""
        return self.folds_data[fold_idx]

    def save_indices(self, output_dir: str):
        """(可选) 将每折的索引保存到磁盘，以备极端严谨的审稿人要求提供原始数据分割"""
        os.makedirs(output_dir, exist_ok=True)
        for fold_data in self.folds_data:
            fold = fold_data['fold']
            np.save(os.path.join(output_dir, f'fold_{fold}_train_idx.npy'), fold_data['train_idx'])
            np.save(os.path.join(output_dir, f'fold_{fold}_val_idx.npy'), fold_data['val_idx'])
            np.save(os.path.join(output_dir, f'fold_{fold}_test_idx.npy'), fold_data['test_idx'])
        print(f"Global CV indices saved to {output_dir}")