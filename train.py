import os
from modules import direct, fusion, pretrain, fine_tune
from utils.data_loader import load_data
from utils.cv_manager import GlobalCVManager  # 引入我们刚刚写的管理器
from summary import summary
import sys

class Config:
    geno_path = "gene.npy"
    pheno_path = "phenptype.csv"
    source_traits = ["phenotype 1"]       
    target_trait = "phenotype 1"
  
    input_len = 33709
    hidden_dim = 128
    conv_channels = 32
    kernel_size = 7
    stride = 2
    cv_folds = 5

    pretrain_epochs = 300 
    finetune_epochs = 300  
    fusion_epochs = 200  
    direct_epochs = 200  
    batch_size = 32
    learning_rate = 0.001

    random_seed = 42
    output_dir = "./results"    


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    
   
    snp, phe = load_data(cfg.geno_path, cfg.pheno_path)
    n_samples = len(snp)
    
    
    cv_manager = GlobalCVManager(n_samples=n_samples, n_splits=cfg.cv_folds, seed=cfg.random_seed)
    cv_manager.save_indices(cfg.output_dir) 
    
    
    args = {
        'input_len': cfg.input_len,
        'hidden_dim': cfg.hidden_dim,
        'conv_channels': cfg.conv_channels,
        'kernel_size': cfg.kernel_size,
        'stride': cfg.stride,
        'pretrain_epoch': cfg.pretrain_epochs,
        'ft_epoch': cfg.finetune_epochs,
        'fu_epoch': cfg.fusion_epochs,
        'direct_epoch': cfg.direct_epochs,  
        'cv_folds': cfg.cv_folds,  
        'batch_size': cfg.batch_size,
        'lr': cfg.learning_rate,
        'seed': cfg.random_seed,
        'output': cfg.output_dir
    }

    print("\n=== Starting Global 5-Fold Cross Validation Pipeline ===")
    
    
    for fold_idx in range(cfg.cv_folds):
        fold_data = cv_manager.get_fold(fold_idx)
        fold_num = fold_data['fold']
        train_idx = fold_data['train_idx']
        val_idx = fold_data['val_idx']
        test_idx = fold_data['test_idx']
        
        print(f"\n" + "="*40)
        print(f"Executing Fold {fold_num}/{cfg.cv_folds}")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        print("="*40)
        
        
        fold_output_dir = os.path.join(cfg.output_dir, f"fold_{fold_num}")
        os.makedirs(fold_output_dir, exist_ok=True)
        args['output'] = fold_output_dir  
        
        
        
        print(f'=== [Fold {fold_num}] 1. Direct Prediction (Baseline) ===')
       
        direct.train(snp, phe, cfg.target_trait, train_idx, val_idx, test_idx, args)
        
        print(f'=== [Fold {fold_num}] 2. Pretraining ===')
        pretrain.train(snp, phe, cfg.source_traits, train_idx, val_idx, args)
        
        print(f'=== [Fold {fold_num}] 3. Fine-tuning ===')
        fine_tune.train(snp, phe, cfg.target_trait, cfg.source_traits, train_idx, val_idx, test_idx, args)
        
        print(f'=== [Fold {fold_num}] 4. Fusion ===')
       
        fusion.train(snp, phe, cfg.target_trait, cfg.source_traits, train_idx, val_idx, test_idx, args)
        
   
    print('\n=== Pipeline Completed. Consolidating Results ===')
    summary(cfg.output_dir, cv_folds=cfg.cv_folds)

if __name__ == "__main__":
    main()
