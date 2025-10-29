import os
from modules import direct, fusion, pretrain, fine_tune
from utils.data_loader import load_data
from summary import summary
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Config:
    # ===== Data parameters =====
    geno_path = "gene.npy"  
    pheno_path = "phe.csv"  
    source_traits = ["Day to siking"]    # phenotype
    target_trait = "Day to siking"    # phenotype
     
    # ===== Model parameters =====
    input_len = 512  
    hidden_dim = 128  
    conv_channels = 32  
    kernel_size = 7  
    stride = 2  
    # ===== other parameters =====
    cv_folds = 5  
    
    # ===== training parameters =====
    pretrain_epochs = 300  
    finetune_epochs = 300  
    fusion_epochs = 200 
    direct_epochs = 300  
    batch_size = 32  
    learning_rate = 0.001  
    
    # ===== other parameters =====
    random_seed = 42  
    output_dir = "./results/NKY22_DTS_PHE_experiment1"  # 

def main():
    cfg = Config()
    
   
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    
    snp, phe = load_data(cfg.geno_path, cfg.pheno_path)
    
    
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
    
      
    print("\n=== Global Parameter Verification ===")
    print("seed:", args['seed'])
    print("batch_size:", args['batch_size'])

    print('=== 1. Direct Prediction ===')
    direct.train(snp, phe, cfg.target_trait, args)
    
    print('=== 2. Pretraining ===')
    pretrain.train(snp, phe, cfg.source_traits, args)
    
    print('=== 3. Fine-tuning ===')
    fine_tune.train(snp, phe, cfg.target_trait, cfg.source_traits, args)
    
    print('=== 4. Fusion ===')
    fusion.train(snp, phe, cfg.target_trait, cfg.source_traits, args)
    
    print('=== 5. Summary ===')
    summary(cfg.output_dir)

if __name__ == "__main__":
    main()
