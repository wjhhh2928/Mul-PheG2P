import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_data
from modules.pretrain import train as pretrain_train, predict_with_pretrained_target

class Config:
    geno_path = "/mnt/mydisk/zyq/GE_T/new_data/bjut/gene_new.npy"
    pheno_path = "/mnt/mydisk/zyq/GE_T/new_data/bjut/phe_new_wusample.csv"
    source_traits = ["Day to tasseling"]       ##["Day to tasseling", "Day to siking", "Day to maturity", "Plant height", "Ear height", "100 kernel weight", "Ear length", "Yield", "Kernel number per row", "Ear diameter", "Row number per ear"]  TGW,TW,GrL,GrW,GrH,GrP
    target_trait = "Day to tasseling"
  
    input_len = 33709
    hidden_dim = 128
    conv_channels = 32
    kernel_size = 7
    stride = 2
    cv_folds = 5

    pretrain_epochs = 300
    batch_size = 32
    learning_rate = 0.001

    random_seed = 42
    output_dir = "./results2/nky_dtt_PHE_experiment1"    ##/home/zy/zyq/GE_T/trans_learning2/results2

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
        'cv_folds': cfg.cv_folds,
        'batch_size': cfg.batch_size,
        'lr': cfg.learning_rate,
        'seed': cfg.random_seed,
        'output': cfg.output_dir,
        'device': 'cpu', 
    }

    print("\n=== 仅执行多性状预训练 ===")
    pretrain_train(snp, phe, cfg.source_traits, args)

    print("\n=== 直接使用目标性状的预训练权重进行预测（零微调） ===")
    predict_with_pretrained_target(snp, phe, cfg.target_trait, args)

    print("\n=== 完成 ===")

if __name__ == "__main__":
    main()
