import torch  
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class SNPDataset(Dataset):
    
    def __init__(self, snp, trait):
        self.snp = torch.FloatTensor(snp)  
        self.trait = torch.FloatTensor(trait)
    
    def __len__(self):
        return len(self.snp)
    
    def __getitem__(self, idx):
        return self.snp[idx], self.trait[idx]

def load_data(geno_path, pheno_path):
    
    
    if geno_path.endswith('.npy'):
        snp = np.load(geno_path)
    else: 
        snp = np.loadtxt(geno_path, dtype=np.float32)
    
    
    phe = pd.read_csv(pheno_path, header=0)  
    
 
    assert len(snp) == len(phe), "The number of samples does not match！"
    
    return snp, phe

