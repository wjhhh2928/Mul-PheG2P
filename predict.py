import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from models import SNPEncoderCNN, MetaTraitFusion
from utils.data_loader import load_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geno", required=True, help="Genotype file for prediction")
    parser.add_argument("--model", required=True, choices=['direct', 'fine_tuned', 'fusion'], 
                      help="Type of model to use for prediction")
    parser.add_argument("--trait", required=True, help="Target trait name")
    parser.add_argument("--output", default="predictions", help="Output directory")
    return parser.parse_args()

def predict():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
  
    snp = load_data(args.geno)
    

    if args.model == 'direct':
        model = SNPEncoderCNN(input_len=snp.shape[1]).to(device)
        head = nn.Linear(128, 1).to(device)
        
        checkpoint = torch.load(os.path.join("results", f'direct_{args.trait}.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        head.load_state_dict(checkpoint['head_state_dict'])
        
        model.eval()
        head.eval()
        
        with torch.no_grad():
            snp_tensor = torch.tensor(snp, dtype=torch.float32).to(device)
            feat = model(snp_tensor)
            pred = head(feat).squeeze(1).cpu().numpy()
    
    elif args.model == 'fine_tuned':
        checkpoint = torch.load(os.path.join("results", f'fine_tuned_{args.trait}.pth'))
        
        models = []
        for state_dict in checkpoint['model_state_dicts']:
            model = SNPEncoderCNN(input_len=snp.shape[1]).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
        
        head = nn.Linear(128, 1).to(device)
        head.load_state_dict(checkpoint['head_state_dict'])
        head.eval()
        
        with torch.no_grad():
            snp_tensor = torch.tensor(snp, dtype=torch.float32).to(device)
            features = []
            for model in models:
                features.append(model(snp_tensor))
            feat = torch.mean(torch.stack(features), dim=0)
            pred = head(feat).squeeze(1).cpu().numpy()
    
    elif args.model == 'fusion':
        
        source_traits = ['PH', 'PnN', 'LA']  
        trait_models = []
        for trait in source_traits:
            checkpoint = torch.load(os.path.join("results", f'pretrain_{trait}.pth'))
            
            model = SNPEncoderCNN(input_len=snp.shape[1]).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            head = nn.Linear(128, 1).to(device)
            head.load_state_dict(checkpoint['head_state_dict'])
            head.eval()
            
            trait_models.append((model, head))
        
        
        fusion_model = MetaTraitFusion(len(source_traits)).to(device)
        fusion_model.load_state_dict(torch.load(os.path.join("results", 'fusion_model.pth')))
        fusion_model.eval()
        
        with torch.no_grad():
            snp_tensor = torch.tensor(snp, dtype=torch.float32).to(device)
            preds = []
            for model, head in trait_models:
                feat = model(snp_tensor)
                pred = head(feat)
                preds.append(pred)
            X = torch.cat(preds, dim=1)
            pred, _ = fusion_model(X)
            pred = pred.cpu().numpy()
    
   
    os.makedirs(args.output, exist_ok=True)
    np.savetxt(os.path.join(args.output, f'prediction_{args.model}_{args.trait}.csv'), pred, delimiter=',')
    print(f"Predictions saved to {os.path.join(args.output, f'prediction_{args.model}_{args.trait}.csv')}")

if __name__ == "__main__":
    predict()