# Mul-PheG2P

An explainable fusion method for multi-phenotype genotype-to-phenotype prediction
Environment

Python ≥ 3.8, PyTorch ≥ 1.10 (CUDA strongly recommended)

Common deps: numpy, pandas, scikit-learn, scipy, matplotlib, etc.

Install example:

pip install -r requirements.txt

Data Preparation

Genotype (geno): .npy or .csv/.tsv, shape [N, L], numeric (float/int; internally cast to float32).

Phenotype (pheno): *.csv, shape [N, T], one trait per column; column names are used to index source/target traits.

Data loading interfaces are implemented in utils/data_loader.py (example function names):

load_data(geno_path, pheno_path) -> (snp, phe)

SNPDataset(X, y) adapts [B, L] tensors for the 1D-CNN encoder.

SNP input shape: the model accepts [B, L] or [B, 1, L]. It will auto-unsqueeze / validate the channel dimension to avoid shape errors.

Quick Start
1) One-click pipeline (recommended)

Run Direct → Pretrain → Fine-tune → Fusion → Summary in sequence:

python train.py

Configure data paths, source/target traits, input length, training hyper-params, output directory, etc. in the Config section of train.py.

2) Pretraining only + Zero-shot evaluation
python train-no-fine-tuning.py


Good for producing multi-trait pretrained weights and zero-shot evaluation on a target trait.

3) Unified single-shot prediction
model ∈ {direct, fine_tuned, fusion}
python predict.py \
  --geno ./data/gene.npy \
  --model fusion \
  --trait "Plant height" \
  --output ./pred_out

4) Use pretrained weights to predict the target trait
python fine_tune_predict.py

Configuration & Hyperparameters

Centralized in the Config of train.py:

Data: geno_path, pheno_path, source_traits, target_trait

Model: input_len, hidden_dim, conv_channels, kernel_size, stride

Training: cv_folds, _epochs, batch_size, learning_rate

Misc: random_seed, output_dir

You may also override via args dictionaries in sub-modules. In fusion.py, you can additionally set fusion_dropout, attn_entropy_lambda, weight_decay, test_size, val_size, etc.

Outputs & Visualization

Typical outputs under ./results/...:

Direct: direct_{trait}.pth, direct_res.csv

Pretrain: pretrain_{Trait}.pth (with config/train_idx/heldout_idx), pretrained_target_res.csv, pretrained_target_true_vs_pred.csv

Fine-tune: fine_tuned_{trait}.pth, prediction CSVs, and attention-weight plots

Fusion: fusion_model_embedding.pth, fusion_attention_weights.png/.csv, fusion_val_best.csv, test_true_vs_pred.csv

Summary: results_summary.csv (Task × Model aggregated mean/variance)

Models & Interfaces

SNPEncoderCNN: 1D convolutions + adaptive pooling + FC → hidden_dim; robust to input shapes (auto-adds channel dim).

MetaTraitFusion: takes [B, num_traits] trait-level predictions/embeddings, learns attention weights for weighted summation, and regresses to the target trait.

FAQ

Shape mismatch: ensure SNP inputs are [B, L] or [B, 1, L]; if [B, 1, 1, L], it will be auto-squeezed in forward.

Out of memory: reduce batch_size; in Fusion, consider lowering hidden_dim and/or number of source traits; or test the zero-shot pipeline on CPU first.

NaNs in data: this repo defaults to 0-fill for features and median/mean fill for targets—verify the missingness mechanism and align samples upstream if possible.

Unstable results: seeds are fixed and cuDNN nondeterminism is disabled; further stabilize by increasing data/regularization or reducing learning rate.

License & Acknowledgments

If this repo helps your research/production, please consider citing and giving a ⭐ Star.
