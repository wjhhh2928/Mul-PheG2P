# 🧬 Mul-PheG2P: Decoupled Learning and Prediction-Space Fusion for Genomic Prediction

[](https://www.python.org/downloads/)
[](https://pytorch.org/)
[](https://www.google.com/search?q=LICENSE)
[](https://github.com/wjhhh2928/Mul-PheG2P) Official PyTorch implementation for **Mul-PheG2P**, a novel paradigm for multi-phenotype genomic prediction (GP) that achieves robust performance and interpretability.

Mul-PheG2P addresses the challenge of balancing synergistic gain with task conflict in multi-trait prediction. It uses a two-stage design: (1) training trait-specific encoders to preserve unique genetic patterns, and (2) aggregating cross-trait signals via an interpretable prediction-layer fusion. This decouples trait-specific learning from cross-trait aggregation.

### 🌟 Key Features

  * **High Performance:** Outperforms or matches GBLUP and recent deep models, especially in small-sample, high-dimensional settings.
  * **Avoids Negative Transfer:** The decoupled design successfully avoids the task conflict and negative transfer common in end-to-end coupled models.
  * **Multi-Scale Interpretability:** Provides macro-level inter-phenotype contributions via attention maps and micro-level SNP/LD block identification via SHAP.
  * **Zero-Finetune Transfer:** Pretrained representations are highly transferable, enabling robust direct inference without fine-tuning, ideal for low-cost screening.

<p align="center">
    <img src="Imgs/Fig1.png" alt="Model" width="85%">
</p>

## 🛠️ Environment

Requires Python ≥ 3.8 and PyTorch ≥ 1.10. CUDA is strongly recommended.

```bash
# Clone the repository
git clone https://github.com/wjhhh2928/Mul-PheG2P.git
cd Mul-PheG2P

# Install dependencies
pip install -r requirements.txt
```

**Main dependencies:** `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`.

## 💾 Data Preparation

  * **Genotype (geno):** `.npy` or `.csv`/`.tsv` file.
      * Shape: `[N, L]` (N = samples, L = SNP loci).
      * Internally cast to `float32`.
  * **Phenotype (pheno):** `.csv` file.
      * Shape: `[N, T]` (N = samples, T = traits).
      * **Important:** Column names are used to index source and target traits.

Data loading logic is implemented in `utils/data_loader.py`.

## 🚀 Quick Start

All configurations (data paths, traits, hyperparameters) are centralized in the `Config` section of `train.py`.

### 1\. Full Pipeline (Recommended)

Run the complete "Direct $\rightarrow$ Pretrain $\rightarrow$ Fine-tune $\rightarrow$ Fusion" pipeline and generate a final results summary.

```bash
# 1. Configure paths and traits in train.py
# 2. Run the full pipeline
python train.py
```

### 2\. Pre-training + Zero-Shot Evaluation

Train trait-specific encoders and evaluate their "zero-finetune" performance on a target trait.

```bash
# 1. Configure paths and traits in train-no-fine-tuning.py
# 2. Run the pre-training and zero-shot evaluation
python train-no-fine-tuning.py
```

### 3\. Inference with a Trained Model

Use a trained model (e.g., `fusion`, `fine_tuned`) for prediction on new genotype data.

```bash
python predict.py \
    --geno ./data/gene.npy \
    --model fusion \
    --trait "YourTargetTrait" \
    --output ./pred_out
```

### 4\. Use Pretrained Weights to Predict a Target Trait

Load pretrained models and fine-tune them on a specific target trait.

```bash
# 1. Configure paths and traits in fine_tune_predict.py
# 2. Run the fine-tuning and prediction
python fine_tune_predict.py
```

## 📝 Configuration & Hyperparameters

All key parameters are centralized in the `Config` section of `train.py`:

  * **Data:** `geno_path`, `pheno_path`, `source_traits`, `target_trait`
  * [cite\_start]**Model:** `input_len`, `hidden_dim`, `conv_channels`, `kernel_size`, `stride` [cite: 266, 267, 276]
  * [cite\_start]**Training:** `cv_folds`, `pre_epochs`, `ft_epochs`, `fu_epochs`, `batch_size`, `learning_rate` [cite: 322]
  * **Misc:** `random_seed`, `output_dir`

## 📈 Outputs & Visualization

Results are saved under `./results/` in a directory named with a timestamp and the target trait.

  * `Direct/`: `direct_{trait}.pth`, `direct_res.csv`
  * `Pretrain/`: `pretrain_{Trait}.png`, `pretrained_target_res.csv`
  * `Fine-tune/`: `fine_tuned_{trait}.pth`, `prediction_*.csv`
  * `Fusion/`:
      * `fusion_model_embedding.pth` (Final model)
      * [cite\_start]`fusion_attention_weights.png`/`.csv` (Macro-level interpretability [cite: 141])
      * `test_true_vs_pred.csv` (Final predictions)
  * `results_summary.csv`: Aggregated metrics for all models.

## 🗂️ Datasets

  * [cite\_start]**Wheat-599 & Wheat-2000:** Available from the [DNNGP repository](https://github.com/AIBreeding/DNNGP/blob/main/example-data.tgz) [cite: 329] or Baidu Netdisk (code: `eveq`): [link](https://pan.baidu.com/s/1ovsuCCxgL2PCwB8jR-e5tA?pwd=eveq)
  * [cite\_start]**Tomato-332:** Available from the [SolOmics database](http://solomics.agis.org.cn/tomato/ftp) [cite: 327] or this repository (`/datasets/tomato332`).
  * [cite\_start]**MaizeGEP:** Data is available from the corresponding author (K.W.) upon reasonable request[cite: 331].

## 🎓 Citation

[cite\_start]If you find this work useful, please consider giving a ⭐ and citing[cite: 15, 50]:

> Wang, J., Zhang, Y., Li, B., Piao, X., Zhao, X., Zhang, D., Wang, A., Zhang, B., & Wang, K. (2025). Mul-PheG2P: Decoupled learning and prediction-space fusion enables robust and interpretable multi-phenotype genomic prediction. *[Journal Name]*

```bibtex
@article{wang2025mulpheg2p,
  title={{Mul-PheG2P: Decoupled learning and prediction-space fusion enables robust and interpretable multi-phenotype genomic prediction}},
  author={Wang, Jiahui and Zhang, Yong and Li, Bo and Piao, Xinglin and Zhao, Xiangyu and Zhang, Dongfeng and Wang, Aiwen and Zhang, Bob and Wang, Kaiyi},
  journal={[Journal Name]},
  year={2025},
  volume={XX},
  pages={XXXX--XXXX},
  doi={[YOUR_PAPER_DOI_HERE]}
}
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
