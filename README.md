# CausaOne-Sign: Causal Explainable One-Shot Signature Verification

This repository provides a modular implementation of the paper:

**"CausaOne-Sign: Causal Explainable One-Shot Signature Verification with Lightweight Cross-Modality Fusion"**

## 📁 Project Structure

- `1_graph_construction.py`: Preprocess signature images and convert them into stroke-aware graphs.
- `2_graph_encoder.py`: Encode graph structures using a Graph Attention Network.
- `3_prototypical_training.py`: Implements episodic one-shot training using prototypical and contrastive losses.
- `4_graph_transformer_explainable.py`: Graph Transformer with attention and Individual Causal Attribution (ICA) for interpretability.
- `5_meta_learning_maml.py`: Implements Model-Agnostic Meta-Learning (MAML) adaptation for unseen users.
- `6_distillation_and_pruning.py`: Applies model compression through pruning and knowledge distillation.
- `7_evaluation_metrics.py`: Evaluation utilities for accuracy, AUC, and F1-score.

## 📝 Datasets Used
- **CEDAR**: English signatures with skilled forgeries.
- **SigComp2011**: Mixed-script benchmark for signature verification.
- **UTSig**: Persian offline signature dataset.
- **BHSig260**: Hindi and Bengali offline signatures.

## ⚙️ Requirements
- Python 3.8+
- PyTorch
- Torch-Geometric
- OpenCV
- NetworkX
- NumPy
- scikit-learn

Install with:
```bash
pip install -r requirements.txt
```

## 🚀 Run Example

1. Preprocess and convert images to graphs:
```bash
python 1_graph_construction.py
```

2. Encode and embed graphs:
```bash
python 2_graph_encoder.py
```

3. Train using episodic one-shot setup:
```bash
python 3_prototypical_training.py
```

4. Enable explainability using causal reasoning:
```bash
python 4_graph_transformer_explainable.py
```

## 🔒 License
This project is licensed under the MIT License.
