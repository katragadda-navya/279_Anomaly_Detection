ACI-IoT-2023 Intrusion Detection using FT-Transformer, MLP & LightGBM

Overview
This project implements a machine learning–based intrusion detection system (IDS) for the ACI-IoT-2023 dataset (Army Cyber Institute IoT Network Dataset).
We build and compare three complementary models — LightGBM, Multi-Layer Perceptron (MLP), and a pure PyTorch FT-Transformer — to classify network traffic as benign or malicious across multiple attack types.

Project Motivation
IoT networks are widely deployed but inherently vulnerable due to limited device security and diverse protocols.
Traditional IDS systems rely on fixed signatures and can’t generalize to unseen threats.
This project leverages modern deep learning and tabular transformers to learn discriminative patterns from flow-based features automatically.

Dataset: ACI-IoT-2023
Source: Army Cyber Institute (USMA)
Type: Flow-based IoT network traffic
Samples: ≈ 2.4 million flows (balanced for experiments)

Features:
Numerical: packet sizes, flow duration, byte rates, etc.
Categorical: protocol, service, flow direction, etc.
Label: attack category (DDoS, PortScan, WebAttack, …)
Task: Multiclass classification (benign vs various attack types)

Pipeline Overview
1️. Preprocessing
Cleans missing / constant columns
Auto-detects categorical vs numerical features
GPU-accelerated scaling (PyTorch CUDA or CPU fallback)
Balances the training split (oversampling)

Saves standardized splits:
preprocessed_iot/
├── ftt_train.csv
├── ftt_val.csv
├── ftt_test.csv
├── ftt_scaler.joblib
├── label_mapping.json

2️. Models
Model	Type	Description
LightGBM	Gradient Boosted Trees	Fast baseline for tabular data
MLP	Deep Neural Network	Fully-connected layers with ReLU, dropout, AdamW optimizer
FT-Transformer	Pure PyTorch implementation	Tokenizes each feature, encodes via transformer blocks, and predicts using CLS head

3. Training
Early stopping on validation loss / F1
GPU acceleration (if available)
Lightweight configs for Colab stability
Balanced F1, accuracy, and confusion matrices for each model

4️. Deployment
Each model exports artifacts for deployment:
/content/
├── lgbm_model.txt
├── ohe.joblib
├── classes.json
├── ftt_pure_artifacts/
│   ├── model.pt
│   ├── meta.json
│   └── ftt_scaler.joblib

Model Comparison

| Model              | Accuracy | Macro-F1 | Inference (ms/sample) | Notes                                        |
| ------------------ | -------- | -------- | --------------------- | -------------------------------------------- |
| **LightGBM**       | ~97–98%  | ~0.97    | 0.2                   | Strong classical baseline                    |
| **MLP**            | ~97–98%  | ~0.97    | 0.3                   | Captures nonlinearities well                 |
| **FT-Transformer** | ~98–99%  | ~0.98    | 1.2                   | Best generalization and feature interactions |


(Actual numbers depend on dataset size & random seed.)
