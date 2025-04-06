# 📘 Grammar Scoring with Wav2Vec2 + Transformer Encoder

## 📂 `transformer_part/` — Sequence-Level Transformer Regression for Grammar Scoring

---

### 📌 Overview

This module implements a **Transformer-based sequence regression model** for grammar scoring, leveraging **Wav2Vec2** for speech representation extraction. Unlike feature-level aggregation used in traditional MLP approaches, this model utilizes the full sequence of contextual token embeddings, followed by a Transformer Encoder and **attention-based pooling** to regress **Mean Opinion Scores (MOS)** in the range of **1.0 to 5.0** with **0.5 resolution**.

---

### 🧠 Model Architecture

| Component              | Description                                                                |
|------------------------|----------------------------------------------------------------------------|
| **Feature Extractor**  | `torchaudio.pipelines.WAV2VEC2_BASE` (frozen, pretrained)                  |
| **Sequence Embedding** | Last hidden layer of Wav2Vec2 encoder (contextual embeddings)              |
| **Backbone**           | `nn.TransformerEncoder` (multi-layer, multi-head)                          |
| **Pooling**            | Attention-based pooling (`Linear → Softmax → Weighted sum`)                |
| **Output Head**        | `LayerNorm` → `Linear(1)` → Scalar MOS prediction                          |

---

### ⚙️ Core Components

| Function/Class        | Description                                                           |
|-----------------------|-----------------------------------------------------------------------|
| `train_model()`       | Trains the Transformer encoder + regression head                      |
| `evaluate_model()`    | Runs evaluation on the test set and generates prediction CSV           |
| `AudioDataset`        | PyTorch dataset for audio + label (or audio + filename for test mode) |
| `extract_sequence_features()` | Converts waveform to `[T × D]` sequence features               |
| `TransformerRegressor`| Transformer encoder + attention pooling + linear regressor            |

---

### 🧪 Training Configuration

| Hyperparameter        | Value         |
|------------------------|---------------|
| Optimizer              | `AdamW`       |
| Learning Rate          | `5e-4`        |
| Batch Size             | `8`           |
| Epochs                 | `50`          |
| Num Heads              | `4`           |
| Num Layers             | `3`           |
| Hidden Dim             | `256`         |
| Dropout Rate           | `0.3`         |
| Loss Function          | `MSELoss()`   |
| Scheduler              | `ReduceLROnPlateau` (patience=2, factor=0.9) |

---

### 📁 Directory Contents

| File                                  | Description                                       |
|---------------------------------------|---------------------------------------------------|
| `train_final.py`                      | Full training and evaluation pipeline             |
| `main.ipynb`                          | Interactive notebook for training/evaluation      |
| `final_transformer_model.pt`          | Trained model weights                             |
| `test_predictions_transformers_submit.csv` | Final predictions on test set               |
| `README.MD`                           | This documentation                                |

---

### 🚀 Execution Instructions

Run the pipeline directly via terminal:

```bash
cd transformer_part
python3 train_final.py
```

Upon completion:

- Trained model saved to → `final_transformer_model.pt`
- Predictions exported to → `test_predictions_transformers_submit.csv`

---

### 📊 Prediction Format

Output format of the prediction CSV:

```csv
filename,label
sample_001.wav,4.0
sample_002.wav,2.5
...
```

---

### ✅ Notes

- Feature extraction is GPU-accelerated using `WAV2VEC2_BASE`.
- Predictions leverage full token-level sequence processing via Transformer layers.
- Visualization and result analysis are handled within the `main.ipynb`.

---