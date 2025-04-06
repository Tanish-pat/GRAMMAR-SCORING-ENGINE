# Grammar Scoring with Wav2Vec2 Regression

## 📂 `regressional_part/` — Wav2Vec2-Based Regression Model for Grammar Scoring

### 📌 Overview

This module implements a regression-based grammar scoring engine using **Wav2Vec2 feature extraction** and a lightweight custom **Multi-Layer Perceptron (MLP)**. The model is trained to predict **Mean Opinion Scores (MOS)** in the range of 1.0 to 5.0 (with 0.5 increments) from raw audio waveforms. It leverages transfer learning from pretrained speech representations to regress MOS scores directly from audio.

---

### 🧠 Model Architecture

- **Feature Extractor**: Pretrained `WAV2VEC2_BASE` model (from `torchaudio.pipelines`)
- **Feature Strategy**: Last three hidden layers → Mean & Std pooling → Concatenated feature vector
- **Regression Head**:
  - LayerNorm
  - `N` Hidden Layers (each with Linear → Swish → Dropout)
  - Final Linear layer (1D output)

---

### ⚙️ Key Components

| Component            | Description                                         |
|----------------------|-----------------------------------------------------|
| `train_final()`      | Trains the regression head using extracted features |
| `evaluate_final()`   | Generates predictions on test data and saves to CSV |
| `AudioDataset`       | PyTorch Dataset class handling audio/label parsing  |
| `extract_features()` | Converts waveform → statistical embeddings          |
| `Swish`              | Custom activation function (x * sigmoid(x))         |
| `RegressionHead`     | MLP block for final score regression                |

---

### 🧪 Training Details

| Hyperparameter        | Value                                |
|-----------------------|--------------------------------------|
| Optimizer             | AdamW                                |
| Learning Rate         | `0.0004956131596941485`              |
| Batch Size            | `32`                                 |
| Epochs                | `80`                                 |
| Dropout Rate          | `~0.3183369837123387`                |
| Hidden Layers         | `2`                                  |
| Hidden Units          | `128`                                |
| Loss Function         | `MSELoss()`                          |

---

### 📁 Files

| File                               | Description                                        |
|------------------------------------|----------------------------------------------------|
| `train_final.py`                   | Training & evaluation pipeline                     |
| `main.ipynb`                       | Interactive training/evaluation notebook           |
| `final_regression_model_submit.pt` | Final trained PyTorch model weights                |
| `test_predictions_submit.csv`      | Final output predictions for the test set          |
| `README.MD`                        | This documentation                                 |

---

### 🚀 Execution

Run the training and evaluation script directly:

```bash
cd regressional_part
python train_final.py
```

Upon completion:
- Model is saved to: `final_regression_model_submit.pt`
- Predictions are exported to: `test_predictions_submit.csv`

---

### 📈 Output Format

```csv
filename,label
sample_001.wav,x.0
sample_002.wav,x.5
...
```

---