# 🎯 Grammar Scoring Engine — Technical Overview

### 🧩 Problem Statement

The objective of this project is to build an automated **Grammar Scoring Engine** that processes spoken audio responses and assigns a **Grammar MOS (Mean Opinion Score)** between **1.0 and 5.0**, at 0.5-point granularity. The model must learn to infer grammar quality — a deeply **subjective and context-sensitive linguistic construct** — directly from **raw waveform audio** using machine learning.

---

### 🛠️ Core Challenges

This task presented multiple technical and real-world constraints:

| Challenge                        | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| **Data Regime**                  | **444 training audio files**, requiring aggressive transfer learning   |
| **Continuous Output Prediction** | Requires fine-grained **regression**, not classification                   |
| **Acoustic Variability**         | Varying accents, intonation, recording conditions, and pacing              |
| **Latency Constraints**          | Needs to be efficient enough for near-real-time evaluation pipelines       |

---

### 🚧 Our Dual-Track Strategy

To address the above challenges **robustly and modularly**, we designed **two complementary pipelines**, each targeting a different balance between computational complexity and expressive power:

| Module                | Strategy                                                                 |
|-----------------------|--------------------------------------------------------------------------|
| **`regressional_part/`** | 🧠 **Feature-level regression**: Wav2Vec2 embeddings → statistical pooling → MLP |
| **`transformer_part/`**  | 🔬 **Sequence-level regression**: Full token embeddings → Transformer encoder  |

Each of these modules operates **independently** and includes its **own `train_final.py` and `main.ipynb`**, making it easy to evaluate or extend them in isolation.

---

### 📦 Modular Folder Structure

```bash
GrammarScoring/
├── dataset/                                # Audio + CSV data
│   ├── audios_train/
│   ├── audios_test/
│   ├── train.csv
│   └── test.csv
│
├── regressional_part/                      # MLP Regression with pooled Wav2Vec2 features
│   ├── train_final.py                      # Training + evaluation pipeline
│   ├── main.ipynb                          # Visualizations & analysis
│   ├── README.MD                           # Readme for this module
│   ├── final_regression_model_submit.pt    # Modelweights
│   └── test_predictions_submit.csv         # Final predictions
│
├── transformer_part/                       # Transformer-based Regression
│   ├── train_final.py                      # Sequence-level regression pipeline
│   ├── main.ipynb                          # Visualizations & analysis
│   ├── README.MD                           # Readme for this module
│   ├── final_regression_model_submit.pt    # Modelweights
│   └── test_predictions_submit.csv         # Final predictions
│
├── .gitignore                              # .gitignore fiel 
├── download_dataset.py                     # Run this first to setup the dataset folder
├── requirements.txt                        # Requirements needed for this project  
└── README.md                               # Project-wide technical summary
```

---

### 🔍 How We Framed the Solution

Instead of treating this as a classic classification task, we modeled this as a **continuous-valued regression problem**, which is more aligned with the **subjective nature of MOS**. By leveraging **Wav2Vec2** for high-level speech representations, we removed the need for manual audio preprocessing, phoneme recognition, or grammar rule parsing.

We designed our models as follows:

| Design Element        | Details                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| **Pretrained Features** | `torchaudio.pipelines.WAV2VEC2_BASE`, fine-tuned only on regression head |
| **MLP Head**            | Efficient inference-time performance for fast scoring                  |
| **Transformer Head**    | Better captures token-wise grammatical structure from speech embeddings |
| **Loss Function**       | Mean Squared Error (MSE) + output rounding to nearest 0.5              |
| **Evaluation**          | Performed in `.ipynb` inside each module using ground truth CSVs       |

---
