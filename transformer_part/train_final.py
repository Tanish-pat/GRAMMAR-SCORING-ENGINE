import os, torch, torchaudio, pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# -----------------------------
# Environment Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_BASE
wav2vec_model = bundle.get_model().to(device).eval()

# -----------------------------
# Configurations
# -----------------------------
TRAIN_AUDIO_DIR = "../dataset/audios_train"
TEST_AUDIO_DIR = "../dataset/audios_test"
TRAIN_CSV_PATH = "../dataset/train.csv"
TEST_CSV_PATH = "../dataset/test.csv"

FINAL_MODEL_PATH = "final_transformer_model.pt"
PREDICTIONS_CSV = "test_predictions_transformers_submit.csv"

BEST_LEARNING_RATE = 5e-4
BEST_BATCH_SIZE = 8
BEST_EPOCHS = 50
BEST_NUM_HEADS = 4
BEST_NUM_LAYERS = 3
BEST_HIDDEN_DIM = 256
BEST_DROPOUT = 0.3

# -----------------------------
# Dataset Class
# -----------------------------
class AudioDataset(Dataset):
    def __init__(self, csv_path, audio_dir, is_test=False):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(os.path.join(self.audio_dir, row['filename']))
        if sr != bundle.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, bundle.sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.to(device)

        if self.is_test:
            return waveform, row['filename']
        else:
            label = torch.tensor(float(row['label']), dtype=torch.float32).to(device)
            return waveform, label

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_sequence_features(waveform):
    with torch.no_grad():
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        features, _ = wav2vec_model.extract_features(waveform)
        return features[-1]  # [B, T, D]

# -----------------------------
# Transformer-Based Regressor
# -----------------------------
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, dropout):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.attn_pool = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Softmax(dim=1)
        )

        self.regressor = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):  # x: [B, T, D]
        x = self.encoder(x)  # [B, T, D]
        attn_weights = self.attn_pool(x)  # [B, T, 1]
        pooled = (attn_weights * x).sum(dim=1)  # [B, D]
        return self.regressor(pooled).squeeze(1)

# -----------------------------
# Collate Functions
# -----------------------------
def collate_fn_train(batch):
    waveforms = [x[0] for x in batch]
    labels = torch.stack([x[1] for x in batch])
    features = [extract_sequence_features(wf.unsqueeze(0)).squeeze(0).cpu() for wf in waveforms]
    features_padded = pad_sequence(features, batch_first=True)  # [B, T, D]
    return features_padded.to(device), labels

def collate_fn_test(batch):
    waveforms = [x[0] for x in batch]
    filenames = [x[1] for x in batch]
    features = [extract_sequence_features(wf.unsqueeze(0)).squeeze(0).cpu() for wf in waveforms]
    features_padded = pad_sequence(features, batch_first=True)
    return features_padded.to(device), filenames

# -----------------------------
# Training Function
# -----------------------------
def train_model():
    train_dataset = AudioDataset(TRAIN_CSV_PATH, TRAIN_AUDIO_DIR, is_test=False)
    train_loader = DataLoader(train_dataset, batch_size=BEST_BATCH_SIZE, shuffle=True, collate_fn=collate_fn_train)

    # Get input_dim from dummy input
    input_dim = extract_sequence_features(torch.randn(1, 16000).to(device)).shape[2]
    model = TransformerRegressor(
        input_dim=input_dim,
        num_heads=BEST_NUM_HEADS,
        num_layers=BEST_NUM_LAYERS,
        hidden_dim=BEST_HIDDEN_DIM,
        dropout=BEST_DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=BEST_LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=2, min_lr=1e-6)
    criterion = nn.MSELoss()

    print("🚀 Starting training...")
    for epoch in range(BEST_EPOCHS):
        model.train()
        total_loss = 0.0
        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{BEST_EPOCHS} [Train]"):
            preds = model(feats)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f} | LR = {current_lr:.6f}")
        scheduler.step(avg_loss)

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"✔️ Final model saved to {FINAL_MODEL_PATH}")

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model():
    if not os.path.exists(FINAL_MODEL_PATH):
        print(f"❌ Error: Model not found at {FINAL_MODEL_PATH}")
        return

    test_dataset = AudioDataset(TEST_CSV_PATH, TEST_AUDIO_DIR, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BEST_BATCH_SIZE, shuffle=False, collate_fn=collate_fn_test)

    input_dim = extract_sequence_features(torch.randn(1, 16000).to(device)).shape[2]
    model = TransformerRegressor(
        input_dim=input_dim,
        num_heads=BEST_NUM_HEADS,
        num_layers=BEST_NUM_LAYERS,
        hidden_dim=BEST_HIDDEN_DIM,
        dropout=BEST_DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(FINAL_MODEL_PATH))
    model.eval()

    all_preds = []
    all_filenames = []

    print("🔍 Evaluating model...")
    with torch.no_grad():
        for feats, filenames in tqdm(test_loader, desc="Evaluation"):
            preds = model(feats)
            all_preds.extend(preds.cpu().numpy())
            all_filenames.extend(filenames)

    output_df = pd.DataFrame({
        'filename': all_filenames,
        'label': all_preds
    })

    output_df.iloc[:, 1] = output_df.iloc[:, 1].apply(lambda x: round(x * 2) / 2)
    output_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"✔️ Predictions saved to {PREDICTIONS_CSV}")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    train_model()
    evaluate_model()
