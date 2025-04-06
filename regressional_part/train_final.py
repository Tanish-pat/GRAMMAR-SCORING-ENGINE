import os, torch, torchaudio, pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_BASE
wav2vec_model = bundle.get_model().to(device).eval()

TRAIN_AUDIO_DIR = "../dataset/audios_train"
TEST_AUDIO_DIR = "../dataset/audios_test"
TRAIN_CSV_PATH = "../dataset/train.csv"
TEST_CSV_PATH = "../dataset/test.csv"

BEST_LEARNING_RATE = 0.0004956131596941485
BEST_BATCH_SIZE = 32
# BEST_EPOCHS = 30
BEST_EPOCHS = 60
BEST_DROPOUT_RATE = 0.3183369837123387
BEST_NUM_HIDDEN_LAYERS = 2
BEST_HIDDEN_DIM = 128
# FINAL_MODEL_PATH = "final_regression_model.pt"
FINAL_MODEL_PATH = "final_regression_model_submit.pt"

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

def extract_features(waveform):
    with torch.no_grad():
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        features, _ = wav2vec_model.extract_features(waveform)
        selected_layers = features[-3:]
        stat_feats = [torch.cat([layer.mean(1), layer.std(1)], dim=1) for layer in selected_layers]
        return torch.cat(stat_feats, dim=1)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class RegressionHead(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_dim, dropout_rate):
        super().__init__()
        layers = [nn.LayerNorm(input_dim)]
        current_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                Swish(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x).squeeze(1)

def collate_fn_train(batch):
    waveforms = [x[0] for x in batch]
    labels = torch.stack([x[1] for x in batch])
    return waveforms, labels

def collate_fn_test(batch):
    waveforms = [x[0] for x in batch]
    filenames = [x[1] for x in batch]
    return waveforms, filenames

def train_final():
    train_dataset = AudioDataset(TRAIN_CSV_PATH, TRAIN_AUDIO_DIR, is_test=False)
    train_loader = DataLoader(train_dataset, batch_size=BEST_BATCH_SIZE, shuffle=True, collate_fn=collate_fn_train)

    input_dim = extract_features(torch.randn(1, 16000).to(device)).shape[1]
    final_model = RegressionHead(input_dim, BEST_NUM_HIDDEN_LAYERS, BEST_HIDDEN_DIM, BEST_DROPOUT_RATE).to(device)

    optimizer = torch.optim.AdamW(final_model.parameters(), lr=BEST_LEARNING_RATE)
    criterion = nn.MSELoss()

    print("Starting final training with best hyperparameters...")
    for epoch in range(BEST_EPOCHS):
        final_model.train()
        total_loss = 0.0
        for waveforms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{BEST_EPOCHS} [Train]"):
            feats = torch.cat([extract_features(wf.unsqueeze(0)).cpu() for wf in waveforms], dim=0).to(device)
            preds = final_model(feats)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{BEST_EPOCHS}: Train Loss={avg_train_loss:.4f}")

    torch.save(final_model.state_dict(), FINAL_MODEL_PATH)
    print(f"✔️ Final model saved to {FINAL_MODEL_PATH}")

def evaluate_final():
    if not os.path.exists(FINAL_MODEL_PATH):
        print(f"Error: Final model not found at {FINAL_MODEL_PATH}. Please run train_final() first.")
        return

    test_dataset = AudioDataset(TEST_CSV_PATH, TEST_AUDIO_DIR, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BEST_BATCH_SIZE, shuffle=False, collate_fn=collate_fn_test)

    input_dim = extract_features(torch.randn(1, 16000).to(device)).shape[1]
    final_model = RegressionHead(input_dim, BEST_NUM_HIDDEN_LAYERS, BEST_HIDDEN_DIM, BEST_DROPOUT_RATE).to(device)
    final_model.load_state_dict(torch.load(FINAL_MODEL_PATH))
    final_model.eval()

    print("Starting final evaluation on the test set...")
    all_preds = []
    all_filenames = []

    with torch.no_grad():
        for waveforms, filenames in tqdm(test_loader, desc="Evaluating"):
            feats = torch.cat([extract_features(wf.unsqueeze(0)).cpu() for wf in waveforms], dim=0).to(device)
            preds = final_model(feats)
            all_preds.extend(preds.cpu().numpy())
            all_filenames.extend(filenames)

    output_df = pd.DataFrame({
        'filename': all_filenames,
        'label': all_preds
    })

    output_df.iloc[:, 1] = output_df.iloc[:, 1].apply(lambda x: round(x * 2) / 2)
    output_df.to_csv("test_predictions_regression_submit.csv", index=False)

    print("✔️ Predictions saved to test_predictions_regression_submit.csv")

if __name__ == "__main__":
    train_final()
    evaluate_final()
