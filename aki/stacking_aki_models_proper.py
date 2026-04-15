import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


SEQ_PATH = "data/processed/team_X_seq_with_mask.npy"
STATIC_PATH = "data/processed/team_X_static.npy"
Y_PATH = "data/processed/team_y.npy"

OUTPUT_DIR = "aki_stacking_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
PATIENCE = 3
SEED = 42


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AKIDataset(Dataset):
    def __init__(self, seq, static, y):
        self.seq = torch.tensor(seq, dtype=torch.float32)
        self.static = torch.tensor(static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.seq[idx], self.static[idx], self.y[idx]


class Model(nn.Module):
    def __init__(self, seq_dim, static_dim, model_type="lstm", hidden_dim=64):
        super().__init__()

        if model_type == "lstm":
            self.rnn = nn.LSTM(seq_dim, hidden_dim, batch_first=True)
        elif model_type == "gru":
            self.rnn = nn.GRU(seq_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("model_type must be 'lstm' or 'gru'")

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + static_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, seq, static):
        out, _ = self.rnn(seq)
        h = out[:, -1, :]
        x = torch.cat([h, static], dim=1)
        return self.fc(x).squeeze(1)


def safe_auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def safe_auprc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, p)


def train_model(model, train_loader, val_loader, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1
    patience_count = 0
    save_path = os.path.join(OUTPUT_DIR, f"aki_best_{name}.pt")

    for epoch in range(EPOCHS):
        model.train()
        for seq, static, y in train_loader:
            seq = seq.to(DEVICE)
            static = static.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(seq, static)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            loss = loss_fn(logits, y)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        preds = []
        labels = []

        with torch.no_grad():
            for seq, static, y in val_loader:
                seq = seq.to(DEVICE)
                static = static.to(DEVICE)
                logits = model(seq, static)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                prob = torch.sigmoid(logits).cpu().numpy()
                preds.extend(prob)
                labels.extend(y.numpy())

        preds = np.nan_to_num(np.asarray(preds), nan=0.5, posinf=1.0, neginf=0.0)
        labels = np.asarray(labels)
        val_auc = safe_auc(labels, preds)
        val_auc_str = f"{val_auc:.4f}" if not np.isnan(val_auc) else "nan"

        print(f"{name.upper()} Epoch {epoch + 1}/{EPOCHS} | Val AUC: {val_auc_str}")

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            patience_count = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"{name.upper()} early stopping")
            break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model


def get_preds(model, loader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for seq, static, y in loader:
            seq = seq.to(DEVICE)
            static = static.to(DEVICE)
            logits = model(seq, static)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            prob = torch.sigmoid(logits).cpu().numpy()
            preds.extend(prob)
            labels.extend(y.numpy())

    preds = np.nan_to_num(np.asarray(preds), nan=0.5, posinf=1.0, neginf=0.0)
    labels = np.asarray(labels)
    return preds, labels


def main():
    set_seed(SEED)

    print("Loading data...")
    print("SEQ_PATH   :", SEQ_PATH)
    print("STATIC_PATH:", STATIC_PATH)
    print("Y_PATH     :", Y_PATH)

    X_seq = np.load(SEQ_PATH)
    X_static = np.load(STATIC_PATH)
    y = np.load(Y_PATH)

    X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_static = np.nan_to_num(X_static, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1).astype(np.float32)
    y = (y > 0).astype(np.float32)

    print("X_seq shape   :", X_seq.shape)
    print("X_static shape:", X_static.shape)
    print("y shape       :", y.shape)

    if len(X_seq) != len(X_static) or len(X_seq) != len(y):
        raise ValueError("Input length mismatch")

    scaler_static = StandardScaler()
    scaler_seq = StandardScaler()

    X_static = scaler_static.fit_transform(X_static)

    n, t, f = X_seq.shape
    X_seq = scaler_seq.fit_transform(X_seq.reshape(-1, f)).reshape(n, t, f)

    joblib.dump(scaler_static, os.path.join(OUTPUT_DIR, "scaler_static.pkl"))
    joblib.dump(scaler_seq, os.path.join(OUTPUT_DIR, "scaler_seq.pkl"))

    idx = np.arange(len(y))
    np.random.shuffle(idx)

    train_idx = idx[: int(0.7 * len(y))]
    val_idx = idx[int(0.7 * len(y)) : int(0.85 * len(y))]
    test_idx = idx[int(0.85 * len(y)) :]

    train_loader = DataLoader(
        AKIDataset(X_seq[train_idx], X_static[train_idx], y[train_idx]),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        AKIDataset(X_seq[val_idx], X_static[val_idx], y[val_idx]),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        AKIDataset(X_seq[test_idx], X_static[test_idx], y[test_idx]),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("Train size:", len(train_idx))
    print("Val size  :", len(val_idx))
    print("Test size :", len(test_idx))

    lstm = Model(X_seq.shape[2], X_static.shape[1], "lstm").to(DEVICE)
    gru = Model(X_seq.shape[2], X_static.shape[1], "gru").to(DEVICE)

    lstm = train_model(lstm, train_loader, val_loader, "lstm")
    gru = train_model(gru, train_loader, val_loader, "gru")

    lstm_val, y_val = get_preds(lstm, val_loader)
    gru_val, _ = get_preds(gru, val_loader)

    lstm_test, y_test = get_preds(lstm, test_loader)
    gru_test, _ = get_preds(gru, test_loader)

    X_meta_val = np.column_stack([lstm_val, gru_val])
    X_meta_test = np.column_stack([lstm_test, gru_test])

    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X_meta_val, y_val)

    joblib.dump(lr, os.path.join(OUTPUT_DIR, "stacking_lr.pkl"))

    pred = lr.predict_proba(X_meta_test)[:, 1]

    final_auc = safe_auc(y_test, pred)
    final_auprc = safe_auprc(y_test, pred)

    print("\nFinal Test AUC  :", final_auc)
    print("Final Test AUPRC:", final_auprc)

    metrics = {
        "test_auc": None if np.isnan(final_auc) else float(final_auc),
        "test_auprc": None if np.isnan(final_auprc) else float(final_auprc),
    }

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved files:")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        print("-", fn)


if __name__ == "__main__":
    main()