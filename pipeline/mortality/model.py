import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.classifier(h).squeeze(-1)