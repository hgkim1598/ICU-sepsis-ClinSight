"""
SIC (Sepsis-Induced Coagulopathy) 예측 모델 정의
─────────────────────────────────────────────────────────────────────────────
아키텍처 : BiLSTM + XGBoost → Stacking Logistic Regression (5-fold OOF)

  Base 1 — LSTMClassifier (bidirectional)
            입력: X_seq  (N, 48, 41)  — 시계열 피처 41개 × 48 timestep
                  (padding_mask 제거 후 42 → 41)
            저장: checkpoints/fold_{k}/best_model.pt

  Base 2 — XGBClassifier
            입력: X_xgb  (N, 173)    — 정적 9개 + 시계열 집계 164개
                  (padding_mask × 4 agg 제거 후 179 → 173)
            저장: checkpoints/fold_{k}/model.json

  Meta   — LogisticRegression
            입력: [P_LSTM, P_XGB]  (N, 2)
            저장: checkpoints/meta/meta_model.pkl
"""

import torch
import torch.nn as nn
import xgboost as xgb


# ── LSTM 모델 ──────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """Bidirectional LSTM 이진 분류기.

    Args:
        input_size:    시계열 피처 수 (42)
        hidden_size:   LSTM hidden 차원 (128)
        num_layers:    LSTM 레이어 수 (2)
        dropout:       레이어 간 dropout (0.3)
        bidirectional: 양방향 여부 (True)
    """

    def __init__(
        self,
        input_size:    int,
        hidden_size:   int,
        num_layers:    int,
        dropout:       float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )
        num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(hidden_size * num_directions, 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (B, T, F_ts) float32
        Returns:
            logits (B,) — sigmoid 적용 전
        """
        output, _ = self.lstm(x_seq)        # (B, T, out_dim)
        h   = output[:, -1, :]              # last timestep
        h   = self.dropout(h)
        return self.fc(h).squeeze(-1)       # (B,)


def build_lstm(input_size: int) -> LSTMClassifier:
    """config.yaml lstm 섹션 기준으로 LSTMClassifier 를 생성합니다.

    Args:
        input_size: 시계열 피처 수 (X_seq.shape[2], 통상 41)

    Returns:
        LSTMClassifier 인스턴스
    """
    return LSTMClassifier(
        input_size    = input_size,
        hidden_size   = 128,
        num_layers    = 2,
        dropout       = 0.3,
        bidirectional = True,
    )


# ── XGBoost 모델 ───────────────────────────────────────────────────────────

def build_xgb(scale_pos_weight: float) -> xgb.XGBClassifier:
    """config.yaml xgboost 섹션 기준으로 XGBClassifier 를 생성합니다.

    Args:
        scale_pos_weight: n_neg / n_pos (train split 에서 계산)

    Returns:
        XGBClassifier 인스턴스
    """
    return xgb.XGBClassifier(
        n_estimators          = 1000,
        max_depth             = 6,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 5,
        gamma                 = 0.1,
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        eval_metric           = "aucpr",
        early_stopping_rounds = 50,
        tree_method           = "approx",
        n_jobs                = 1,
        scale_pos_weight      = scale_pos_weight,
        random_state          = 42,
    )
