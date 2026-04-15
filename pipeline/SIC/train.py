"""
SIC 모델 학습 스크립트 — 5-fold OOF Stacking
─────────────────────────────────────────────────────────────────────────────
데이터:
  data/dataset/dl/{train,val}.pkl          — X_seq (N, 48, 41), X_static, y
                                             (padding_mask 제거 후 42 → 41)
  data/dataset/xgb/{train,val}.parquet     — 집계 피처 (N, 173+)
  data/dataset/scaler.pkl                  — StandardScaler 번들

출력 (--output-dir, 기본 ./checkpoints):
  fold_{1..5}/
    best_model.pt         LSTM 가중치
    model.json            XGBoost 가중치
    feature_names.json    XGB 피처명
    training_history.json LSTM 학습 이력
  meta/
    meta_model.pkl        Stacking LogisticRegression

사용법:
  python train.py --data-dir ./data/dataset --output-dir ./checkpoints
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

import xgboost as xgb

from model import LSTMClassifier, build_lstm, build_xgb

# ── 설정 ───────────────────────────────────────────────────────────────────
SEED     = 42
N_FOLDS  = 5
SEQ_LEN  = 48
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DL_CFG = dict(
    batch_size    = 256,
    max_epochs    = 50,
    patience      = 7,
    learning_rate = 0.001,
    weight_decay  = 0.0001,
    focal_alpha   = 0.5,
    focal_gamma   = 2.0,
)

ID_COL    = "stay_id"
LABEL_COL = "label"


# ── 공통 컴포넌트 ──────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Binary Focal Loss with optional pos_weight."""

    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha      = alpha
        self.gamma      = gamma
        self.pos_weight = (
            torch.tensor([pos_weight], dtype=torch.float32)
            if pos_weight is not None else None
        )

    def forward(self, logits, targets):
        targets = targets.float()
        pw  = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )
        prob     = torch.sigmoid(logits)
        p_t      = targets * prob + (1 - targets) * (1 - prob)
        alpha_t  = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss     = (alpha_t * (1 - p_t) ** self.gamma * bce).mean()
        return loss


class SequenceDataset(Dataset):
    def __init__(self, x_seq, y):
        self.x = torch.from_numpy(np.asarray(x_seq, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y,     dtype=np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def _pos_weight(y):
    n_pos = int(y.sum())
    return (len(y) - n_pos) / max(n_pos, 1)


# ── 데이터 로드 ────────────────────────────────────────────────────────────

def _load_dl(path: Path):
    """pkl → (X_seq, y, stay_ids)"""
    with path.open("rb") as f:
        d = pickle.load(f)
    return (
        np.asarray(d["X_seq"],    dtype=np.float32),
        np.asarray(d["y"],        dtype=np.int32),
        np.asarray(d["stay_ids"]),
    )


def _load_xgb(path: Path, stay_ids_ref, feat_cols=None):
    """parquet → (X_xgb, feat_cols) — stay_ids_ref 순서로 정렬"""
    df = pd.read_parquet(path)
    df = df.set_index(ID_COL).loc[stay_ids_ref].reset_index()
    if feat_cols is None:
        feat_cols = [c for c in df.columns if c not in {ID_COL, LABEL_COL}]
    X = df[feat_cols].values.astype(np.float32)
    return X, feat_cols


# ── LSTM 학습 루프 ─────────────────────────────────────────────────────────

def _train_lstm(model, X_tr, y_tr, X_vl, y_vl, ckpt_path: Path, logger):
    """LSTM 학습 + early stopping → best_model.pt 저장."""
    pw         = _pos_weight(y_tr)
    train_crit = FocalLoss(DL_CFG["focal_alpha"], DL_CFG["focal_gamma"], pw)
    val_crit   = FocalLoss(DL_CFG["focal_alpha"], DL_CFG["focal_gamma"], None)

    loader = DataLoader(
        SequenceDataset(X_tr, y_tr),
        batch_size = DL_CFG["batch_size"],
        shuffle    = True,
        num_workers = 0,
    )
    optim = torch.optim.Adam(
        model.parameters(),
        lr           = DL_CFG["learning_rate"],
        weight_decay = DL_CFG["weight_decay"],
    )
    model.to(DEVICE)

    X_vl_t = torch.from_numpy(X_vl).to(DEVICE)
    y_vl_t = torch.from_numpy(y_vl.astype(np.float32)).to(DEVICE)

    best_loss, no_improve, history = float("inf"), 0, []
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, DL_CFG["max_epochs"] + 1):
        model.train()
        for x_b, y_b in loader:
            optim.zero_grad()
            loss = train_crit(model(x_b.to(DEVICE)), y_b.to(DEVICE))
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            vl_loss  = val_crit(model(X_vl_t), y_vl_t).item()
            vl_probs = torch.sigmoid(model(X_vl_t)).cpu().numpy()
        try:
            auroc = roc_auc_score(y_vl, vl_probs)
        except ValueError:
            auroc = float("nan")

        history.append({"epoch": epoch, "val_loss": round(vl_loss, 6),
                        "val_auroc": round(auroc, 6)})
        logger.info(f"    epoch {epoch:3d}  val_loss={vl_loss:.4f}  auroc={auroc:.4f}")

        if vl_loss < best_loss:
            best_loss, no_improve = vl_loss, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= DL_CFG["patience"]:
                logger.info(f"    Early stop at epoch {epoch}")
                break

    hist_path = ckpt_path.parent / "training_history.json"
    with hist_path.open("w") as f:
        json.dump(history, f, indent=2)

    return best_loss


def _lstm_batch_predict(model, X_seq):
    """X_seq → sigmoid 확률 (N,)"""
    model.eval()
    ds     = SequenceDataset(X_seq, np.zeros(len(X_seq)))
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    probs  = []
    with torch.no_grad():
        for x_b, _ in loader:
            probs.append(torch.sigmoid(model(x_b.to(DEVICE))).cpu().numpy())
    return np.concatenate(probs).astype(np.float32)


# ── 5-fold OOF 학습 ────────────────────────────────────────────────────────

def run_train(data_dir: Path, output_dir: Path, logger):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    logger.info("데이터 로드 중...")
    X_seq_tr, y_tr, ids_tr = _load_dl(data_dir / "dl" / "train.pkl")
    X_seq_vl, y_vl, ids_vl = _load_dl(data_dir / "dl" / "val.pkl")
    X_xgb_tr, feat_cols    = _load_xgb(data_dir / "xgb" / "train.parquet", ids_tr)
    X_xgb_vl, _            = _load_xgb(data_dir / "xgb" / "val.parquet",   ids_vl, feat_cols)

    input_size = X_seq_tr.shape[2]
    N          = len(y_tr)
    logger.info(f"  train N={N:,}  pos={y_tr.sum():,}  input_size={input_size}")
    logger.info(f"  val   N={len(y_vl):,}  xgb_features={len(feat_cols)}")

    # OOF 예측 배열
    p_lstm_oof = np.zeros(N, dtype=np.float32)
    p_xgb_oof  = np.zeros(N, dtype=np.float32)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_seq_tr, y_tr), start=1):
        logger.info("─" * 50)
        logger.info(f"[Fold {fold}/{N_FOLDS}]  train={len(tr_idx):,}  val={len(vl_idx):,}")
        fold_dir = output_dir / f"fold_{fold}"

        # ── LSTM ─────────────────────────────────────────────────────────
        lstm = build_lstm(input_size)
        logger.info(f"  LSTM params={sum(p.numel() for p in lstm.parameters()):,}")
        _train_lstm(
            lstm,
            X_seq_tr[tr_idx], y_tr[tr_idx],
            X_seq_tr[vl_idx], y_tr[vl_idx],
            fold_dir / "best_model.pt",
            logger,
        )
        lstm.load_state_dict(
            torch.load(fold_dir / "best_model.pt", map_location=DEVICE, weights_only=True)
        )
        p_lstm_oof[vl_idx] = _lstm_batch_predict(lstm, X_seq_tr[vl_idx])
        logger.info(f"  LSTM OOF fold {fold}: mean={p_lstm_oof[vl_idx].mean():.4f}")

        # ── XGBoost ───────────────────────────────────────────────────────
        spw     = _pos_weight(y_tr[tr_idx])
        clf_xgb = build_xgb(spw)
        clf_xgb.fit(
            X_xgb_tr[tr_idx], y_tr[tr_idx],
            eval_set=[(X_xgb_tr[vl_idx], y_tr[vl_idx])],
            verbose=False,
        )
        clf_xgb.save_model(str(fold_dir / "model.json"))
        with (fold_dir / "feature_names.json").open("w") as f:
            json.dump(feat_cols, f, indent=2)

        p_xgb_oof[vl_idx] = clf_xgb.predict_proba(X_xgb_tr[vl_idx])[:, 1]
        logger.info(f"  XGB  OOF fold {fold}: mean={p_xgb_oof[vl_idx].mean():.4f}")

    # ── Meta LR 학습 ──────────────────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("[Meta] Logistic Regression 학습...")
    X_meta_tr = np.column_stack([p_lstm_oof, p_xgb_oof])

    meta_clf = LogisticRegression(
        max_iter=1000, solver="lbfgs", class_weight="balanced", C=1.0, random_state=SEED
    )
    meta_clf.fit(X_meta_tr, y_tr)

    meta_dir = output_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    with (meta_dir / "meta_model.pkl").open("wb") as f:
        pickle.dump(meta_clf, f)
    logger.info(f"  Meta LR 저장 → {meta_dir / 'meta_model.pkl'}")

    # ── Val 앙상블 평가 ───────────────────────────────────────────────────
    logger.info("[Val] 앙상블 예측 중...")
    p_lstm_vl = np.zeros(len(y_vl), dtype=np.float32)
    p_xgb_vl  = np.zeros(len(y_vl), dtype=np.float32)

    for fold in range(1, N_FOLDS + 1):
        fold_dir = output_dir / f"fold_{fold}"
        lstm = build_lstm(input_size)
        lstm.load_state_dict(
            torch.load(fold_dir / "best_model.pt", map_location=DEVICE, weights_only=True)
        )
        p_lstm_vl += _lstm_batch_predict(lstm, X_seq_vl)

        clf_xgb = xgb.XGBClassifier(n_jobs=1)
        clf_xgb.load_model(str(fold_dir / "model.json"))
        p_xgb_vl += clf_xgb.predict_proba(X_xgb_vl)[:, 1].astype(np.float32)

    p_lstm_vl /= N_FOLDS
    p_xgb_vl  /= N_FOLDS

    X_meta_vl  = np.column_stack([p_lstm_vl, p_xgb_vl])
    p_final_vl = meta_clf.predict_proba(X_meta_vl)[:, 1]

    auroc = roc_auc_score(y_vl, p_final_vl)
    logger.info(f"  Val AUROC (앙상블): {auroc:.4f}")
    logger.info("학습 완료.")


# ── CLI ────────────────────────────────────────────────────────────────────

def _get_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("sic_train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIC 5-fold OOF 학습")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="data/dataset/ 경로 (dl/, xgb/ 하위 폴더 포함)",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints",
        help="체크포인트 저장 경로 (기본: ./checkpoints)",
    )
    args   = parser.parse_args()
    logger = _get_logger()

    logger.info(f"device : {DEVICE}")
    logger.info(f"data   : {args.data_dir}")
    logger.info(f"output : {args.output_dir}")

    run_train(
        data_dir   = Path(args.data_dir),
        output_dir = Path(args.output_dir),
        logger     = logger,
    )
