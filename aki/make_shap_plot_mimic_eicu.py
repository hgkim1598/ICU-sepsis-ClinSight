import os
import numpy as np
import pandas as pd
import torch
import shap
import joblib
import matplotlib.pyplot as plt

from stacking_aki_models_proper import Model


# =========================
# 설정
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "shap_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = "aki_stacking_outputs/aki_best_lstm.pt"
SCALER_STATIC_PATH = "aki_stacking_outputs/scaler_static.pkl"
SCALER_SEQ_PATH = "aki_stacking_outputs/scaler_seq.pkl"

STATIC_COLUMNS_PATH = "team_X_static_columns.csv"

# MIMIC
MIMIC_SEQ_PATH = "data/processed/team_X_seq_with_mask.npy"
MIMIC_STATIC_PATH = "data/processed/team_X_static.npy"

# eICU
EICU_SEQ_PATH = "eicu_processed/eicu_team_X_seq_with_mask.npy"
EICU_STATIC_PATH = "eicu_processed/eicu_team_X_static.npy"

# 시계열 feature 이름 (14개)
SEQ_FEATURE_NAMES = [
    "heart_rate",
    "mbp",
    "sbp",
    "dbp",
    "resp_rate",
    "spo2",
    "temperature",
    "gcs",
    "pao2fio2ratio",
    "heart_rate_mask",
    "mbp_mask",
    "sbp_mask",
    "dbp_mask",
    "resp_rate_mask",
]

BACKGROUND_SIZE = 100
EXPLAIN_SIZE = 300


# =========================
# static feature 이름 로드
# =========================
def load_static_feature_names(columns_path: str, expected_dim: int) -> list[str]:
    df = pd.read_csv(columns_path)

    cols = [str(c).strip() for c in df.columns.tolist()]

    drop_cols = {
        "Unnamed: 0",
        "subject_id",
        "stay_id",
        "hadm_id",
        "aki_label",
        "label",
        "y",
        "index",
    }
    cols = [c for c in cols if c not in drop_cols and not c.startswith("Unnamed")]

    if len(cols) <= 2 and df.shape[0] > 0:
        vals = df.iloc[0].dropna().astype(str).tolist()
        vals = [v.strip() for v in vals if v.strip() not in drop_cols and not v.startswith("Unnamed")]
        if len(vals) >= expected_dim:
            cols = vals

    if len(cols) != expected_dim:
        print(f"[WARN] static feature name count mismatch before fix: {len(cols)}")
        print(f"[WARN] raw static names: {cols}")
        cols = cols[:expected_dim]

    if len(cols) != expected_dim:
        raise ValueError(
            f"static feature count mismatch after fix (expected={expected_dim}, got={len(cols)})"
        )

    return cols


# =========================
# 모델 로드
# =========================
def load_model(seq_dim: int, static_dim: int) -> Model:
    model = Model(seq_dim, static_dim, "lstm").to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


# =========================
# 데이터 준비
# =========================
def prepare_data(seq_path: str, static_path: str):
    X_seq = np.load(seq_path)
    X_static = np.load(static_path)

    X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_static = np.nan_to_num(X_static, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    print(f"Loaded seq   : {seq_path} -> {X_seq.shape}")
    print(f"Loaded static: {static_path} -> {X_static.shape}")

    scaler_static = joblib.load(SCALER_STATIC_PATH)
    scaler_seq = joblib.load(SCALER_SEQ_PATH)

    X_static = scaler_static.transform(X_static)

    n, t, f = X_seq.shape
    X_seq = scaler_seq.transform(X_seq.reshape(-1, f)).reshape(n, t, f)

    return X_seq, X_static


# =========================
# SHAP용 예측 함수
# =========================
def make_predict_fn(model: Model, seq_timesteps: int, seq_feature_dim: int):
    def predict_fn(x: np.ndarray) -> np.ndarray:
        seq_part = x[:, :seq_feature_dim]
        static_part = x[:, seq_feature_dim:]

        # 마지막 timestep 정보를 전체 timestep에 반복
        seq_full = np.repeat(seq_part[:, None, :], seq_timesteps, axis=1)

        seq_tensor = torch.tensor(seq_full, dtype=torch.float32).to(DEVICE)
        static_tensor = torch.tensor(static_part, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits = model(seq_tensor, static_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()

        return probs

    return predict_fn


# =========================
# SHAP plot 생성
# =========================
def make_shap_plots(
    seq_path: str,
    static_path: str,
    output_prefix: str,
    static_columns_path: str,
):
    X_seq, X_static = prepare_data(seq_path, static_path)

    static_feature_names = load_static_feature_names(
        static_columns_path,
        expected_dim=X_static.shape[1],
    )

    print(f"{output_prefix} static dim: {X_static.shape[1]}")
    print(f"{output_prefix} static names count: {len(static_feature_names)}")
    print(f"{output_prefix} static names: {static_feature_names}")

    if X_static.shape[1] != len(static_feature_names):
        raise ValueError(
            f"{output_prefix}: static feature 개수 불일치 "
            f"(data={X_static.shape[1]}, names={len(static_feature_names)})"
        )

    if X_seq.shape[2] != len(SEQ_FEATURE_NAMES):
        raise ValueError(
            f"{output_prefix}: seq feature 개수 불일치 "
            f"(data={X_seq.shape[2]}, names={len(SEQ_FEATURE_NAMES)})"
        )

    feature_names = SEQ_FEATURE_NAMES + static_feature_names

    # 설명용 입력 = 마지막 timestep + static
    X_seq_last = X_seq[:, -1, :]
    X_input = np.concatenate([X_seq_last, X_static], axis=1)

    use_n = max(BACKGROUND_SIZE, EXPLAIN_SIZE)
    X_input = X_input[:use_n]

    model = load_model(seq_dim=X_seq.shape[2], static_dim=X_static.shape[1])
    predict_fn = make_predict_fn(
        model,
        seq_timesteps=X_seq.shape[1],
        seq_feature_dim=X_seq.shape[2],
    )

    background = X_input[:BACKGROUND_SIZE]
    explain_data = X_input[:EXPLAIN_SIZE]

    explainer = shap.Explainer(predict_fn, background, feature_names=feature_names)
    shap_values = explainer(explain_data)

    bar_path = os.path.join(SAVE_DIR, f"{output_prefix}_shap_bar.png")
    beeswarm_path = os.path.join(SAVE_DIR, f"{output_prefix}_shap_beeswarm.png")

    plt.figure()
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title(f"{output_prefix.upper()} SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title(f"{output_prefix.upper()} SHAP Beeswarm")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {bar_path}")
    print(f"Saved: {beeswarm_path}")


# =========================
# 메인
# =========================
def main():
    print("===== MIMIC SHAP =====")
    make_shap_plots(
        seq_path=MIMIC_SEQ_PATH,
        static_path=MIMIC_STATIC_PATH,
        output_prefix="mimic",
        static_columns_path=STATIC_COLUMNS_PATH,
    )

    print("\n===== eICU SHAP =====")
    make_shap_plots(
        seq_path=EICU_SEQ_PATH,
        static_path=EICU_STATIC_PATH,
        output_prefix="eicu",
        static_columns_path=STATIC_COLUMNS_PATH,
    )

    print("\nAll done.")
    print("Expected files:")
    print(f"- {SAVE_DIR}/mimic_shap_bar.png")
    print(f"- {SAVE_DIR}/mimic_shap_beeswarm.png")
    print(f"- {SAVE_DIR}/eicu_shap_bar.png")
    print(f"- {SAVE_DIR}/eicu_shap_beeswarm.png")


if __name__ == "__main__":
    main()