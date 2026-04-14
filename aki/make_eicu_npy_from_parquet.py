import numpy as np
import pandas as pd
import os

SEQ_PATH = "eicu_raw_data/eicu_derived/eicu_timeseries_processed.parquet"
STATIC_PATH = "eicu_raw_data/eicu_derived/eicu_static_processed_with_aki_label.parquet"

OUTPUT_DIR = "eicu_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ID_COL = "stay_id"
TIME_COL = "hour_bin"

SEQ_FEATURES = [
    "heartrate",
    "map",
    "sbp",
    "dbp",
    "rr",
    "spo2",
    "temperature",
    "gcs",
    "pf_ratio",
]

MASK_FEATURES = [
    "heartrate",
    "map",
    "sbp",
    "dbp",
    "rr",
]

FINAL_SEQ_FEATURES = SEQ_FEATURES + [f"{f}_mask" for f in MASK_FEATURES]

STATIC_FEATURES = [
    "age",
    "gender",
    "lactate_slope",
    "creatinine_slope",
    "bun_slope",
    "urine_diff",              # urine_trend 대신
    "urine_min",
    "bilirubin_min",           # bilirubin_last 대신 대체
    "bilirubin_missing_flag",  # bilirubin_missing 대신
    "sodium_last",
    "sodium_last",             # sodium_trend 대체용 임시
    "sodium_min",
    "potassium_last",
    "potassium_last",          # potassium_trend 대체용 임시
    "potassium_min",
    "glucose_last",
    "glucose_last",            # glucose_trend 대체용 임시
    "glucose_min",
    "bicarbonate_last",
    "bicarbonate_last",        # bicarbonate_trend 대체용 임시
    "bicarbonate_min",
    "wbc_slope",
    "platelet_slope",
    "hemoglobin_diff",         # hemoglobin_trend 대신
]

N_TIMESTEPS = 12

seq_df = pd.read_parquet(SEQ_PATH)
static_df = pd.read_parquet(STATIC_PATH)

print("Loaded:", seq_df.shape, static_df.shape)

patients = sorted(seq_df[ID_COL].unique())
X_seq_list = []

for pid in patients:
    p = seq_df[seq_df[ID_COL] == pid].copy()

    p[TIME_COL] = p[TIME_COL].astype(int)
    p = p[(p[TIME_COL] >= 0) & (p[TIME_COL] < N_TIMESTEPS)]

    base = pd.DataFrame({TIME_COL: range(N_TIMESTEPS)})
    base[ID_COL] = pid

    p = base.merge(p, on=[ID_COL, TIME_COL], how="left")

    for col in MASK_FEATURES:
        p[f"{col}_mask"] = p[col].notna().astype(float)

    p[SEQ_FEATURES] = p[SEQ_FEATURES].ffill()
    p[SEQ_FEATURES] = p[SEQ_FEATURES].fillna(0)

    for col in [f"{f}_mask" for f in MASK_FEATURES]:
        p[col] = p[col].fillna(0)

    arr = p[FINAL_SEQ_FEATURES].values.astype(np.float32)
    X_seq_list.append(arr)

X_seq = np.stack(X_seq_list)

static_df = static_df.drop_duplicates(ID_COL)
static_df = static_df.set_index(ID_COL).reindex(patients).reset_index()

# gender 같은 문자열 컬럼 포함 가능해서 feature별로 개별 처리
for col in STATIC_FEATURES:
    static_df[col] = pd.to_numeric(static_df[col], errors="coerce")

for col in STATIC_FEATURES:
    if static_df[col].isna().all():
        static_df[col] = 0.0
    else:
        static_df[col] = static_df[col].fillna(static_df[col].median())

static_df["aki_label"] = pd.to_numeric(static_df["aki_label"], errors="coerce").fillna(0)

X_static = static_df[STATIC_FEATURES].to_numpy(dtype=np.float32)
y = static_df["aki_label"].to_numpy(dtype=np.float32)

np.save(f"{OUTPUT_DIR}/eicu_team_X_seq_with_mask.npy", X_seq)
np.save(f"{OUTPUT_DIR}/eicu_team_X_static.npy", X_static)
np.save(f"{OUTPUT_DIR}/eicu_team_y.npy", y)

print("\nSaved:")
print("X_seq:", X_seq.shape)
print("X_static:", X_static.shape)
print("y:", y.shape)