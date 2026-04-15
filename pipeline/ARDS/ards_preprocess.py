"""
ARDS 추론용 전처리.
환자 1명의 raw vital_df + lab_df + patient_meta → (1, 43) ndarray.
XGBoost는 NaN을 자체 처리하므로 별도 imputation 없음 (의도된 설계).
"""
import numpy as np
import pandas as pd
import sys, os
ARDS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ARDS_DIR)
from ards_config import STAT_RULES, COL_MAP, SOURCE_MAP, FEAT_COLS, BG_COLS, WINDOW_H


# ── 헬퍼 함수 ─────────────────────────────────────────────────
def _extract_stats(values, stat_list):
    """시계열 값 배열에서 summary statistics를 추출한다."""
    out = {stat: np.nan for stat in stat_list}
    vals = pd.Series(values).dropna()

    if len(vals) == 0:
        if "missing" in stat_list:
            out["missing"] = 1.0
        return out

    if "last" in stat_list:
        out["last"] = float(vals.iloc[-1])
    if "mean" in stat_list:
        out["mean"] = float(vals.mean())
    if "trend" in stat_list:
        out["trend"] = float(vals.iloc[-1] - vals.iloc[0]) if len(vals) > 1 else 0.0
    if "min" in stat_list:
        out["min"] = float(vals.min())
    if "max" in stat_list:
        out["max"] = float(vals.max())
    if "missing" in stat_list:
        out["missing"] = 0.0

    return out


def _resolve_onset(patient_meta):
    """patient_meta에서 onset 시점을 가져온다 (키 이름 호환)."""
    for key in ['onset_time', 'sepsis_onset_time']:
        if key in patient_meta and patient_meta[key] is not None:
            return pd.Timestamp(patient_meta[key])
    raise KeyError("patient_meta에 'onset_time' 또는 'sepsis_onset_time' 키가 필요합니다.")


def _resolve_gender(patient_meta):
    """patient_meta에서 성별 값을 가져온다 (키 이름 호환)."""
    # gender_bin(0=F, 1=M) 우선, 없으면 gender 사용
    if 'gender_bin' in patient_meta:
        return patient_meta['gender_bin']
    if 'gender' in patient_meta:
        return patient_meta['gender']
    return np.nan


def _split_bg_from_lab(lab_df):
    """
    lab_df에서 bg(혈액가스) 컬럼을 분리한다.
    파이프라인에서 vital_df + lab_df 2개만 넘겨줘도 내부에서 처리.

    Returns: (bg_df, pure_lab_df)
    """
    bg_cols_present = [c for c in BG_COLS if c in lab_df.columns]
    lab_only_cols = [c for c in lab_df.columns if c not in BG_COLS or c == 'charttime']

    if bg_cols_present:
        bg_df = lab_df[['charttime'] + bg_cols_present].copy()
    else:
        bg_df = pd.DataFrame(columns=['charttime'])

    pure_lab_df = lab_df[lab_only_cols].copy()
    return bg_df, pure_lab_df


# ── 전처리 메인 ───────────────────────────────────────────────
def preprocess(vital_df, lab_df, patient_meta):
    """
    환자 1명의 원시 데이터를 모델 입력 벡터(1×43)로 변환한다.
    lab_df 안에 bg 컬럼(lactate, ph, bicarbonate)이 있으면 자동 분리한다.

    Parameters
    ----------
    vital_df : pd.DataFrame
        컬럼: charttime, spo2, resp_rate, heart_rate, mbp, sbp, temperature
    lab_df : pd.DataFrame
        컬럼: charttime, creatinine, bun, wbc, platelet
        (+ 선택적으로 lactate, ph, bicarbonate 포함 가능)
    patient_meta : dict
        필수 keys: age, onset_time(또는 sepsis_onset_time)
        성별: gender_bin(또는 gender)

    Returns
    -------
    np.ndarray : shape (1, 43)
    """
    onset = _resolve_onset(patient_meta)
    win_start = onset - pd.Timedelta(hours=WINDOW_H)
    gender_val = _resolve_gender(patient_meta)

    # lab_df에서 bg 컬럼 자동 분리
    bg_df, pure_lab_df = _split_bg_from_lab(lab_df)

    # 윈도우 내 데이터 필터링
    source_dfs = {
        "vital": vital_df[(vital_df['charttime'] >= win_start) & (vital_df['charttime'] < onset)],
        "bg":    bg_df[(bg_df['charttime'] >= win_start) & (bg_df['charttime'] < onset)] if len(bg_df) > 0 else bg_df,
        "lab":   pure_lab_df[(pure_lab_df['charttime'] >= win_start) & (pure_lab_df['charttime'] < onset)],
    }

    feats = {
        "age": patient_meta.get('age', np.nan),
        "gender_bin": gender_val,
    }

    for feat_name, stat_list in STAT_RULES.items():
        source_key = SOURCE_MAP[feat_name]
        source_col = COL_MAP[feat_name]
        df = source_dfs[source_key]

        if source_col in df.columns:
            values = df.sort_values('charttime')[source_col].values
        else:
            values = np.array([])

        stats = _extract_stats(values, stat_list)
        for stat_name, val in stats.items():
            feats[f"{feat_name}_{stat_name}"] = val

    # XGBoost는 NaN을 자체 처리하므로 별도 imputation 불필요
    x = np.array([feats.get(c, np.nan) for c in FEAT_COLS], dtype=np.float32).reshape(1, -1)
    return x
