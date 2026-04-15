# ICU Sepsis Mortality Prediction Pipeline

패혈증 ICU 환자의 사망률을 예측하는 추론 파이프라인입니다.
BiLSTM(시계열) + XGBoost(정적 피처) → Stacking Logistic Regression 구조입니다.



## 모델 구조
vital_ts (시계열) → BiLSTM → prob_lstm
→ Stacking LR → 최종 사망 확률
lab_df   (정적)   → XGBoost → prob_xgb


### 실행 방법
pip install -r requirements.txt



## 빠른 실행 (더미 데이터로 테스트)

```bash
python predict.py
```

`predict.py` 하단에 아래 코드 추가하면 바로 테스트 가능합니다.

```python
if __name__ == '__main__':
    import numpy as np
    from datetime import datetime, timedelta

    intime       = datetime(2024, 1, 1, 8, 0)
    sepsis_onset = datetime(2024, 1, 1, 14, 0)

    patient_meta = {
        'age': 68,
        'gender': 1,
        'intime': intime,
        'sepsis_onset_time': sepsis_onset,
        'window_start_vital': max(sepsis_onset - timedelta(hours=6), intime),
        'window_start_lab':   sepsis_onset - timedelta(hours=6),
        'window_end':         sepsis_onset + timedelta(hours=42),
    }

    timestamps = pd.date_range(
        start=patient_meta['window_start_vital'],
        end=patient_meta['window_end'], freq='1h'
    )
    n = len(timestamps)
    np.random.seed(42)

    vital_ts = pd.DataFrame({
        'charttime':     timestamps,
        'heart_rate':    np.random.normal(95, 10, n).clip(60, 140),
        'mbp':           np.random.normal(65, 8, n).clip(45, 100),
        'sbp':           np.random.normal(105, 12, n).clip(70, 160),
        'dbp':           np.random.normal(60, 8, n).clip(40, 100),
        'resp_rate':     np.random.normal(22, 4, n).clip(10, 35),
        'spo2':          np.random.normal(94, 3, n).clip(80, 100),
        'temperature':   np.random.normal(38.2, 0.5, n).clip(36, 40),
        'gcs':           np.random.choice([13,14,15], n).astype(float),
        'pao2fio2ratio': np.random.normal(220, 50, n).clip(100, 400),
    })

    lab_times = sorted(np.random.choice(
        pd.date_range(patient_meta['window_start_lab'],
                      patient_meta['window_end'], freq='2h').tolist(),
        size=12, replace=False
    ))
    lab_df = pd.DataFrame({
        'charttime':       lab_times,
        'lactate':         np.random.normal(3.2, 1.0, 12).clip(0.5, 8.0),
        'creatinine':      np.random.normal(1.8, 0.5, 12).clip(0.5, 5.0),
        'bun':             np.random.normal(28, 8, 12).clip(5, 60),
        'sodium':          np.random.normal(138, 4, 12).clip(125, 150),
        'potassium':       np.random.normal(4.1, 0.5, 12).clip(3.0, 6.0),
        'glucose':         np.random.normal(145, 30, 12).clip(70, 300),
        'bicarbonate':     np.random.normal(20, 3, 12).clip(12, 30),
        'albumin':         np.random.normal(2.8, 0.4, 12).clip(1.5, 4.5),
        'wbc':             np.random.normal(14, 4, 12).clip(2, 30),
        'platelet':        np.random.normal(180, 60, 12).clip(50, 400),
        'hemoglobin':      np.random.normal(9.5, 1.5, 12).clip(6, 14),
        'bilirubin_total': np.random.normal(1.8, 0.8, 12).clip(0.3, 8.0),
    })

    import json
    result = predict_mortality(vital_ts, lab_df, patient_meta)
    print(json.dumps(result, indent=2))
```



## 모델 파일

모델 파일은 Git에 포함되지 않습니다. 두 가지 방식으로 사용 가능합니다.

**방식 A: S3에서 자동 로드 (기본값)**

AWS 자격증명이 설정된 환경에서 자동으로 S3에서 로드합니다.

```bash
export S3_BUCKET=say2-1team
export MODEL_PREFIX=Final_model/saved_models
export USE_S3=true
```

**방식 B: 로컬 모델 파일 사용**

아래 세 파일을 `./models/` 폴더에 위치시킵니다.
models/
├── bilstm_best.pt
├── xgb_stacking.json
└── stacking_lr.pkl

```bash
export USE_S3=false
export LOCAL_MODEL_PATH=./models
```


## 입력 형식

### vital_ts (pd.DataFrame)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| charttime | datetime | 측정 시각 |
| heart_rate | float | 심박수 (bpm) |
| mbp | float | 평균 동맥압 (mmHg) |
| sbp | float | 수축기 혈압 (mmHg) |
| dbp | float | 이완기 혈압 (mmHg) |
| resp_rate | float | 호흡수 (회/분) |
| spo2 | float | 산소포화도 (%) |
| temperature | float | 체온 (°C) |
| gcs | float | 글래스고 혼수 척도 (3-15) |
| pao2fio2ratio | float | P/F ratio |

### lab_df (pd.DataFrame)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| charttime | datetime | 검사 시각 |
| lactate | float | 젖산 (mmol/L) |
| creatinine | float | 크레아티닌 (mg/dL) |
| bun | float | 혈중 요소 질소 (mg/dL) |
| sodium | float | 나트륨 (mEq/L) |
| potassium | float | 칼륨 (mEq/L) |
| glucose | float | 혈당 (mg/dL) |
| bicarbonate | float | 중탄산염 (mEq/L) |
| albumin | float | 알부민 (g/dL) |
| wbc | float | 백혈구 (K/uL) |
| platelet | float | 혈소판 (K/uL) |
| hemoglobin | float | 헤모글로빈 (g/dL) |
| bilirubin_total | float | 총 빌리루빈 (mg/dL) |

### patient_meta (dict)

| 키 | 타입 | 설명 |
|----|------|------|
| age | int | 나이 |
| gender | int | 성별 (1=남, 0=여) |
| intime | datetime | ICU 입실 시각 |
| sepsis_onset_time | datetime | 패혈증 발생 시점 |
| window_start_vital | datetime | 활력징후 window 시작 (max(onset-6h, intime)) |
| window_start_lab | datetime | 검사 window 시작 (onset-6h) |
| window_end | datetime | window 종료 (onset+42h) |

## 사용법
`predict_mortality` 함수를 import해서 본인 데이터에 바로 적용할 수 있습니다.

```python
from predict import predict_mortality
import pandas as pd
from datetime import datetime, timedelta


# 환자 메타 정보
intime       = datetime(2024, 1, 1, 8, 0)
sepsis_onset = datetime(2024, 1, 1, 14, 0)

patient_meta = {
    'age': 68,
    'gender': 1,
    'intime': intime,
    'sepsis_onset_time': sepsis_onset,
    'window_start_vital': max(sepsis_onset - timedelta(hours=6), intime),
    'window_start_lab':   sepsis_onset - timedelta(hours=6),
    'window_end':         sepsis_onset + timedelta(hours=42),
}

# 활력징후 데이터 (charttime별 행)
vital_ts = pd.DataFrame({
    'charttime':     [...],
    'heart_rate':    [...],
    'mbp':           [...],
    # ...
})

# 검사 데이터
lab_df = pd.DataFrame({
    'charttime':  [...],
    'lactate':    [...],
    # ...
})

# 추론 실행
result = predict_mortality(vital_ts, lab_df, patient_meta)
print(result)
```

---

## 출력 형식

## 출력 형식

```json
{
  "mortality": {
    "probability": 0.852,
    "prediction": 1,
    "threshold": 0.21,
    "shap": [
      {"feature": "lactate_last", "value": 0.661, "unit": "mmol/L"},
      {"feature": "sodium_min",   "value": 0.749, "unit": "mEq/L"},
      {"feature": "bun_last",     "value": 0.312, "unit": "mg/dL"}
    ]
  }
}
```

- `probability`: 사망 확률 (0~1)
- `prediction`: 최종 분류 결과 (1=사망 고위험, 0=저위험)
- `threshold`: 분류 기준값 (0.21, F1 최적화 기준)
- `shap`: 정적 피처 63개의 SHAP 기여값
  - `feature`: 피처명
  - `value`: SHAP 값 (양수=사망 위험 증가, 음수=감소)
  - `unit`: 피처 단위

- `probability`: 사망 확률 (0~1)
- `shap`: 정적 피처 63개의 SHAP 기여값 (양수=사망 위험 증가, 음수=감소)

---

## 주의사항

- 입력 데이터의 `charttime`은 `window_start ~ window_end` 범위 내 데이터만 포함
- 결측값은 허용되며 내부적으로 forward fill + masking 처리
- 모델은 MIMIC-IV 기반으로 학습되었으며 외부 기관 적용 시 성능 차이 있을 수 있음




---
### 담당
박범진