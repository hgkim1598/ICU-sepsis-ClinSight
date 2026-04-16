# ICU Sepsis SIC Prediction Pipeline

패혈증 유발 응고장애(SIC, Sepsis-Induced Coagulopathy) 발생 확률을 예측하는 추론 파이프라인입니다.
BiLSTM(시계열) + XGBoost(집계 피처) → Stacking Logistic Regression 구조입니다.



## 모델 구조

```
X_seq  (시계열 48h × 42 features) → BiLSTM  → p_lstm ┐
                                                         → Stacking LR → 최종 SIC 확률
X_xgb  (정적 9 + 집계 170 features) → XGBoost → p_xgb ┘
```

- **BiLSTM**: hidden 128 × 2 layers × bidirectional, 5-fold OOF 앙상블
- **XGBoost**: n_estimators 1000, max_depth 6, 5-fold OOF 앙상블
- **Meta**: Logistic Regression (class_weight=balanced)



## 파일 구성

| 파일 | 설명 |
|------|------|
| `model.py` | LSTMClassifier, build_lstm(), build_xgb() 아키텍처 정의 |
| `train.py` | 5-fold OOF 학습 → checkpoints/ 생성 |
| `predict.py` | S3/로컬 모델 로드 → 대시보드 연동 추론 |



## 설치

```bash
pip install torch xgboost scikit-learn shap pandas numpy boto3
```



## 학습

```bash
python train.py \
  --data-dir  ./data/dataset \
  --output-dir ./checkpoints
```

### 데이터 디렉터리 구조

```
data/dataset/
├── dl/
│   ├── train.pkl       # X_seq (N, 48, 42), y, stay_ids
│   └── val.pkl
└── xgb/
    ├── train.parquet   # 정적 + 집계 피처 (N, 179+)
    └── val.parquet
```

### 출력 체크포인트 구조

```
checkpoints/
├── fold_1/
│   ├── best_model.pt         BiLSTM 가중치
│   ├── model.json            XGBoost 가중치
│   ├── feature_names.json    XGB 피처명
│   └── training_history.json LSTM 학습 이력
├── fold_2/ ... fold_5/
└── meta/
    └── meta_model.pkl        Stacking LR
```

> **모델 파일은 Git에 포함되지 않습니다.** 학습 후 생성하거나 S3에서 로드하세요.



## 추론 (대시보드 연동)

```python
from predict import predict_sic
import pandas as pd
from datetime import datetime, timedelta

onset = datetime(2024, 1, 1, 14, 0)

patient_meta = {
    'age':               65,
    'sex_male':          1,
    'onset_time':        onset,
    'flag_coagulopathy': 1,
    'flag_ckd':          0,
    # 나머지 flag_* 생략 시 0으로 처리
}

vital_df = pd.DataFrame({
    'charttime': pd.date_range(start=onset - timedelta(hours=47), periods=48, freq='1h'),
    'map':       [70.0] * 48,
    'pf_ratio':  [250.0] * 48,
})

lab_df = pd.DataFrame({
    'charttime':       [onset - timedelta(hours=12), onset - timedelta(hours=6)],
    'lactate':         [3.2, 4.1],
    'creatinine':      [1.8, 2.1],
    'bilirubin_total': [1.5, 1.8],
    'wbc':             [14.0, 16.0],
    'rdw':             [15.2, 15.5],
    'aptt':            [42.0, 48.0],
})

result = predict_sic(vital_df, lab_df, patient_meta)
print(result)
# {"sic": {"probability": 0.412, "shap": [...]}}
```



## 모델 파일 로드 방식

### 방식 A: S3에서 자동 로드 (기본값)

```bash
export S3_BUCKET=say2-1team
export MODEL_PREFIX_SIC=Final_model/saved_models/sic
export USE_S3=true
```

S3 경로 구조:
```
Final_model/saved_models/sic/
├── fold_1~5/  best_model.pt, model.json, feature_names.json
├── meta/      meta_model.pkl
└── scaler.pkl
```

### 방식 B: 로컬 파일 사용

```bash
export USE_S3=false
export LOCAL_MODEL_PATH_SIC=./models/sic
```

로컬 파일 준비 시 `predict.py`의 `prepare_artifacts_for_deploy()` 헬퍼 사용:

```python
from predict import prepare_artifacts_for_deploy

prepare_artifacts_for_deploy(
    checkpoint_dir  = "./checkpoints",
    scaler_pkl_path = "./data/dataset/scaler.pkl",
    output_dir      = "./models/sic",
)
```



## 입력 형식

### vital_df

| 컬럼 | 타입 | 설명 |
|------|------|------|
| charttime | datetime | 측정 시각 |
| map | float | 평균동맥압 (mmHg) |
| pf_ratio | float | PaO2/FiO2 ratio |

### lab_df

| 컬럼 | 타입 | 설명 |
|------|------|------|
| charttime | datetime | 검사 시각 |
| lactate | float | 젖산 (mmol/L) |
| creatinine | float | 크레아티닌 (mg/dL) |
| bilirubin_total | float | 총 빌리루빈 (mg/dL) |
| wbc | float | 백혈구 (K/uL) |
| rdw | float | 적혈구 분포 폭 (%) |
| aptt | float | 활성화 부분 트롬보플라스틴 시간 (sec) |

### patient_meta

| 키 | 타입 | 설명 |
|----|------|------|
| age | int | 나이 |
| sex_male | int | 성별 (1=남, 0=여) |
| onset_time | datetime | 패혈증 발생 시점 (`sepsis_onset_time`도 허용) |
| flag_liver_failure | int | 간부전 여부 (선택, 기본 0) |
| flag_ckd | int | 만성 신장 질환 (선택, 기본 0) |
| flag_coagulopathy | int | 응고장애 과거력 (선택, 기본 0) |
| flag_diabetes | int | 당뇨 (선택, 기본 0) |
| flag_immunosuppression | int | 면역 억제 (선택, 기본 0) |
| flag_chf | int | 울혈성 심부전 (선택, 기본 0) |
| flag_septic_shock_hx | int | 패혈성 쇼크 과거력 (선택, 기본 0) |



## 출력 형식

```json
{
  "sic": {
    "probability": 0.412,
    "shap": [
      {"feature": "lactate_max_mean", "value": 0.0821},
      {"feature": "aptt_max_mean",    "value": 0.0634}
    ]
  }
}
```

- `probability`: SIC 발생 확률 (0~1)
- `shap`: XGB fold_1 기준 피처 기여값 (절댓값 내림차순 정렬)



## 주의사항

- 입력 데이터는 결측값을 허용하며 내부적으로 forward fill 처리
- 모델은 MIMIC-IV 기반으로 학습되었으며 외부 기관 적용 시 성능 차이 있을 수 있음
- 시계열은 onset 기준 이전 48시간 윈도우 사용



---
### 담당
이천기 (SIC 파트)
