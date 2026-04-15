# AKI (Acute Kidney Injury, 급성 신손상 예측 모델)

ICU 환자의 AKI 발생 예측 모델입니다.

## 담당
- 이민경

## 구성
- GRU 모델: `gru_v_final.h5`
- XGBoost 모델: `xgb_v8_final.pkl`
- 최종 예측: GRU와 XGBoost 확률을 0.5:0.5로 평균한 soft voting ensemble

## 입력 데이터
- `aki_seq_X_v7.npy`

## 실행 방법

### 1. 로컬 모델 파일로 실행
```bash
python predict.py --input_path aki_seq_X_v7.npy