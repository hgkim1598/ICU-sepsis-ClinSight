# ARDS (Acute Respiratory Distress Syndrome, 급성 호흡곤란 증후군 예측 모델)

ICU 환자의 급성 호흡곤란 증후군(ARDS) 발생 예측 모델입니다.

## 담당
- 김효경

## 폴더 구조
```
ards/
├── README.md
├── model.py        # 모델 구조 정의
├── train.py        # 모델 학습 및 .pt/.pkl 파일 생성
└── inference.py    # 저장된 모델 불러와서 추론
```

## 실행 방법

### 학습
```bash
python train.py
```

### 추론
```bash
python inference.py
```

## 출력 파일
- `train.py` 실행 시 모델 파일 생성 (`.pt` 또는 `.pkl`)
- 모델 파일은 git에 포함되지 않으므로 직접 학습 후 사용