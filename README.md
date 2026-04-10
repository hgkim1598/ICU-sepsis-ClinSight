# ICU-sepsis-ClinSight

ICU 패혈증 환자의 사망률 및 합병증 예측 AI 모델

---

## 프로젝트 개요

ICU에 입원한 패혈증 환자를 대상으로 사망률과 주요 합병증(AKI, ARDS, SIC) 발생 여부를 예측하는 AI 모델을 개발하고, Streamlit 기반 대시보드를 통해 결과를 시각화합니다.

## 예측 대상

| 모델 | 설명 |
|---|---|
| Mortality | 사망률 예측 |
| AKI | 급성 신손상 (Acute Kidney Injury) |
| ARDS | 급성 호흡곤란 증후군 (Acute Respiratory Distress Syndrome) |
| SIC | 패혈증 유발 응고병증 (Sepsis-Induced Coagulopathy) |

## 데이터셋

- MIMIC-IV
- 데이터 파일은 git에 포함되지 않습니다. `data/README.md` 참고

## 폴더 구조

```
ICU-sepsis-ClinSight/
├── README.md
├── .gitignore
│
├── pipeline/                   # 모델 학습 및 추론 파이프라인
│   ├── README.md
│   ├── requirements.txt
│   ├── utils.py                # 공통 함수
│   ├── mortality/              # 사망률 예측 모델
│   ├── aki/                    # AKI 예측 모델
│   ├── ards/                   # ARDS 예측 모델
│   └── sic/                    # SIC 예측 모델
│
├── dashboard/                  # Streamlit 대시보드
│   ├── README.md
│   ├── requirements.txt
│   └── app.py
│
└── data/                       # 샘플 데이터만 포함 (실제 데이터 제외)
    ├── README.md
    └── sample/
        └── .gitkeep
```

## 팀원

| 이름 | 역할 |
|---|---|
| 이천기 | 팀장 / SIC 모델 |
| 박범진 | Mortality 모델 / 파이프라인 |
| 김효경 | ARDS 모델 / 대시보드 (Streamlit) |
| 이민경 | AKI  모델 |
| 최지우 | 대시보드 (Streamlit) |

## 실행 방법

### 1. 패키지 설치

```bash
pip install -r pipeline/requirements.txt
pip install -r dashboard/requirements.txt
```

### 2. 모델 학습

```bash
cd pipeline
python mortality/train.py
python aki/train.py
python ards/train.py
python sic/train.py
```

### 3. 대시보드 실행

```bash
cd dashboard
streamlit run app.py
```

실행 후 브라우저에서 `http://localhost:8501` 접속