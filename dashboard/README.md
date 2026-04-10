# Dashboard

Streamlit 기반 pre 발표용 시각화 대시보드입니다.

## 주요 기능

- 모델 입력 데이터 확인
- 모델 출력 결과 확인
- 성능 지표 확인 (정확도 등)
- 환자/샘플별 결과 조회

## 폴더 구조

```
dashboard/
├── README.md
├── requirements.txt
└── app.py          # Streamlit 앱 진입점
```

## 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 앱 실행

```bash
streamlit run app.py
```

실행 후 브라우저에서 `http://localhost:8501` 접속

## 담당

- 김효경