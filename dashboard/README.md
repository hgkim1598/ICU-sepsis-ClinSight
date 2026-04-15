# Dashboard

Streamlit 기반 pre 발표용 시각화 대시보드입니다.  
백엔드는 FastAPI, 프론트엔드는 Streamlit으로 구성되어 있다.

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
pip install -r dashboard/requirements.txt
```

### 2. 서버 실행 (터미널 2개 필요)

> **중요:** FastAPI(백엔드)를 먼저 켜고, 그 다음 Streamlit(프론트)을 켜야 한다.  
> Streamlit이 FastAPI에 API 요청을 보내서 데이터를 가져오기 때문에,  
> 백엔드가 꺼져 있으면 대시보드가 정상 동작하지 않는다.

#### 터미널 1 — FastAPI 백엔드

```bash
# 프로젝트 루트에서 실행
source .venv/Scripts/activate    # Windows (Git Bash)
# source .venv/bin/activate      # Mac/Linux

uvicorn pipeline.api:app --reload --port 8000

# KMP_DUPLICATE_LIB_OK: torch + xgboost 동시 로드 시 OpenMP 라이브러리 충돌 방지
# Mac / Linux / Windows Git Bash — 전부 동일
KMP_DUPLICATE_LIB_OK=TRUE uvicorn pipeline.api:app --reload --host 127.0.0.1 --port 8000

# Windows PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"; uvicorn pipeline.api:app --reload --host 127.0.0.1 --port 8000

# Windows CMD
set KMP_DUPLICATE_LIB_OK=TRUE && uvicorn pipeline.api:app --reload --host 127.0.0.1 --port 8000
```

정상 실행되면 아래와 같은 로그가 나온다:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Started reloader process
```

- API 문서 확인: http://localhost:8000/docs

#### 터미널 2 — Streamlit 프론트엔드

```bash
# 새 터미널 열기 (터미널 1은 그대로 두기)
source .venv/Scripts/activate    # Windows (Git Bash)
# source .venv/bin/activate      # Mac/Linux

streamlit run dashboard/app.py
```

정상 실행되면 아래와 같은 로그가 나온다:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

- 대시보드 접속: http://localhost:8501

### 3. 종료

각 터미널에서 `Ctrl + C`로 종료합니다.  
Streamlit → FastAPI 순서로 끄는 것을 권장

## 담당

- 김효경