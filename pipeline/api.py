
import json
import boto3
import pandas as pd
from io import BytesIO
from datetime import datetime
from fastapi import FastAPI, HTTPException
from mortality.predict import predict_mortality
from ARDS.ards_predict import predict_ards
from SIC.sic_predict import predict_sic
from mortality.predict import predict_mortality
from AKI.aki_predict import predict_aki





app = FastAPI(
    title="Sepsis ICU Mortality Prediction API",
    description="패혈증 ICU 환자 사망률 예측 파이프라인",
    version="1.0.0"
)

S3_BUCKET      = 'say2-1team'
PATIENT_PREFIX = 'pipeline/patients'


def _s3():
    return boto3.client('s3')


def _load_patient(patient_id: str):
    s3     = _s3()
    prefix = f'{PATIENT_PREFIX}/{patient_id}'

    try:
        # patient_meta
        obj  = s3.get_object(Bucket=S3_BUCKET, Key=f'{prefix}/patient_meta.json')
        meta = json.loads(obj['Body'].read())

        # datetime 복원
        for key in ['intime', 'sepsis_onset_time', 'window_start_vital', 'window_start_lab', 'window_end']:
            meta[key] = datetime.fromisoformat(meta[key])

        # vital_ts
        obj      = s3.get_object(Bucket=S3_BUCKET, Key=f'{prefix}/vital_ts.parquet')
        vital_ts = pd.read_parquet(BytesIO(obj['Body'].read()))

        # lab_df
        obj    = s3.get_object(Bucket=S3_BUCKET, Key=f'{prefix}/lab_df.parquet')
        lab_df = pd.read_parquet(BytesIO(obj['Body'].read()))

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"환자 데이터 없음: {patient_id} ({e})")

    return meta, vital_ts, lab_df


# ── 환자 목록 ─────────────────────────────────────────────────
@app.get("/patients")
def list_patients():
    s3  = _s3()
    res = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f'{PATIENT_PREFIX}/', Delimiter='/')
    ids = [
        p['Prefix'].split('/')[-2]
        for p in res.get('CommonPrefixes', [])
    ]
    return {"patients": ids}


# ── Raw 데이터 조회 (프론트용) ────────────────────────────────
@app.get("/patients/{patient_id}/data")
def get_patient_data(patient_id: str):
    meta, vital_ts, lab_df = _load_patient(patient_id)
    return {
        "patient_id":   patient_id,
        "patient_meta": {k: str(v) for k, v in meta.items()},
        "vital_ts":     vital_ts.assign(charttime=vital_ts['charttime'].astype(str)).to_dict(orient='records'),
        "lab_df":       lab_df.assign(charttime=lab_df['charttime'].astype(str)).to_dict(orient='records'),
    }


# ── 추론 실행 ─────────────────────────────────────────────────
@app.post("/predict/{patient_id}")
def predict(patient_id: str):
    meta, vital_ts, lab_df = _load_patient(patient_id)

    mortality_result = predict_mortality(vital_ts, lab_df, meta, patient_id=patient_id)
    ards_result      = predict_ards(vital_ts, lab_df, meta)
    sic_result       = predict_sic(vital_ts, lab_df, meta)
    aki_result       = predict_aki(vital_ts, lab_df, meta)

    return {
        **mortality_result,
        **ards_result,
        **sic_result,
        **aki_result,
    }