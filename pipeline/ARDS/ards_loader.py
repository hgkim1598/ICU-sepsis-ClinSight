"""
ARDS 모델 아티팩트 로더.
S3 또는 로컬에서 joblib 아티팩트(dict: base_model / calibrator / features / threshold)를
한 번만 로드해 모듈 전역에 캐시한다.
"""

import sys, os
ARDS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ARDS_DIR)

import os
import joblib
import boto3
from io import BytesIO

from ards_config import S3_BUCKET, MODEL_PREFIX, USE_S3, LOCAL_MODEL_PATH, ARTIFACT_FILENAME


_artifact = None


def _load_artifact():
    if USE_S3:
        s3 = boto3.client('s3')
        key = f'{MODEL_PREFIX}/{ARTIFACT_FILENAME}'
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        artifact = joblib.load(BytesIO(obj['Body'].read()))
    else:
        path = os.path.join(LOCAL_MODEL_PATH, ARTIFACT_FILENAME)
        artifact = joblib.load(path)
    return artifact


def _get_artifact():
    global _artifact
    if _artifact is None:
        print("[ARDS] 모델 로드 중...")
        _artifact = _load_artifact()
        print(f"[ARDS] 모델 로드 완료 (피처 {len(_artifact['features'])}개)")
    return _artifact
