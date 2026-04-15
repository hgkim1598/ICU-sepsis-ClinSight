import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import boto3
import joblib
import numpy as np
import tempfile
import tensorflow as tf

from aki_config import S3_BUCKET, MODEL_PREFIX, USE_S3, LOCAL_MODEL_PATH

tmp = tempfile.gettempdir()

_gru_model = None
_xgb_model = None


def _load_models():
    if USE_S3:
        s3 = boto3.client('s3')

        # GRU (.h5)
        gru_path = os.path.join(tmp, 'aki_gru.h5')
        obj = s3.get_object(Bucket=S3_BUCKET, Key=f'{MODEL_PREFIX}/aki_gru_final.h5')
        with open(gru_path, 'wb') as f:
            f.write(obj['Body'].read())

        # XGBoost (.pkl)
        xgb_path = os.path.join(tmp, 'aki_xgb.pkl')
        obj = s3.get_object(Bucket=S3_BUCKET, Key=f'{MODEL_PREFIX}/aki_xgb_final.pkl')
        with open(xgb_path, 'wb') as f:
            f.write(obj['Body'].read())
    else:
        gru_path = os.path.join(LOCAL_MODEL_PATH, 'aki_gru_final.h5')
        xgb_path = os.path.join(LOCAL_MODEL_PATH, 'aki_xgb_final.pkl')

    gru_model = tf.keras.models.load_model(gru_path)
    xgb_model = joblib.load(xgb_path)

    return gru_model, xgb_model


def get_models():
    global _gru_model, _xgb_model
    if _gru_model is None:
        print("AKI 모델 로드 중...")
        _gru_model, _xgb_model = _load_models()
        print("AKI 모델 로드 완료")
    return _gru_model, _xgb_model