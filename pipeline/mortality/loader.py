import os
import tempfile
import boto3
import joblib
import torch
import xgboost as xgb
from io import BytesIO

from config import S3_BUCKET, MODEL_PREFIX, USE_S3, LOCAL_MODEL_PATH
from model import BiLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tmp = tempfile.gettempdir()

_bilstm, _clf_xgb, _lr = None, None, None


def _load_models():
    if USE_S3:
        s3 = boto3.client('s3')

        obj = s3.get_object(Bucket=S3_BUCKET, Key=f'{MODEL_PREFIX}/mortality_bilstm.pt')
        state = torch.load(BytesIO(obj['Body'].read()), map_location=device)

        obj = s3.get_object(Bucket=S3_BUCKET, Key=f'{MODEL_PREFIX}/mortality_xgb.json')
        with open(os.path.join(tmp, 'mortality_xgb.json'), 'wb') as f:
            f.write(obj['Body'].read())

        obj = s3.get_object(Bucket=S3_BUCKET, Key=f'{MODEL_PREFIX}/mortality_stacking_lr.pkl')
        with open(os.path.join(tmp, 'mortality_stacking_lr.pkl'), 'wb') as f:
            f.write(obj['Body'].read())
    else:
        import shutil
        state = torch.load(os.path.join(LOCAL_MODEL_PATH, 'mortality_bilstm.pt'), map_location=device)
        shutil.copy(os.path.join(LOCAL_MODEL_PATH, 'mortality_xgb.json'),        os.path.join(tmp, 'mortality_xgb.json'))
        shutil.copy(os.path.join(LOCAL_MODEL_PATH, 'mortality_stacking_lr.pkl'), os.path.join(tmp, 'mortality_stacking_lr.pkl'))

    bilstm = BiLSTM().to(device)
    bilstm.load_state_dict(state)
    bilstm.eval()

    clf_xgb = xgb.XGBClassifier()
    clf_xgb.load_model(os.path.join(tmp, 'mortality_xgb.json'))

    lr = joblib.load(os.path.join(tmp, 'mortality_stacking_lr.pkl'))

    return bilstm, clf_xgb, lr


def get_models():
    global _bilstm, _clf_xgb, _lr
    if _bilstm is None:
        print("모델 로드 중...")
        _bilstm, _clf_xgb, _lr = _load_models()
        print("모델 로드 완료")
    return _bilstm, _clf_xgb, _lr
