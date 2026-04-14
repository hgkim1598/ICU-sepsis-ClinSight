import boto3
import joblib
import torch
import xgboost as xgb
from io import BytesIO

from config import S3_BUCKET, MODEL_PREFIX, USE_S3, LOCAL_MODEL_PATH
from model import BiLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_bilstm, _clf_xgb, _lr = None, None, None


def _load_models():
    if USE_S3:
        s3 = boto3.client('s3')

        obj = s3.get_object(Bucket=S3_BUCKET, Key=f'{MODEL_PREFIX}/bilstm_best.pt')
        state = torch.load(BytesIO(obj['Body'].read()), map_location=device)

        obj = s3.get_object(Bucket=S3_BUCKET, Key=f'{MODEL_PREFIX}/xgb_stacking.json')
        with open('/tmp/xgb_stacking.json', 'wb') as f:
            f.write(obj['Body'].read())

        obj = s3.get_object(Bucket=S3_BUCKET, Key=f'{MODEL_PREFIX}/OOF/stacking_lr_oof.pkl')
        with open('/tmp/stacking_lr_oof.pkl', 'wb') as f:
            f.write(obj['Body'].read())
    else:
        import shutil
        state = torch.load(f'{LOCAL_MODEL_PATH}/bilstm_best.pt', map_location=device)
        shutil.copy(f'{LOCAL_MODEL_PATH}/xgb_stacking.json',       '/tmp/xgb_stacking.json')
        shutil.copy(f'{LOCAL_MODEL_PATH}/OOF/stacking_lr_oof.pkl', '/tmp/stacking_lr_oof.pkl')

    bilstm = BiLSTM().to(device)
    bilstm.load_state_dict(state)
    bilstm.eval()

    clf_xgb = xgb.XGBClassifier()
    clf_xgb.load_model('/tmp/xgb_stacking.json')

    lr = joblib.load('/tmp/stacking_lr_oof.pkl')

    return bilstm, clf_xgb, lr


def get_models():
    global _bilstm, _clf_xgb, _lr
    if _bilstm is None:
        print("모델 로드 중...")
        _bilstm, _clf_xgb, _lr = _load_models()
        print("모델 로드 완료")
    return _bilstm, _clf_xgb, _lr