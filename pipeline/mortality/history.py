import json
import boto3
from io import BytesIO

from config import S3_BUCKET, HISTORY_PREFIX


def _s3():
    return boto3.client('s3')


def _latest_key(patient_id: str) -> str:
    return f'{HISTORY_PREFIX}/{patient_id}/latest.json'


def load_latest(patient_id: str) -> dict | None:
    """S3에서 환자의 직전 추론 결과를 로드. 없으면 None 반환."""
    try:
        obj = _s3().get_object(Bucket=S3_BUCKET, Key=_latest_key(patient_id))
        return json.loads(obj['Body'].read())
    except Exception:
        return None


def save_result(patient_id: str, result: dict) -> None:
    """result를 latest.json에 덮어쓰기 저장."""
    body = json.dumps(result, ensure_ascii=False).encode('utf-8')
    _s3().put_object(
        Bucket=S3_BUCKET,
        Key=_latest_key(patient_id),
        Body=body,
        ContentType='application/json'
    )


def compute_changes(current_feats: dict, previous_result: dict | None) -> dict:
    """
    현재 feats와 이전 feature_values를 비교해 change/change_direction 계산.
    Returns: {feature_name: {'change': float|None, 'change_direction': str}}
    """
    if previous_result is None:
        return {}

    prev_map = {
        item['feature']: item.get('raw_value')
        for item in previous_result.get('mortality', {}).get('feature_values', [])
    }

    changes = {}
    for feat, cur_val in current_feats.items():
        prev_val = prev_map.get(feat)
        if prev_val is None or cur_val is None:
            changes[feat] = {'change': None, 'change_direction': 'unknown'}
            continue

        diff = round(float(cur_val) - float(prev_val), 4)
        if diff > 0:
            direction = 'up'
        elif diff < 0:
            direction = 'down'
        else:
            direction = 'equal'

        changes[feat] = {'change': diff, 'change_direction': direction}

    return changes