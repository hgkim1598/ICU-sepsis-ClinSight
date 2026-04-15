"""
ARDS 예측 모델 정의
- XGBoost + Platt Scaling (CalibratedClassifierCV)
- conservative feature track (43 features)
"""
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN_ESTIMATOR = True
except Exception:
    FrozenEstimator = None
    HAS_FROZEN_ESTIMATOR = False


def build_xgb_model(scale_pos_weight=1.0):
    """학습용 XGBClassifier 생성 (하이퍼파라미터 고정)"""
    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=30,
        verbosity=0,
    )


def calibrate_model(model, X_val, y_val):
    """Platt Scaling으로 확률 캘리브레이션"""
    if HAS_FROZEN_ESTIMATOR:
        calibrator = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    else:
        calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrator.fit(X_val, y_val)
    return calibrator
