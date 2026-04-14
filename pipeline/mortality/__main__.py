import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from predict import predict_mortality

if __name__ == '__main__':
    intime       = datetime(2024, 1, 1, 8, 0)
    sepsis_onset = datetime(2024, 1, 1, 14, 0)

    patient_meta = {
        'age':                68,
        'gender':             1,
        'intime':             intime,
        'sepsis_onset_time':  sepsis_onset,
        'window_start_vital': max(sepsis_onset - timedelta(hours=6), intime),
        'window_start_lab':   sepsis_onset - timedelta(hours=6),
        'window_end':         sepsis_onset + timedelta(hours=42),
    }

    timestamps = pd.date_range(
        start=patient_meta['window_start_vital'],
        end=patient_meta['window_end'],
        freq='1h'
    )
    n = len(timestamps)
    np.random.seed(42)

    vital_ts = pd.DataFrame({
        'charttime':     timestamps,
        'heart_rate':    np.random.normal(95, 10, n).clip(60, 140),
        'mbp':           np.random.normal(65, 8, n).clip(45, 100),
        'sbp':           np.random.normal(105, 12, n).clip(70, 160),
        'dbp':           np.random.normal(60, 8, n).clip(40, 100),
        'resp_rate':     np.random.normal(22, 4, n).clip(10, 35),
        'spo2':          np.random.normal(94, 3, n).clip(80, 100),
        'temperature':   np.random.normal(38.2, 0.5, n).clip(36, 40),
        'gcs':           np.random.choice([13, 14, 15], n).astype(float),
        'pao2fio2ratio': np.random.normal(220, 50, n).clip(100, 400),
    })

    lab_times = sorted(np.random.choice(
        pd.date_range(patient_meta['window_start_lab'],
                      patient_meta['window_end'], freq='2h').tolist(),
        size=12, replace=False
    ))
    lab_df = pd.DataFrame({
        'charttime':       lab_times,
        'lactate':         np.random.normal(3.2, 1.0, 12).clip(0.5, 8.0),
        'creatinine':      np.random.normal(1.8, 0.5, 12).clip(0.5, 5.0),
        'bun':             np.random.normal(28, 8, 12).clip(5, 60),
        'sodium':          np.random.normal(138, 4, 12).clip(125, 150),
        'potassium':       np.random.normal(4.1, 0.5, 12).clip(3.0, 6.0),
        'glucose':         np.random.normal(145, 30, 12).clip(70, 300),
        'bicarbonate':     np.random.normal(20, 3, 12).clip(12, 30),
        'albumin':         np.random.normal(2.8, 0.4, 12).clip(1.5, 4.5),
        'wbc':             np.random.normal(14, 4, 12).clip(2, 30),
        'platelet':        np.random.normal(180, 60, 12).clip(50, 400),
        'hemoglobin':      np.random.normal(9.5, 1.5, 12).clip(6, 14),
        'bilirubin_total': np.random.normal(1.8, 0.8, 12).clip(0.3, 8.0),
    })

    result = predict_mortality(vital_ts, lab_df, patient_meta, patient_id='test-patient-001')
    print(json.dumps(result, indent=2, ensure_ascii=False))