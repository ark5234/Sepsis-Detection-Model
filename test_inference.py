import pandas as pd
from app.ml.model_service import SepsisModelService
from pathlib import Path
import numpy as np

service = SepsisModelService(Path("artifacts"))

payload = {
    "measurements": [{
        "hr": 72.0,
        "o2sat": 98.0,
        "temp": 37.0,
        "sbp": 120.0,
        "map": 90.0,
        "dbp": 80.0,
        "resp": 16.0,
        "wbc": 7.0,
        "age": 23.0,
    }]
}

# 1. Test manual prediction
result = service.predict_manual(payload)
print("PREDICTION RESULT:", result)

# 2. Inspect tensors
from app.ml.model_service import _build_single_patient_tensors

df = pd.DataFrame([{
    "Patient_ID": "manual-entry",
    "Hour": 0,
    "HR": 72.0, "O2Sat": 98.0, "Temp": 37.0, "SBP": 120.0, "MAP": 90.0, "DBP": 80.0, "Resp": 16.0, "WBC": 7.0, "Age": 23.0
}])

v, vr, l, d, m, seq_len = _build_single_patient_tensors(df, service.vital_scaler, service.lab_scaler)

print("\n--- VITAL SCALED (first 2 hours) ---")
print(v[:2])
print("\n--- LAB SCALED (first 2 hours) ---")
print(l[:2])
print("\n--- MEASURED (first 2 hours) ---")
print(m[:2])

