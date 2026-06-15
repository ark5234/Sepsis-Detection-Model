"""
DPCT Model Service — loads trained PyTorch weights and runs inference.

The webapp is inference-only: training the Dual-Path Clinical Transformer
requires a CUDA GPU and takes ~30 minutes. Users train on Kaggle/Colab and
drop the saved weights file into artifacts/dpct_model_weights.pth.

Weights file expectation
------------------------
The file must be a dict saved via torch.save() containing:
    {
        "model_state_dict": <OrderedDict>,   # DualPathClinicalTransformer.state_dict()
        "vital_scaler":     <StandardScaler>,
        "lab_scaler":       <StandardScaler>,
        "threshold":        float,           # optimal F1 threshold (e.g. 0.87)
        "metadata":         dict,            # optional: metrics, trained_at, etc.
    }

If the weights file does not exist the service exposes a "not ready" status
and returns informative errors on all prediction endpoints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .dpct_model import (
    DualPathClinicalTransformer,
    VITAL_COLS,
    LAB_COLS,
    MAX_SEQ_LEN,
    N_VITALS,
    N_LABS,
)

# Optional torch import — graceful error if not installed
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Column config (must match training notebook) ───────────────────────────────
PATIENT_COL = "Patient_ID"
HOUR_COL    = "Hour"
TARGET_COL  = "SepsisLabel"

WEIGHTS_FILENAME = "dpct_model_weights.pth"

# Default threshold if weights file has no threshold key
DEFAULT_THRESHOLD = 0.87


@dataclass
class PredictionResult:
    patient_id: str
    probability: float
    risk_band: str
    predicted_label: int
    flags: List[str]
    attention_peak_hour: Optional[int]


# ══════════════════════════════════════════════════════════════════════════════
#  Preprocessing helpers  (mirrors Kaggle Section 4 logic)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_time_since_last_measurement(lab_array: np.ndarray) -> np.ndarray:
    T, F = lab_array.shape
    delta        = np.zeros((T, F), dtype=np.float32)
    last_measured = np.full(F, -1, dtype=np.float32)
    for t in range(T):
        for j in range(F):
            if not np.isnan(lab_array[t, j]):
                last_measured[j] = t
            delta[t, j] = float(t) if last_measured[j] < 0 else float(t - last_measured[j])
    return delta


def _pad_to(arr: np.ndarray, max_len: int) -> np.ndarray:
    T, F = arr.shape
    if T < max_len:
        arr = np.concatenate([arr, np.zeros((max_len - T, F), dtype=arr.dtype)], axis=0)
    return arr


def _build_single_patient_tensors(
    pat_df: pd.DataFrame,
    vital_scaler: StandardScaler,
    lab_scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Convert one patient's raw DataFrame into the five arrays the DPCT expects:
        vitals, vitals_raw, labs, lab_delta, lab_measured
    Returns arrays padded to MAX_SEQ_LEN plus the actual sequence length.
    """
    pat_df = pat_df.sort_values(HOUR_COL).reset_index(drop=True)
    T_actual = min(len(pat_df), MAX_SEQ_LEN)

    # ── Vital signs ──────────────────────────────────────────────────────────
    v_cols_present = [c for c in VITAL_COLS if c in pat_df.columns]
    v_raw = np.tile(vital_scaler.mean_, (T_actual, 1)).astype(np.float64)
    for j, col in enumerate(VITAL_COLS):
        if col in pat_df.columns:
            vals = pat_df[col].values[:T_actual].astype(np.float64)
            # forward-fill
            for t in range(1, T_actual):
                if np.isnan(vals[t]) and not np.isnan(vals[t - 1]):
                    vals[t] = vals[t - 1]
            mask = ~np.isnan(vals)
            v_raw[mask, j] = vals[mask]

    vitals_raw = v_raw.copy().astype(np.float32)          # unscaled (for clinical gate)
    v_scaled   = vital_scaler.transform(v_raw).astype(np.float32)

    # ── Lab values ───────────────────────────────────────────────────────────
    l_raw = np.full((T_actual, N_LABS), np.nan, dtype=np.float64)
    for j, col in enumerate(LAB_COLS):
        if col in pat_df.columns:
            l_raw[:, j] = pat_df[col].values[:T_actual].astype(np.float64)

    measured = (~np.isnan(l_raw)).astype(np.float32)
    delta    = _compute_time_since_last_measurement(l_raw)
    l_filled = np.where(np.isnan(l_raw), 0.0, l_raw)
    l_scaled = lab_scaler.transform(l_filled).astype(np.float32)
    l_scaled  = l_scaled * measured   # re-zero unmeasured positions

    # ── Pad ──────────────────────────────────────────────────────────────────
    return (
        _pad_to(v_scaled, MAX_SEQ_LEN),
        _pad_to(vitals_raw, MAX_SEQ_LEN),
        _pad_to(l_scaled, MAX_SEQ_LEN),
        _pad_to(delta, MAX_SEQ_LEN),
        _pad_to(measured, MAX_SEQ_LEN),
        T_actual,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Clinical flags (rule-based bedside alerts)
# ══════════════════════════════════════════════════════════════════════════════

def _clinical_flags(row: pd.Series) -> List[str]:
    flags: List[str] = []
    if pd.notna(row.get("MAP"))  and row["MAP"]  < 65:  flags.append("Hypotension (MAP < 65)")
    if pd.notna(row.get("HR"))   and row["HR"]   > 100: flags.append("Tachycardia (HR > 100)")
    if pd.notna(row.get("Resp")) and row["Resp"] > 22:  flags.append("Tachypnea (Resp > 22)")
    if pd.notna(row.get("O2Sat"))and row["O2Sat"]< 94:  flags.append("Hypoxia (O2Sat < 94%)")
    if pd.notna(row.get("Temp")) and row["Temp"] > 38.3:flags.append("Fever (Temp > 38.3°C)")
    if pd.notna(row.get("Temp")) and row["Temp"] < 36.0:flags.append("Hypothermia (Temp < 36°C)")
    return flags


def _risk_band(probability: float, threshold: float = 0.88) -> str:
    if probability >= threshold: return "Critical"
    if probability >= (threshold * 0.8): return "High"
    if probability >= (threshold * 0.5): return "Moderate"
    return "Low"


def _make_pad_mask(length: int, max_len: int) -> "torch.Tensor":
    mask = torch.ones(1, max_len, dtype=torch.bool)
    mask[0, :length] = False
    return mask


# ══════════════════════════════════════════════════════════════════════════════
#  SepsisModelService
# ══════════════════════════════════════════════════════════════════════════════

class SepsisModelService:
    """
    Loads a saved DPCT state-dict from disk and provides predict_* methods
    for the FastAPI routes. Training is intentionally not supported here.
    """

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir    = Path(artifact_dir)
        self.weights_path    = self.artifact_dir / WEIGHTS_FILENAME

        self.model:           Optional[DualPathClinicalTransformer] = None
        self.vital_scaler:    Optional[StandardScaler]              = None
        self.lab_scaler:      Optional[StandardScaler]              = None
        self.threshold:       float                                 = DEFAULT_THRESHOLD
        self.metadata:        Dict[str, Any]                        = {}
        self.device:          str                                   = "cpu"

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load(self) -> bool:
        if not TORCH_AVAILABLE:
            return False
        if not self.weights_path.exists():
            return False

        try:
            bundle = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            model  = DualPathClinicalTransformer()
            model.load_state_dict(bundle["model_state_dict"])
            model.eval()

            self.model        = model
            self.vital_scaler = bundle["vital_scaler"]
            self.lab_scaler   = bundle["lab_scaler"]
            self.threshold    = float(bundle.get("threshold", DEFAULT_THRESHOLD))
            self.metadata     = dict(bundle.get("metadata", {}))
            return True
        except Exception as exc:
            self.metadata = {"load_error": str(exc)}
            return False

    def is_ready(self) -> bool:
        return self.model is not None and TORCH_AVAILABLE

    # ── Status ───────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        if not TORCH_AVAILABLE:
            return {
                "ready":   False,
                "message": "PyTorch is not installed. Run: pip install torch",
            }
        if not self.is_ready():
            load_error = self.metadata.get("load_error")
            if load_error:
                msg = f"Weights file found but failed to load: {load_error}"
            else:
                msg = (
                    f"No trained weights found. Download dpct_model_weights.pth "
                    f"from Kaggle and place it in the artifacts/ directory."
                )
            return {"ready": False, "message": msg}

        metrics = self.metadata.get("metrics", {})
        return {
            "ready":       True,
            "message":     "DPCT model loaded and ready for inference.",
            "model_type":  "DualPathClinicalTransformer",
            "threshold":   round(self.threshold, 4),
            "trained_at":  self.metadata.get("trained_at", "unknown"),
            "metrics": {
                "roc_auc": metrics.get("roc_auc", self.metadata.get("roc_auc", "N/A")),
                "pr_auc":  metrics.get("pr_auc",  self.metadata.get("pr_auc",  "N/A")),
            },
        }

    # ── Core inference ───────────────────────────────────────────────────────

    def _infer_single(self, pat_df: pd.DataFrame) -> Tuple[float, int]:
        """Run DPCT inference on one patient's raw DataFrame.
        Returns (probability, attention_peak_hour).
        """
        v, vr, l, d, m, seq_len = _build_single_patient_tensors(
            pat_df, self.vital_scaler, self.lab_scaler
        )
        pad_mask = _make_pad_mask(seq_len, MAX_SEQ_LEN)

        with torch.no_grad():
            logit, attn = self.model(
                torch.tensor(v).unsqueeze(0),
                torch.tensor(vr).unsqueeze(0),
                torch.tensor(l).unsqueeze(0),
                torch.tensor(d).unsqueeze(0),
                torch.tensor(m).unsqueeze(0),
                pad_mask,
            )
            prob = torch.sigmoid(logit).item()
            attn_weights = attn[0, :seq_len].cpu().numpy()
            peak_hour    = int(np.argmax(attn_weights))

        return prob, peak_hour

    # ── Public API ───────────────────────────────────────────────────────────

    def predict_from_dataframe(self, dataset: pd.DataFrame) -> List[Dict[str, Any]]:
        if not self.is_ready():
            raise ValueError("Model is not loaded. Place dpct_model_weights.pth in artifacts/.")

        # Ensure Patient_ID and Hour columns exist
        if PATIENT_COL not in dataset.columns:
            dataset = dataset.copy()
            dataset[PATIENT_COL] = "patient_0"
        if HOUR_COL not in dataset.columns:
            dataset = dataset.copy()
            dataset[HOUR_COL] = range(len(dataset))

        patient_ids = dataset[PATIENT_COL].unique().tolist()
        results: List[Dict[str, Any]] = []

        for pid in patient_ids:
            pat_df = dataset[dataset[PATIENT_COL] == pid].copy()
            try:
                prob, peak_hour = self._infer_single(pat_df)
                label = int(prob >= self.threshold)
                last_row = pat_df.sort_values(HOUR_COL).iloc[-1]
                flags    = _clinical_flags(last_row)
                results.append({
                    "patient_id":         str(pid),
                    "probability":        round(prob, 4),
                    "risk_band":          _risk_band(prob, self.threshold),
                    "predicted_label":    label,
                    "attention_peak_hour": peak_hour,
                    "flags":              flags,
                })
            except Exception as exc:
                results.append({
                    "patient_id":      str(pid),
                    "error":           str(exc),
                    "probability":     None,
                    "risk_band":       "Unknown",
                    "predicted_label": None,
                    "flags":           [],
                })

        return results

    def predict_manual(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesise a timeline from a bedside vitals payload and predict."""
        measurements = payload.get("measurements", [])
        if not measurements:
            raise ValueError("No measurements provided.")

        rows = []
        for i, m in enumerate(measurements):
            rows.append({
                PATIENT_COL: "manual-entry",
                HOUR_COL: i,
                "HR":    m.get("hr"),
                "O2Sat": m.get("o2sat"),
                "Temp":  m.get("temp"),
                "SBP":   m.get("sbp"),
                "MAP":   m.get("map"),
                "DBP":   m.get("dbp"),
                "Resp":  m.get("resp"),
                "Age":   m.get("age"),
                "WBC":   m.get("wbc"),
            })

        df = pd.DataFrame(rows)

        results = self.predict_from_dataframe(df)
        result  = results[0]
        result["patient_id"] = "manual-entry"
        return result
