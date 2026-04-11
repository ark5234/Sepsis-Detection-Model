from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


VITAL_SIGN_COLUMNS = ["hr", "o2sat", "temp", "sbp", "map", "dbp", "resp"]
TIME_COLUMN_CANDIDATES = ["hour", "iculos", "icu_los", "icu_hour", "time"]
PATIENT_COLUMN_CANDIDATES = ["patient_id", "patientid", "subject_id", "stay_id", "icustay_id"]
TARGET_COLUMN_CANDIDATES = ["sepsislabel", "sepsis_label", "sepsis", "target", "label"]


@dataclass(frozen=True)
class ColumnMapping:
    patient_id: str
    hour: str
    target: Optional[str] = None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    return normalized


def _choose_column(df: pd.DataFrame, candidates: list[str], required_parts: Optional[Tuple[str, ...]] = None) -> Optional[str]:
    columns = list(df.columns)
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    if required_parts:
        for column in columns:
            if all(part in column for part in required_parts):
                return column

    return None


def _encode_gender(df: pd.DataFrame) -> None:
    gender_col = _choose_column(df, ["gender", "sex"])
    if not gender_col:
        return

    mapping = {
        "female": 0,
        "f": 0,
        "0": 0,
        0: 0,
        "male": 1,
        "m": 1,
        "1": 1,
        1: 1,
    }

    encoded = df[gender_col].map(mapping)
    numeric_version = pd.to_numeric(df[gender_col], errors="coerce")
    df[gender_col] = encoded.fillna(numeric_version).fillna(0)

    if gender_col != "gender":
        df["gender"] = df[gender_col]


def _coerce_numeric(df: pd.DataFrame, excluded: set[str]) -> None:
    for column in df.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")


def prepare_time_series_dataframe(raw_df: pd.DataFrame, require_target: bool) -> Tuple[pd.DataFrame, ColumnMapping]:
    if raw_df.empty:
        raise ValueError("The uploaded CSV is empty.")

    df = normalize_columns(raw_df)

    patient_col = _choose_column(df, PATIENT_COLUMN_CANDIDATES, required_parts=("patient", "id"))
    if not patient_col:
        patient_col = "patient_id"
        df[patient_col] = np.arange(len(df), dtype=int)

    hour_col = _choose_column(df, TIME_COLUMN_CANDIDATES)
    if not hour_col:
        hour_col = "hour"
        df[hour_col] = df.groupby(patient_col).cumcount()

    target_col = _choose_column(df, TARGET_COLUMN_CANDIDATES, required_parts=("sepsis",))
    if require_target and not target_col:
        raise ValueError(
            "No sepsis target column found. Include one of: SepsisLabel, sepsis, target, or label."
        )

    if target_col and target_col != "sepsislabel":
        df["sepsislabel"] = df[target_col]
        target_col = "sepsislabel"

    _encode_gender(df)

    excluded = {patient_col}
    if target_col:
        excluded.add(target_col)
    _coerce_numeric(df, excluded=excluded)

    fallback_hours = df.groupby(patient_col).cumcount()
    df[hour_col] = pd.to_numeric(df[hour_col], errors="coerce").fillna(fallback_hours).astype(float)

    if target_col:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).clip(0, 1).astype(int)

    df = df.sort_values([patient_col, hour_col]).reset_index(drop=True)

    columns_to_fill = [column for column in df.columns if column != patient_col]
    df[columns_to_fill] = df.groupby(patient_col, sort=False)[columns_to_fill].ffill()

    if target_col:
        df[target_col] = df[target_col].fillna(0).astype(int)

    return df, ColumnMapping(patient_id=patient_col, hour=hour_col, target=target_col)


def engineer_temporal_features(df: pd.DataFrame, patient_col: str, hour_col: str) -> pd.DataFrame:
    engineered = df.copy()

    for feature in [column for column in VITAL_SIGN_COLUMNS if column in engineered.columns]:
        grouped = engineered.groupby(patient_col, sort=False)[feature]

        engineered[f"{feature}_rolling_mean_6h"] = grouped.transform(
            lambda series: series.rolling(window=6, min_periods=1).mean()
        )
        engineered[f"{feature}_rolling_std_6h"] = grouped.transform(
            lambda series: series.rolling(window=6, min_periods=1).std()
        ).fillna(0.0)

        feature_diff = grouped.diff().fillna(0.0)
        engineered[f"{feature}_diff"] = feature_diff
        engineered[f"{feature}_trend"] = feature_diff.groupby(engineered[patient_col]).transform(
            lambda series: series.rolling(window=3, min_periods=1).mean()
        ).fillna(0.0)

    engineered["cardiovascular_risk"] = 0.0
    if "map" in engineered.columns:
        engineered.loc[engineered["map"] < 70, "cardiovascular_risk"] = 1.0
        engineered.loc[engineered["map"] < 60, "cardiovascular_risk"] = 2.0

    engineered["respiratory_risk"] = 0.0
    if "o2sat" in engineered.columns:
        engineered.loc[engineered["o2sat"] < 95, "respiratory_risk"] = 1.0
        engineered.loc[engineered["o2sat"] < 90, "respiratory_risk"] = 2.0

    if "hr" in engineered.columns and "sbp" in engineered.columns:
        denominator = engineered["sbp"].replace(0, np.nan)
        engineered["shock_index"] = (engineered["hr"] / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if hour_col in engineered.columns:
        engineered["icu_day"] = (engineered[hour_col] // 24) + 1
        engineered["hour_of_day"] = engineered[hour_col] % 24
        engineered["is_night"] = (
            (engineered["hour_of_day"] >= 22) | (engineered["hour_of_day"] <= 6)
        ).astype(float)

    return engineered


def aggregate_patient_level(
    df: pd.DataFrame,
    mapping: ColumnMapping,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    excluded = {mapping.patient_id, mapping.hour}
    if mapping.target:
        excluded.add(mapping.target)

    numeric_columns = [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column]) and df[column].notna().any()
    ]

    if not numeric_columns:
        raise ValueError("No numeric clinical features available after preprocessing.")

    grouped = df.groupby(mapping.patient_id, sort=False)

    aggregated = grouped[numeric_columns].agg(["mean", "std", "min", "max", "last"])
    aggregated.columns = [f"{column}_{stat}" for column, stat in aggregated.columns]

    trend_features = grouped[numeric_columns].last() - grouped[numeric_columns].first()
    trend_features.columns = [f"{column}_trend_total" for column in numeric_columns]

    aggregated = pd.concat([aggregated, trend_features], axis=1)
    aggregated = aggregated.replace([np.inf, -np.inf], np.nan)

    target_series: Optional[pd.Series] = None
    if mapping.target:
        target_series = grouped[mapping.target].max().fillna(0).astype(int)

    return aggregated, target_series


def build_manual_timeline(payload: Dict[str, Any], horizon_hours: int = 6) -> pd.DataFrame:
    patient_id = "manual_patient"

    normalized_payload: Dict[str, Any] = {
        str(key).strip().lower(): value for key, value in payload.items() if value is not None
    }

    base_hour = 0
    if "iculos" in normalized_payload:
        try:
            icu_value = float(normalized_payload["iculos"])
            base_hour = max(int(icu_value - horizon_hours + 1), 0)
        except (TypeError, ValueError):
            base_hour = 0

    rows = []
    for offset in range(horizon_hours):
        row = {"patient_id": patient_id, "hour": float(base_hour + offset)}
        for key, value in normalized_payload.items():
            if key in {"patient_id", "hour"}:
                continue
            row[key] = value
        rows.append(row)

    return pd.DataFrame(rows)


def clinical_flags_from_row(row: pd.Series) -> list[str]:
    flags: list[str] = []

    def _value(column: str) -> Optional[float]:
        if column not in row.index:
            return None
        value = row[column]
        if pd.isna(value):
            return None
        return float(value)

    temperature = _value("temp")
    if temperature is not None and temperature >= 38.0:
        flags.append("Fever pattern detected (Temp >= 38.0 C)")

    heart_rate = _value("hr")
    if heart_rate is not None and heart_rate >= 90.0:
        flags.append("Tachycardia pattern detected (HR >= 90 bpm)")

    map_value = _value("map")
    if map_value is not None and map_value < 70.0:
        flags.append("Low perfusion risk detected (MAP < 70 mmHg)")

    oxygen = _value("o2sat")
    if oxygen is not None and oxygen < 95.0:
        flags.append("Respiratory compromise risk detected (O2Sat < 95%)")

    respiration = _value("resp")
    if respiration is not None and respiration > 22.0:
        flags.append("Tachypnea pattern detected (Resp > 22 breaths/min)")

    return flags
