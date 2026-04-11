from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .feature_engineering import (
    ColumnMapping,
    build_manual_timeline,
    clinical_flags_from_row,
    engineer_temporal_features,
    prepare_time_series_dataframe,
    aggregate_patient_level,
)


@dataclass
class TrainingResult:
    metrics: Dict[str, float]
    class_distribution: Dict[str, int]
    patients: int
    feature_count: int


class SepsisModelService:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_path = self.artifact_dir / "sepsis_model.joblib"

        self.pipeline: Optional[Pipeline] = None
        self.feature_columns: List[str] = []
        self.threshold: float = 0.5
        self.metadata: Dict[str, Any] = {}

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self) -> bool:
        if not self.artifact_path.exists():
            return False

        bundle = joblib.load(self.artifact_path)
        self.pipeline = bundle["pipeline"]
        self.feature_columns = list(bundle["feature_columns"])
        self.threshold = float(bundle.get("threshold", 0.5))
        self.metadata = dict(bundle.get("metadata", {}))
        return True

    def is_ready(self) -> bool:
        return self.pipeline is not None and len(self.feature_columns) > 0

    def status(self) -> Dict[str, Any]:
        if not self.is_ready():
            return {
                "ready": False,
                "message": "No trained model is loaded. Upload a dataset and run training.",
            }

        return {
            "ready": True,
            "message": "Model is loaded and ready for prediction.",
            "threshold": round(self.threshold, 4),
            "feature_count": len(self.feature_columns),
            "model_type": self.metadata.get("model_type", "RandomForestClassifier"),
            "trained_at": self.metadata.get("trained_at"),
            "patients": self.metadata.get("patients"),
            "metrics": self.metadata.get("metrics", {}),
        }

    def _prepare_patient_features(
        self,
        df: pd.DataFrame,
        require_target: bool,
    ) -> Tuple[pd.DataFrame, ColumnMapping, pd.DataFrame, Optional[pd.Series]]:
        prepared_df, mapping = prepare_time_series_dataframe(df, require_target=require_target)
        engineered_df = engineer_temporal_features(prepared_df, mapping.patient_id, mapping.hour)
        aggregated_df, targets = aggregate_patient_level(engineered_df, mapping)
        aggregated_df = aggregated_df.replace([np.inf, -np.inf], np.nan)
        return engineered_df, mapping, aggregated_df, targets

    @staticmethod
    def _optimize_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
        thresholds = np.arange(0.1, 0.91, 0.01)
        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            score = f1_score(y_true, predictions, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)

        return best_threshold

    @staticmethod
    def _risk_band(probability: float) -> str:
        if probability >= 0.75:
            return "Critical"
        if probability >= 0.5:
            return "High"
        if probability >= 0.25:
            return "Moderate"
        return "Low"

    def train_from_dataframe(self, dataset: pd.DataFrame) -> TrainingResult:
        _, _, patient_features, targets = self._prepare_patient_features(dataset, require_target=True)

        if targets is None:
            raise ValueError("Training target could not be inferred from the dataset.")

        y = targets.reindex(patient_features.index).astype(int)
        X = patient_features.copy()

        if y.nunique() < 2:
            raise ValueError("The dataset must contain both sepsis and non-sepsis patients.")

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=None,
            )

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=450,
                        random_state=42,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                        min_samples_leaf=2,
                    ),
                ),
            ]
        )

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]
        threshold = self._optimize_threshold(y_test.to_numpy(), probabilities)
        predictions = (probabilities >= threshold).astype(int)

        pr_auc = average_precision_score(y_test, probabilities)

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "pr_auc": float(pr_auc),
            "threshold": float(threshold),
        }

        matrix = confusion_matrix(y_test, predictions)
        if matrix.size == 4:
            tn, fp, fn, tp = matrix.ravel()
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
            metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else 0.0

        class_distribution = {
            "non_sepsis": int((y == 0).sum()),
            "sepsis": int((y == 1).sum()),
        }

        self.pipeline = pipeline
        self.feature_columns = list(X.columns)
        self.threshold = threshold
        self.metadata = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(),
            "patients": int(len(X)),
            "feature_count": int(len(self.feature_columns)),
            "model_type": "RandomForestClassifier",
            "metrics": metrics,
        }

        bundle = {
            "pipeline": self.pipeline,
            "feature_columns": self.feature_columns,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }
        joblib.dump(bundle, self.artifact_path)

        return TrainingResult(
            metrics=metrics,
            class_distribution=class_distribution,
            patients=int(len(X)),
            feature_count=int(len(self.feature_columns)),
        )

    def _align_feature_space(self, patient_features: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_columns:
            raise ValueError("Model feature columns are unavailable.")

        aligned = patient_features.reindex(columns=self.feature_columns)
        aligned = aligned.replace([np.inf, -np.inf], np.nan)
        return aligned

    def predict_from_dataframe(self, dataset: pd.DataFrame) -> List[Dict[str, Any]]:
        if not self.is_ready():
            raise ValueError("Model is not trained yet. Train the model first.")

        engineered_df, mapping, patient_features, _ = self._prepare_patient_features(dataset, require_target=False)

        if patient_features.empty:
            raise ValueError("No patient records were available after preprocessing.")

        aligned_features = self._align_feature_space(patient_features)
        probabilities = self.pipeline.predict_proba(aligned_features)[:, 1]

        latest_rows = (
            engineered_df.sort_values([mapping.patient_id, mapping.hour])
            .groupby(mapping.patient_id, sort=False)
            .tail(1)
            .set_index(mapping.patient_id)
        )

        predictions: List[Dict[str, Any]] = []
        for index, patient_id in enumerate(aligned_features.index.tolist()):
            probability = float(probabilities[index])
            predicted_label = int(probability >= self.threshold)

            latest_row = latest_rows.loc[patient_id]
            if isinstance(latest_row, pd.DataFrame):
                latest_row = latest_row.iloc[-1]

            predictions.append(
                {
                    "patient_id": str(patient_id),
                    "probability": round(probability, 4),
                    "risk_band": self._risk_band(probability),
                    "predicted_label": predicted_label,
                    "flags": clinical_flags_from_row(latest_row),
                }
            )

        return predictions

    def predict_manual(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        manual_df = build_manual_timeline(payload)
        prediction = self.predict_from_dataframe(manual_df)[0]
        prediction["patient_id"] = "manual-entry"
        return prediction
