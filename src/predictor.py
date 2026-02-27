"""
Prediction and anomaly detection module.

Loads a persisted model and scaler, produces multi-step forecasts with
confidence intervals, and flags anomalous consumption readings.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    MODEL_DIR,
    PREDICTION_HORIZON,
    CONFIDENCE_LEVEL,
    ANOMALY_THRESHOLDS,
    MODEL_PARAMS,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


class EnergyPredictor:
    """Loads a trained model and provides predictions with anomaly detection."""

    def __init__(self, model_dir: str = MODEL_DIR) -> None:
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.metadata: Dict = {}
        self.model_type: Optional[str] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load best model, scaler, and metadata from *model_dir*."""
        meta_path = os.path.join(self.model_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No metadata found at {meta_path}. Train a model first."
            )

        with open(meta_path) as f:
            self.metadata = json.load(f)

        self.model_type = self.metadata["best_model"]

        # Load scaler
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        self.scaler = joblib.load(scaler_path)

        # Load model
        if self.model_type == "lstm":
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            from tensorflow.keras.models import load_model
            model_path = os.path.join(self.model_dir, "best_model.keras")
            self.model = load_model(model_path)
        else:
            model_path = os.path.join(self.model_dir, "best_model.joblib")
            self.model = joblib.load(model_path)

        self._loaded = True
        logger.info("Loaded %s model from %s", self.model_type, model_path)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        X: pd.DataFrame,
        horizon: int = PREDICTION_HORIZON,
    ) -> pd.DataFrame:
        """
        Generate point predictions for the supplied feature rows.

        Parameters
        ----------
        X : pd.DataFrame   Feature matrix (must match training feature order).
        horizon : int       Number of rows to predict (truncates X).

        Returns
        -------
        pd.DataFrame with columns ``predicted_kwh``, ``lower_bound``,
        ``upper_bound``.
        """
        self._ensure_loaded()
        feature_names = self.metadata.get("feature_names", list(X.columns))
        X_aligned = X[[c for c in feature_names if c in X.columns]].head(horizon)
        X_scaled = self.scaler.transform(X_aligned)

        if self.model_type == "lstm":
            seq_len = MODEL_PARAMS["lstm"]["sequence_length"]
            if len(X_scaled) < seq_len:
                # Pad with repetition if needed
                pad = np.tile(X_scaled[0], (seq_len - len(X_scaled), 1))
                X_scaled = np.vstack([pad, X_scaled])
            sequences = []
            for i in range(seq_len, len(X_scaled) + 1):
                sequences.append(X_scaled[i - seq_len:i])
            sequences = np.array(sequences)
            predictions = self.model.predict(sequences, verbose=0).flatten()
        else:
            predictions = self.model.predict(X_scaled)

        predictions = np.maximum(predictions, 0)

        # Confidence intervals using residual-based estimation
        lower, upper = self._confidence_intervals(predictions)

        result = pd.DataFrame({
            "predicted_kwh": np.round(predictions, 2),
            "lower_bound": np.round(lower, 2),
            "upper_bound": np.round(upper, 2),
        })
        return result

    def _confidence_intervals(
        self, predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate confidence intervals using historical model RMSE.

        Uses the test-set RMSE stored in metadata as a proxy for the
        prediction standard error.
        """
        rmse = self._get_model_rmse()
        from scipy.stats import norm
        z = norm.ppf((1 + CONFIDENCE_LEVEL) / 2)
        margin = z * rmse
        return predictions - margin, predictions + margin

    def _get_model_rmse(self) -> float:
        results = self.metadata.get("results", {})
        model_metrics = results.get(self.model_type, {})
        return model_metrics.get("rmse", 10.0)  # default fallback

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------
    def detect_anomalies(
        self,
        actual: pd.Series,
        predicted: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Flag anomalous energy consumption values.

        Uses three complementary methods:
        1. Z-score on the actual series.
        2. IQR-based outlier detection.
        3. Percentage deviation from predicted values (if available).

        Returns
        -------
        pd.DataFrame  Boolean columns ``zscore_anomaly``, ``iqr_anomaly``,
        ``deviation_anomaly``, and an aggregate ``is_anomaly`` flag.
        """
        result = pd.DataFrame(index=actual.index)

        # Z-score method
        mean_val = actual.mean()
        std_val = actual.std()
        if std_val > 0:
            z_scores = np.abs((actual - mean_val) / std_val)
            result["zscore_anomaly"] = z_scores > ANOMALY_THRESHOLDS["z_score_threshold"]
        else:
            result["zscore_anomaly"] = False

        # IQR method
        q1 = actual.quantile(0.25)
        q3 = actual.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - ANOMALY_THRESHOLDS["iqr_multiplier"] * iqr
        upper = q3 + ANOMALY_THRESHOLDS["iqr_multiplier"] * iqr
        result["iqr_anomaly"] = (actual < lower) | (actual > upper)

        # Deviation from prediction
        if predicted is not None and len(predicted) == len(actual):
            deviation = np.abs(actual.values - predicted.values) / (predicted.values + 1e-8)
            result["deviation_anomaly"] = deviation > ANOMALY_THRESHOLDS["percentage_deviation"]
        else:
            result["deviation_anomaly"] = False

        result["is_anomaly"] = (
            result["zscore_anomaly"] | result["iqr_anomaly"] | result["deviation_anomaly"]
        )
        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def get_model_performance(self) -> Dict:
        """Return stored model comparison results."""
        self._ensure_loaded()
        return self.metadata.get("results", {})

    def get_feature_names(self) -> List[str]:
        self._ensure_loaded()
        return self.metadata.get("feature_names", [])
