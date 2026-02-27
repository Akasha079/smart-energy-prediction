"""
Model training and evaluation module.

Trains Linear Regression, Random Forest, XGBoost, and LSTM models,
compares their performance on MAE / RMSE / R-squared, saves the best
model to disk, and produces a comparison report.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    MODEL_DIR,
    MODEL_PARAMS,
    TEST_SIZE,
    VALIDATION_SIZE,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains, evaluates, and persists energy-prediction models."""

    def __init__(self, model_dir: str = MODEL_DIR) -> None:
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.best_model_name: Optional[str] = None
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Data splitting helpers
    # ------------------------------------------------------------------
    def _split_data(
        self, X: pd.DataFrame, y: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split into train / validation / test sets and scale features."""
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, shuffle=False,
        )
        val_frac = VALIDATION_SIZE / (1 - TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_frac, shuffle=False,
        )

        X_train_sc = self.scaler.fit_transform(X_train)
        X_val_sc = self.scaler.transform(X_val)
        X_test_sc = self.scaler.transform(X_test)

        return X_train_sc, X_val_sc, X_test_sc, y_train.values, y_val.values, y_test.values

    # ------------------------------------------------------------------
    # Individual model trainers
    # ------------------------------------------------------------------
    def _train_linear_regression(self, X_train, y_train) -> LinearRegression:
        logger.info("Training Linear Regression ...")
        model = LinearRegression(**MODEL_PARAMS["linear_regression"])
        model.fit(X_train, y_train)
        return model

    def _train_random_forest(self, X_train, y_train) -> RandomForestRegressor:
        logger.info("Training Random Forest ...")
        model = RandomForestRegressor(**MODEL_PARAMS["random_forest"])
        model.fit(X_train, y_train)
        return model

    def _train_xgboost(self, X_train, y_train):
        """Train an XGBoost regressor."""
        logger.info("Training XGBoost ...")
        try:
            from xgboost import XGBRegressor
        except ImportError:
            logger.warning("xgboost not installed -- skipping.")
            return None
        model = XGBRegressor(**MODEL_PARAMS["xgboost"])
        model.fit(X_train, y_train, verbose=False)
        return model

    def _train_lstm(self, X_train, y_train, X_val, y_val):
        """Train a Keras LSTM model."""
        logger.info("Training LSTM ...")
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            from tensorflow import keras
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            logger.warning("tensorflow not installed -- skipping LSTM.")
            return None

        params = MODEL_PARAMS["lstm"]
        seq_len = params["sequence_length"]

        def _make_sequences(X, y, seq_len):
            Xs, ys = [], []
            for i in range(seq_len, len(X)):
                Xs.append(X[i - seq_len:i])
                ys.append(y[i])
            return np.array(Xs), np.array(ys)

        X_tr_seq, y_tr_seq = _make_sequences(X_train, y_train, seq_len)
        X_va_seq, y_va_seq = _make_sequences(X_val, y_val, seq_len)

        if len(X_tr_seq) == 0 or len(X_va_seq) == 0:
            logger.warning("Not enough data for LSTM sequences -- skipping.")
            return None

        model = Sequential([
            LSTM(params["units"], input_shape=(seq_len, X_train.shape[1]),
                 return_sequences=True),
            Dropout(params["dropout"]),
            LSTM(params["units"] // 2),
            Dropout(params["dropout"]),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
            loss="mse",
        )
        es = EarlyStopping(
            monitor="val_loss", patience=params["patience"], restore_best_weights=True,
        )
        model.fit(
            X_tr_seq, y_tr_seq,
            validation_data=(X_va_seq, y_va_seq),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=[es],
            verbose=0,
        )
        # Store seq_len on the model for prediction
        model._seq_len = seq_len
        return model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4),
        }

    def _predict_lstm(self, model, X_scaled: np.ndarray) -> np.ndarray:
        seq_len = getattr(model, "_seq_len", MODEL_PARAMS["lstm"]["sequence_length"])
        Xs = []
        for i in range(seq_len, len(X_scaled)):
            Xs.append(X_scaled[i - seq_len:i])
        Xs = np.array(Xs)
        return model.predict(Xs, verbose=0).flatten()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models_to_train: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all (or selected) models and evaluate on the test set.

        Parameters
        ----------
        X : pd.DataFrame   Feature matrix.
        y : pd.Series      Target vector.
        models_to_train : list of str, optional
            Subset of ["linear_regression", "random_forest", "xgboost", "lstm"].

        Returns
        -------
        dict   ``{model_name: {mae, rmse, r2}}``
        """
        self.feature_names = list(X.columns)
        available = ["linear_regression", "random_forest", "xgboost", "lstm"]
        to_train = models_to_train or available

        X_tr, X_va, X_te, y_tr, y_va, y_te = self._split_data(X, y)

        trainers = {
            "linear_regression": lambda: self._train_linear_regression(X_tr, y_tr),
            "random_forest": lambda: self._train_random_forest(X_tr, y_tr),
            "xgboost": lambda: self._train_xgboost(X_tr, y_tr),
            "lstm": lambda: self._train_lstm(X_tr, y_tr, X_va, y_va),
        }

        for name in to_train:
            if name not in trainers:
                logger.warning("Unknown model: %s", name)
                continue
            model = trainers[name]()
            if model is None:
                continue
            self.models[name] = model

            # Predict on test set
            if name == "lstm":
                seq_len = getattr(model, "_seq_len", MODEL_PARAMS["lstm"]["sequence_length"])
                y_pred = self._predict_lstm(model, X_te)
                y_actual = y_te[seq_len:]
            else:
                y_pred = model.predict(X_te)
                y_actual = y_te

            metrics = self._evaluate(y_actual, y_pred)
            self.results[name] = metrics
            logger.info("%s -> MAE=%.2f  RMSE=%.2f  R2=%.4f",
                        name, metrics["mae"], metrics["rmse"], metrics["r2"])

        # Determine best model by lowest RMSE
        if self.results:
            self.best_model_name = min(self.results, key=lambda k: self.results[k]["rmse"])
            logger.info("Best model: %s", self.best_model_name)

        return self.results

    def save_best_model(self) -> str:
        """Persist the best model, scaler, and metadata to *model_dir*."""
        if not self.best_model_name:
            raise RuntimeError("No trained models to save.")

        model = self.models[self.best_model_name]

        if self.best_model_name == "lstm":
            model_path = os.path.join(self.model_dir, "best_model.keras")
            model.save(model_path)
        else:
            model_path = os.path.join(self.model_dir, "best_model.joblib")
            joblib.dump(model, model_path)

        # Save scaler
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        joblib.dump(self.scaler, scaler_path)

        # Save metadata
        meta = {
            "best_model": self.best_model_name,
            "feature_names": self.feature_names,
            "results": self.results,
        }
        meta_path = os.path.join(self.model_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved best model (%s) to %s", self.best_model_name, model_path)
        return model_path

    def get_comparison_report(self) -> pd.DataFrame:
        """Return a DataFrame comparing all trained models."""
        if not self.results:
            return pd.DataFrame()
        report = pd.DataFrame(self.results).T
        report.index.name = "model"
        report.sort_values("rmse", inplace=True)
        return report

    def get_feature_importance(self, model_name: Optional[str] = None) -> Optional[pd.Series]:
        """
        Return feature importances for tree-based models.

        Falls back to the best model if *model_name* is not specified.
        """
        name = model_name or self.best_model_name
        model = self.models.get(name)
        if model is None:
            return None
        if hasattr(model, "feature_importances_"):
            return pd.Series(
                model.feature_importances_, index=self.feature_names,
            ).sort_values(ascending=False)
        if hasattr(model, "coef_"):
            return pd.Series(
                np.abs(model.coef_), index=self.feature_names,
            ).sort_values(ascending=False)
        return None
