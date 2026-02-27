"""
Feature engineering pipeline for the energy prediction system.

Transforms raw data into model-ready features including lag values,
rolling statistics, cyclical time encodings, interaction terms,
and peak-hour indicators.
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import TARGET_COLUMN

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates and manages engineered features for energy prediction models."""

    def __init__(self, target_column: str = TARGET_COLUMN) -> None:
        self.target_column = target_column
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Cyclical (sin / cos) encodings
    # ------------------------------------------------------------------
    @staticmethod
    def add_cyclical_features(
        df: pd.DataFrame,
        column: str,
        period: int,
    ) -> pd.DataFrame:
        """
        Encode a periodic integer column as sin/cos pair.

        Parameters
        ----------
        df : pd.DataFrame
        column : str   Column containing integer values (e.g. hour 0-23).
        period : int   The natural period of the feature.

        Returns
        -------
        pd.DataFrame   With two new columns: ``<column>_sin`` and ``<column>_cos``.
        """
        radians = 2 * np.pi * df[column] / period
        df[f"{column}_sin"] = np.sin(radians).round(6)
        df[f"{column}_cos"] = np.cos(radians).round(6)
        return df

    # ------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------
    @staticmethod
    def add_lag_features(
        df: pd.DataFrame,
        column: str,
        lags: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Create lagged versions of *column*.

        Parameters
        ----------
        df : pd.DataFrame
        column : str
        lags : list of int, default [1, 24, 168]
            Number of periods to lag.

        Returns
        -------
        pd.DataFrame
        """
        if lags is None:
            lags = [1, 24, 168]
        for lag in lags:
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)
        return df

    # ------------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------------
    @staticmethod
    def add_rolling_features(
        df: pd.DataFrame,
        column: str,
        windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Add rolling mean and standard deviation for *column*.

        Parameters
        ----------
        df : pd.DataFrame
        column : str
        windows : list of int, default [24, 168]
            Rolling window sizes (in rows / hours).

        Returns
        -------
        pd.DataFrame
        """
        if windows is None:
            windows = [24, 168]
        for w in windows:
            df[f"{column}_rolling_mean_{w}"] = (
                df[column].rolling(window=w, min_periods=1).mean().round(4)
            )
            df[f"{column}_rolling_std_{w}"] = (
                df[column].rolling(window=w, min_periods=1).std().round(4)
            )
        return df

    # ------------------------------------------------------------------
    # Interaction features
    # ------------------------------------------------------------------
    @staticmethod
    def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create multiplicative interaction terms between related features."""
        if "temperature" in df.columns and "humidity" in df.columns:
            df["temp_humidity_interaction"] = (
                df["temperature"] * df["humidity"]
            ).round(4)

        if "occupancy" in df.columns and "sq_footage" in df.columns:
            df["occupancy_sqft_interaction"] = (
                df["occupancy"] * df["sq_footage"]
            ).round(4)

        return df

    # ------------------------------------------------------------------
    # Peak-hour indicator
    # ------------------------------------------------------------------
    @staticmethod
    def add_peak_hour_indicator(
        df: pd.DataFrame,
        peak_hours: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Add binary flag for peak electricity-demand hours.

        Parameters
        ----------
        df : pd.DataFrame
        peak_hours : list of int, default [8..11, 17..20]
        """
        if peak_hours is None:
            peak_hours = list(range(8, 12)) + list(range(17, 21))
        df["is_peak_hour"] = df["hour"].isin(peak_hours).astype(int)
        return df

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Run the complete feature-engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame   Raw dataset (must include ``hour``, ``month``,
            ``day_of_week``, weather and building columns, and the target).
        fit : bool
            If True, records feature names for later use.

        Returns
        -------
        pd.DataFrame   Transformed dataset (NaN rows from lags are dropped).
        """
        logger.info("Starting feature engineering on %d rows ...", len(df))
        df = df.copy()

        # Cyclical encodings
        df = self.add_cyclical_features(df, "hour", 24)
        df = self.add_cyclical_features(df, "month", 12)
        df = self.add_cyclical_features(df, "day_of_week", 7)

        # Lag features
        df = self.add_lag_features(df, self.target_column, lags=[1, 24, 168])

        # Rolling statistics
        df = self.add_rolling_features(df, self.target_column, windows=[24, 168])

        # Interaction features
        df = self.add_interaction_features(df)

        # Peak-hour flag
        df = self.add_peak_hour_indicator(df)

        # Drop rows with NaN introduced by lags / rolling
        initial_len = len(df)
        df.dropna(inplace=True)
        dropped = initial_len - len(df)
        if dropped:
            logger.info("Dropped %d rows containing NaN (lag warm-up).", dropped)

        df.reset_index(drop=True, inplace=True)

        if fit:
            self._feature_names = [
                c for c in df.columns
                if c not in ("timestamp", self.target_column, "building_id")
            ]

        logger.info(
            "Feature engineering complete: %d features, %d rows.",
            len(self._feature_names), len(df),
        )
        return df

    @property
    def feature_names(self) -> List[str]:
        """Return the list of feature column names after transform."""
        return list(self._feature_names)

    def get_X_y(self, df: pd.DataFrame):
        """
        Split a transformed DataFrame into feature matrix and target vector.

        Returns
        -------
        X : pd.DataFrame
        y : pd.Series
        """
        feature_cols = [c for c in self.feature_names if c in df.columns]
        X = df[feature_cols]
        y = df[self.target_column]
        return X, y
