"""
Configuration settings for the Smart Energy Consumption Prediction System.

Centralizes all model parameters, feature definitions, prediction horizons,
and threshold values used throughout the application.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "data"))
MODEL_DIR = os.getenv("MODEL_PATH", str(BASE_DIR / "models"))

# ---------------------------------------------------------------------------
# Data generation settings
# ---------------------------------------------------------------------------
DATA_GENERATION = {
    "start_date": "2023-01-01",
    "end_date": "2024-12-31",
    "frequency": "1H",  # hourly data
    "num_buildings": 5,
    "random_seed": 42,
}

# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------
TIME_FEATURES = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "is_weekend",
    "is_holiday",
    "season",
]

WEATHER_FEATURES = [
    "temperature",
    "humidity",
    "wind_speed",
    "cloud_cover",
]

BUILDING_FEATURES = [
    "occupancy",
    "sq_footage",
    "num_floors",
    "building_age",
]

APPLIANCE_FEATURES = [
    "hvac_usage",
    "lighting_usage",
    "equipment_usage",
]

ENGINEERED_FEATURES = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "temp_humidity_interaction",
    "occupancy_sqft_interaction",
    "is_peak_hour",
    "consumption_lag_1",
    "consumption_lag_24",
    "consumption_lag_168",
    "consumption_rolling_mean_24",
    "consumption_rolling_std_24",
    "consumption_rolling_mean_168",
]

TARGET_COLUMN = "energy_consumption_kwh"

ALL_FEATURE_COLUMNS = (
    TIME_FEATURES
    + WEATHER_FEATURES
    + BUILDING_FEATURES
    + APPLIANCE_FEATURES
    + ENGINEERED_FEATURES
)

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
MODEL_PARAMS = {
    "linear_regression": {},
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    },
    "lstm": {
        "units": 64,
        "dropout": 0.2,
        "epochs": 50,
        "batch_size": 32,
        "sequence_length": 24,
        "learning_rate": 0.001,
        "patience": 10,
    },
}

# ---------------------------------------------------------------------------
# Prediction settings
# ---------------------------------------------------------------------------
PREDICTION_HORIZON = int(os.getenv("PREDICTION_HORIZON", "24"))  # hours ahead
CONFIDENCE_LEVEL = 0.95

# ---------------------------------------------------------------------------
# Anomaly detection thresholds
# ---------------------------------------------------------------------------
ANOMALY_THRESHOLDS = {
    "z_score_threshold": 3.0,
    "iqr_multiplier": 1.5,
    "percentage_deviation": 0.30,  # 30 % deviation from predicted
}

# ---------------------------------------------------------------------------
# Flask / dashboard
# ---------------------------------------------------------------------------
FLASK_CONFIG = {
    "SECRET_KEY": os.getenv("SECRET_KEY", "smart-energy-dev-key-change-in-prod"),
    "DEBUG": os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1"),
    "HOST": os.getenv("FLASK_HOST", "0.0.0.0"),
    "PORT": int(os.getenv("FLASK_PORT", "5000")),
}

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
