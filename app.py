"""
Flask application for the Smart Energy Consumption Prediction System.

Provides a web dashboard, REST API for predictions, data upload,
model training triggers, and visualization endpoints.
"""

import os
import sys
import json
import logging
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.settings import FLASK_CONFIG, DATA_DIR, MODEL_DIR, TARGET_COLUMN
from src.data_generator import EnergyDataGenerator
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import EnergyPredictor
from src.visualizer import EnergyVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = FLASK_CONFIG["SECRET_KEY"]

# Shared instances
feature_engineer = FeatureEngineer()
visualizer = EnergyVisualizer()
predictor = EnergyPredictor()

DATA_PATH = os.path.join(DATA_DIR, "energy_data.csv")


# ======================================================================
# Helpers
# ======================================================================
def _load_data() -> pd.DataFrame:
    """Load the CSV dataset, returning an empty DataFrame on failure."""
    if not os.path.exists(DATA_PATH):
        logger.warning("Dataset not found at %s", DATA_PATH)
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df


def _load_transformed_data():
    """Load data and apply feature engineering."""
    df = _load_data()
    if df.empty:
        return df, df, pd.Series(dtype=float)
    transformed = feature_engineer.transform(df)
    X, y = feature_engineer.get_X_y(transformed)
    return transformed, X, y


# ======================================================================
# Dashboard page
# ======================================================================
@app.route("/")
def dashboard():
    """Render the main dashboard page."""
    data_exists = os.path.exists(DATA_PATH)
    model_exists = os.path.exists(os.path.join(MODEL_DIR, "metadata.json"))
    return render_template(
        "dashboard.html",
        data_exists=data_exists,
        model_exists=model_exists,
    )


# ======================================================================
# Visualization API
# ======================================================================
@app.route("/api/charts/consumption", methods=["GET"])
def chart_consumption():
    """Return consumption-over-time chart JSON."""
    df = _load_data()
    if df.empty:
        return jsonify({"error": "No data available"}), 404
    # Sample for performance (max 5000 points)
    if len(df) > 5000:
        df = df.sample(5000, random_state=42).sort_values("timestamp")
    chart = visualizer.consumption_over_time(df)
    return chart, 200, {"Content-Type": "application/json"}


@app.route("/api/charts/hourly", methods=["GET"])
def chart_hourly():
    df = _load_data()
    if df.empty:
        return jsonify({"error": "No data available"}), 404
    chart = visualizer.hourly_pattern(df)
    return chart, 200, {"Content-Type": "application/json"}


@app.route("/api/charts/daily", methods=["GET"])
def chart_daily():
    df = _load_data()
    if df.empty:
        return jsonify({"error": "No data available"}), 404
    chart = visualizer.daily_pattern(df)
    return chart, 200, {"Content-Type": "application/json"}


@app.route("/api/charts/monthly", methods=["GET"])
def chart_monthly():
    df = _load_data()
    if df.empty:
        return jsonify({"error": "No data available"}), 404
    chart = visualizer.monthly_trend(df)
    return chart, 200, {"Content-Type": "application/json"}


@app.route("/api/charts/model-comparison", methods=["GET"])
def chart_model_comparison():
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        return jsonify({"error": "No trained model found"}), 404
    with open(meta_path) as f:
        meta = json.load(f)
    chart = visualizer.model_comparison(meta.get("results", {}))
    return chart, 200, {"Content-Type": "application/json"}


@app.route("/api/charts/prediction", methods=["GET"])
def chart_prediction():
    """Produce a prediction-vs-actual chart on the test tail."""
    df = _load_data()
    if df.empty:
        return jsonify({"error": "No data available"}), 404

    try:
        predictor.load()
    except FileNotFoundError:
        return jsonify({"error": "No trained model found"}), 404

    transformed = feature_engineer.transform(df, fit=True)
    X, y = feature_engineer.get_X_y(transformed)

    tail = min(200, len(X))
    X_tail = X.tail(tail)
    y_tail = y.tail(tail).reset_index(drop=True)

    preds = predictor.predict(X_tail, horizon=tail)
    timestamps = transformed["timestamp"].tail(tail).reset_index(drop=True)

    chart = visualizer.prediction_vs_actual(
        actual=y_tail,
        predicted=preds["predicted_kwh"],
        timestamps=timestamps,
        lower=preds["lower_bound"],
        upper=preds["upper_bound"],
    )
    return chart, 200, {"Content-Type": "application/json"}


@app.route("/api/charts/anomalies", methods=["GET"])
def chart_anomalies():
    df = _load_data()
    if df.empty:
        return jsonify({"error": "No data available"}), 404

    try:
        predictor.load()
    except FileNotFoundError:
        return jsonify({"error": "No trained model found"}), 404

    transformed = feature_engineer.transform(df, fit=True)
    X, y = feature_engineer.get_X_y(transformed)

    tail = min(500, len(X))
    X_tail = X.tail(tail)
    y_tail = y.tail(tail).reset_index(drop=True)

    preds = predictor.predict(X_tail, horizon=tail)
    anomalies = predictor.detect_anomalies(y_tail, preds["predicted_kwh"])
    timestamps = transformed["timestamp"].tail(tail).reset_index(drop=True)

    chart_df = pd.DataFrame({TARGET_COLUMN: y_tail.values})
    chart = visualizer.anomaly_chart(chart_df, anomalies["is_anomaly"], timestamps)
    return chart, 200, {"Content-Type": "application/json"}


@app.route("/api/charts/feature-importance", methods=["GET"])
def chart_feature_importance():
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        return jsonify({"error": "No trained model found"}), 404

    trainer = ModelTrainer()
    # Try loading best model for feature importance
    try:
        import joblib
        model_path = os.path.join(MODEL_DIR, "best_model.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            with open(meta_path) as f:
                meta = json.load(f)
            feature_names = meta.get("feature_names", [])
            if hasattr(model, "feature_importances_"):
                importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
            elif hasattr(model, "coef_"):
                importance = pd.Series(abs(model.coef_), index=feature_names).sort_values(ascending=False)
            else:
                return jsonify({"error": "Model does not expose feature importances"}), 400
            chart = visualizer.feature_importance(importance)
            return chart, 200, {"Content-Type": "application/json"}
    except Exception as exc:
        logger.exception("Feature importance error")
        return jsonify({"error": str(exc)}), 500

    return jsonify({"error": "Could not compute feature importance"}), 400


# ======================================================================
# Prediction API
# ======================================================================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accept JSON payload with features and return predictions.

    Expects ``{"features": {col: [values], ...}, "horizon": 24}``.
    """
    try:
        predictor.load()
    except FileNotFoundError:
        return jsonify({"error": "No trained model. Train first."}), 404

    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({"error": "Provide 'features' in JSON body"}), 400

    horizon = data.get("horizon", 24)
    X = pd.DataFrame(data["features"])
    preds = predictor.predict(X, horizon=horizon)
    return jsonify(preds.to_dict(orient="list"))


@app.route("/api/predict/summary", methods=["GET"])
def prediction_summary():
    """Quick summary stats from the most recent prediction window."""
    df = _load_data()
    if df.empty:
        return jsonify({"error": "No data available"}), 404
    try:
        predictor.load()
    except FileNotFoundError:
        return jsonify({"error": "No trained model"}), 404

    transformed = feature_engineer.transform(df, fit=True)
    X, y = feature_engineer.get_X_y(transformed)
    tail = min(48, len(X))
    preds = predictor.predict(X.tail(tail), horizon=tail)
    actual = y.tail(tail).values

    return jsonify({
        "current_consumption": round(float(actual[-1]), 2),
        "predicted_next": round(float(preds["predicted_kwh"].iloc[-1]), 2),
        "avg_24h": round(float(actual[-min(24, len(actual)):].mean()), 2),
        "peak_24h": round(float(actual[-min(24, len(actual)):].max()), 2),
        "anomaly_count": int(predictor.detect_anomalies(
            pd.Series(actual), preds["predicted_kwh"]
        )["is_anomaly"].sum()),
    })


# ======================================================================
# Data management
# ======================================================================
@app.route("/api/data/generate", methods=["POST"])
def generate_data():
    """Generate synthetic demo data."""
    try:
        gen = EnergyDataGenerator()
        df = gen.generate(output_path=DATA_PATH)
        return jsonify({
            "status": "success",
            "rows": len(df),
            "path": DATA_PATH,
        })
    except Exception as exc:
        logger.exception("Data generation failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/data/upload", methods=["POST"])
def upload_data():
    """Upload a CSV file to replace the current dataset."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files accepted"}), 400

    os.makedirs(DATA_DIR, exist_ok=True)
    file.save(DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    return jsonify({"status": "success", "rows": len(df)})


@app.route("/api/data/stats", methods=["GET"])
def data_stats():
    """Return basic statistics about the loaded dataset."""
    df = _load_data()
    if df.empty:
        return jsonify({"error": "No data available"}), 404

    stats = {
        "total_rows": len(df),
        "date_range": {
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
        },
        "columns": list(df.columns),
        "consumption_stats": {
            "mean": round(float(df[TARGET_COLUMN].mean()), 2),
            "std": round(float(df[TARGET_COLUMN].std()), 2),
            "min": round(float(df[TARGET_COLUMN].min()), 2),
            "max": round(float(df[TARGET_COLUMN].max()), 2),
        },
    }
    return jsonify(stats)


# ======================================================================
# Model training
# ======================================================================
@app.route("/api/model/train", methods=["POST"])
def train_model():
    """Trigger model training on the current dataset."""
    df = _load_data()
    if df.empty:
        return jsonify({"error": "No data available. Generate or upload first."}), 404

    try:
        transformed = feature_engineer.transform(df)
        X, y = feature_engineer.get_X_y(transformed)

        data = request.get_json(silent=True) or {}
        models_to_train = data.get("models", ["linear_regression", "random_forest", "xgboost"])

        trainer = ModelTrainer()
        results = trainer.train_all(X, y, models_to_train=models_to_train)
        model_path = trainer.save_best_model()
        report = trainer.get_comparison_report().to_dict()

        return jsonify({
            "status": "success",
            "best_model": trainer.best_model_name,
            "model_path": model_path,
            "results": results,
            "comparison": report,
        })
    except Exception as exc:
        logger.exception("Training failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/model/status", methods=["GET"])
def model_status():
    """Check whether a trained model exists and return its metadata."""
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        return jsonify({"trained": False})
    with open(meta_path) as f:
        meta = json.load(f)
    return jsonify({"trained": True, **meta})


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    app.run(
        host=FLASK_CONFIG["HOST"],
        port=FLASK_CONFIG["PORT"],
        debug=FLASK_CONFIG["DEBUG"],
    )
