"""
Visualization module using Plotly.

Generates interactive charts for consumption trends, predictions,
feature importance, daily/hourly patterns, and anomalies.
All chart functions return Plotly JSON strings suitable for embedding
in the Flask frontend.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import TARGET_COLUMN

logger = logging.getLogger(__name__)

_DARK_TEMPLATE = "plotly_dark"
_COLOR_PRIMARY = "#00d4ff"
_COLOR_SECONDARY = "#ff6b6b"
_COLOR_TERTIARY = "#51cf66"
_COLOR_WARNING = "#fcc419"


class EnergyVisualizer:
    """Creates Plotly charts for the energy dashboard."""

    def __init__(self, target_column: str = TARGET_COLUMN) -> None:
        self.target_column = target_column

    @staticmethod
    def _to_json(fig: go.Figure) -> str:
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # ------------------------------------------------------------------
    # 1. Consumption over time
    # ------------------------------------------------------------------
    def consumption_over_time(
        self,
        df: pd.DataFrame,
        title: str = "Energy Consumption Over Time",
    ) -> str:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df[self.target_column],
            mode="lines",
            name="Consumption",
            line=dict(color=_COLOR_PRIMARY, width=1),
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Energy (kWh)",
            template=_DARK_TEMPLATE,
            height=400,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return self._to_json(fig)

    # ------------------------------------------------------------------
    # 2. Prediction vs actual
    # ------------------------------------------------------------------
    def prediction_vs_actual(
        self,
        actual: pd.Series,
        predicted: pd.Series,
        timestamps: Optional[pd.Series] = None,
        lower: Optional[pd.Series] = None,
        upper: Optional[pd.Series] = None,
    ) -> str:
        x = timestamps if timestamps is not None else list(range(len(actual)))
        fig = go.Figure()

        # Confidence band
        if lower is not None and upper is not None:
            fig.add_trace(go.Scatter(
                x=pd.concat([x, x[::-1]]) if isinstance(x, pd.Series) else list(x) + list(reversed(x)),
                y=pd.concat([upper, lower[::-1]]) if isinstance(upper, pd.Series) else list(upper) + list(reversed(lower)),
                fill="toself",
                fillcolor="rgba(0,212,255,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Confidence Interval",
            ))

        fig.add_trace(go.Scatter(
            x=x, y=actual, mode="lines",
            name="Actual", line=dict(color=_COLOR_PRIMARY, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=x, y=predicted, mode="lines",
            name="Predicted", line=dict(color=_COLOR_SECONDARY, width=2, dash="dash"),
        ))

        fig.update_layout(
            title="Prediction vs Actual Consumption",
            xaxis_title="Time",
            yaxis_title="Energy (kWh)",
            template=_DARK_TEMPLATE,
            height=400,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return self._to_json(fig)

    # ------------------------------------------------------------------
    # 3. Feature importance
    # ------------------------------------------------------------------
    def feature_importance(
        self,
        importance: pd.Series,
        top_n: int = 15,
    ) -> str:
        imp = importance.head(top_n).sort_values()
        fig = go.Figure(go.Bar(
            x=imp.values,
            y=imp.index,
            orientation="h",
            marker_color=_COLOR_PRIMARY,
        ))
        fig.update_layout(
            title=f"Top {top_n} Feature Importances",
            xaxis_title="Importance",
            yaxis_title="Feature",
            template=_DARK_TEMPLATE,
            height=450,
            margin=dict(l=160, r=20, t=50, b=40),
        )
        return self._to_json(fig)

    # ------------------------------------------------------------------
    # 4. Hourly pattern
    # ------------------------------------------------------------------
    def hourly_pattern(self, df: pd.DataFrame) -> str:
        hourly = df.groupby("hour")[self.target_column].agg(["mean", "std"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hourly["hour"],
            y=hourly["mean"],
            error_y=dict(type="data", array=hourly["std"].values, visible=True),
            marker_color=_COLOR_PRIMARY,
            name="Avg Consumption",
        ))
        fig.update_layout(
            title="Average Hourly Consumption Pattern",
            xaxis_title="Hour of Day",
            yaxis_title="Energy (kWh)",
            template=_DARK_TEMPLATE,
            height=350,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return self._to_json(fig)

    # ------------------------------------------------------------------
    # 5. Daily pattern
    # ------------------------------------------------------------------
    def daily_pattern(self, df: pd.DataFrame) -> str:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily = df.groupby("day_of_week")[self.target_column].mean().reset_index()
        daily["day_name"] = daily["day_of_week"].map(lambda d: day_names[d])
        fig = go.Figure(go.Bar(
            x=daily["day_name"],
            y=daily[self.target_column],
            marker_color=_COLOR_TERTIARY,
        ))
        fig.update_layout(
            title="Average Daily Consumption Pattern",
            xaxis_title="Day of Week",
            yaxis_title="Energy (kWh)",
            template=_DARK_TEMPLATE,
            height=350,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return self._to_json(fig)

    # ------------------------------------------------------------------
    # 6. Anomaly highlights
    # ------------------------------------------------------------------
    def anomaly_chart(
        self,
        df: pd.DataFrame,
        anomaly_flags: pd.Series,
        timestamps: Optional[pd.Series] = None,
    ) -> str:
        x = timestamps if timestamps is not None else df.index
        y = df[self.target_column].values

        normal_mask = ~anomaly_flags.values
        anomaly_mask = anomaly_flags.values

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x[normal_mask] if isinstance(x, pd.Series) else [x[i] for i, m in enumerate(normal_mask) if m],
            y=y[normal_mask],
            mode="markers",
            marker=dict(color=_COLOR_PRIMARY, size=3),
            name="Normal",
        ))
        fig.add_trace(go.Scatter(
            x=x[anomaly_mask] if isinstance(x, pd.Series) else [x[i] for i, m in enumerate(anomaly_mask) if m],
            y=y[anomaly_mask],
            mode="markers",
            marker=dict(color=_COLOR_SECONDARY, size=8, symbol="x"),
            name="Anomaly",
        ))
        fig.update_layout(
            title="Anomaly Detection",
            xaxis_title="Time",
            yaxis_title="Energy (kWh)",
            template=_DARK_TEMPLATE,
            height=400,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return self._to_json(fig)

    # ------------------------------------------------------------------
    # 7. Model comparison
    # ------------------------------------------------------------------
    def model_comparison(self, results: Dict[str, Dict[str, float]]) -> str:
        models = list(results.keys())
        metrics = ["mae", "rmse", "r2"]
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["MAE", "RMSE", "R-squared"],
        )
        colors = [_COLOR_PRIMARY, _COLOR_SECONDARY, _COLOR_TERTIARY, _COLOR_WARNING]
        for i, metric in enumerate(metrics):
            values = [results[m].get(metric, 0) for m in models]
            fig.add_trace(
                go.Bar(
                    x=models, y=values,
                    marker_color=colors[:len(models)],
                    showlegend=False,
                ),
                row=1, col=i + 1,
            )
        fig.update_layout(
            title="Model Performance Comparison",
            template=_DARK_TEMPLATE,
            height=350,
            margin=dict(l=40, r=20, t=70, b=40),
        )
        return self._to_json(fig)

    # ------------------------------------------------------------------
    # 8. Monthly trend
    # ------------------------------------------------------------------
    def monthly_trend(self, df: pd.DataFrame) -> str:
        monthly = df.groupby("month")[self.target_column].mean().reset_index()
        month_labels = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        monthly["month_name"] = monthly["month"].map(lambda m: month_labels[m - 1])
        fig = go.Figure(go.Scatter(
            x=monthly["month_name"],
            y=monthly[self.target_column],
            mode="lines+markers",
            line=dict(color=_COLOR_WARNING, width=3),
            marker=dict(size=10),
        ))
        fig.update_layout(
            title="Monthly Consumption Trend",
            xaxis_title="Month",
            yaxis_title="Avg Energy (kWh)",
            template=_DARK_TEMPLATE,
            height=350,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return self._to_json(fig)
