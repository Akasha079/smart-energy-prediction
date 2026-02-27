"""
Synthetic energy consumption data generator.

Produces realistic hourly energy consumption records that incorporate
time-of-day patterns, weather effects, building characteristics,
appliance usage, and weekend / holiday seasonality.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR, DATA_GENERATION

logger = logging.getLogger(__name__)


class EnergyDataGenerator:
    """Generates realistic synthetic energy consumption datasets."""

    # US federal holidays (month, day) -- simplified set
    HOLIDAYS = [
        (1, 1), (1, 15), (2, 19), (5, 27), (7, 4),
        (9, 2), (10, 14), (11, 11), (11, 28), (12, 25),
    ]

    def __init__(
        self,
        start_date: str = DATA_GENERATION["start_date"],
        end_date: str = DATA_GENERATION["end_date"],
        frequency: str = DATA_GENERATION["frequency"],
        num_buildings: int = DATA_GENERATION["num_buildings"],
        random_seed: int = DATA_GENERATION["random_seed"],
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.num_buildings = num_buildings
        self.rng = np.random.RandomState(random_seed)

    # ------------------------------------------------------------------
    # Time features
    # ------------------------------------------------------------------
    def _generate_time_features(self, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """Create calendar and temporal features from timestamps."""
        df = pd.DataFrame({"timestamp": timestamps})
        df["hour"] = timestamps.hour
        df["day_of_week"] = timestamps.dayofweek
        df["day_of_month"] = timestamps.day
        df["month"] = timestamps.month
        df["quarter"] = timestamps.quarter
        df["is_weekend"] = (timestamps.dayofweek >= 5).astype(int)
        df["is_holiday"] = df.apply(
            lambda r: int((r["timestamp"].month, r["timestamp"].day) in self.HOLIDAYS),
            axis=1,
        )
        df["season"] = df["month"].map(self._month_to_season)
        return df

    @staticmethod
    def _month_to_season(month: int) -> int:
        """Map month number to season (0=winter, 1=spring, 2=summer, 3=fall)."""
        if month in (12, 1, 2):
            return 0
        if month in (3, 4, 5):
            return 1
        if month in (6, 7, 8):
            return 2
        return 3

    # ------------------------------------------------------------------
    # Weather features
    # ------------------------------------------------------------------
    def _generate_weather_features(self, n: int, months: pd.Series) -> pd.DataFrame:
        """Simulate weather correlated with season and time."""
        base_temps = months.map({
            1: 2, 2: 4, 3: 10, 4: 15, 5: 20, 6: 27,
            7: 30, 8: 29, 9: 23, 10: 16, 11: 9, 12: 3,
        }).values.astype(float)

        temperature = base_temps + self.rng.normal(0, 3, n)
        humidity = np.clip(60 + self.rng.normal(0, 15, n) - 0.3 * temperature, 20, 100)
        wind_speed = np.clip(self.rng.exponential(8, n), 0, 50)
        cloud_cover = np.clip(self.rng.beta(2, 3, n) * 100, 0, 100)

        return pd.DataFrame({
            "temperature": np.round(temperature, 1),
            "humidity": np.round(humidity, 1),
            "wind_speed": np.round(wind_speed, 1),
            "cloud_cover": np.round(cloud_cover, 1),
        })

    # ------------------------------------------------------------------
    # Building features
    # ------------------------------------------------------------------
    def _generate_building_features(self, n: int, building_id: int) -> pd.DataFrame:
        """Generate static building characteristics (constant per building)."""
        self.rng.seed(building_id + 1000)  # reproducible per building
        sq_footage = self.rng.choice([5000, 10000, 20000, 50000, 100000])
        num_floors = max(1, int(sq_footage / 8000))
        building_age = self.rng.randint(1, 50)

        return pd.DataFrame({
            "building_id": building_id,
            "sq_footage": sq_footage,
            "num_floors": num_floors,
            "building_age": building_age,
        }, index=range(n))

    # ------------------------------------------------------------------
    # Occupancy and appliance usage
    # ------------------------------------------------------------------
    def _generate_occupancy(
        self, hours: np.ndarray, is_weekend: np.ndarray, n: int,
    ) -> np.ndarray:
        """Realistic occupancy that varies by hour and weekday/weekend."""
        weekday_curve = np.array([
            0.05, 0.03, 0.02, 0.02, 0.03, 0.08, 0.30, 0.70,
            0.90, 0.95, 0.95, 0.90, 0.80, 0.90, 0.95, 0.90,
            0.80, 0.60, 0.35, 0.20, 0.15, 0.10, 0.08, 0.06,
        ])
        weekend_curve = np.array([
            0.05, 0.03, 0.02, 0.02, 0.02, 0.03, 0.05, 0.08,
            0.12, 0.15, 0.15, 0.15, 0.12, 0.12, 0.12, 0.10,
            0.10, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05,
        ])
        occupancy = np.where(
            is_weekend,
            weekend_curve[hours],
            weekday_curve[hours],
        )
        noise = self.rng.normal(0, 0.05, n)
        return np.clip(occupancy + noise, 0, 1).round(3)

    def _generate_appliance_usage(
        self, hours: np.ndarray, temperature: np.ndarray,
        occupancy: np.ndarray, n: int,
    ) -> pd.DataFrame:
        """Simulate HVAC, lighting, and equipment usage."""
        # HVAC driven by temperature deviation from comfort (21 C)
        temp_deviation = np.abs(temperature - 21)
        hvac_usage = np.clip(
            temp_deviation / 20 + occupancy * 0.3 + self.rng.normal(0, 0.05, n),
            0, 1,
        )

        # Lighting higher during dark hours and occupancy
        dark_factor = np.where((hours >= 7) & (hours <= 18), 0.3, 0.8)
        lighting_usage = np.clip(
            dark_factor * occupancy + self.rng.normal(0, 0.05, n), 0, 1,
        )

        # Equipment correlated with occupancy
        equipment_usage = np.clip(
            occupancy * 0.8 + self.rng.normal(0, 0.08, n), 0, 1,
        )

        return pd.DataFrame({
            "hvac_usage": np.round(hvac_usage, 3),
            "lighting_usage": np.round(lighting_usage, 3),
            "equipment_usage": np.round(equipment_usage, 3),
        })

    # ------------------------------------------------------------------
    # Target variable
    # ------------------------------------------------------------------
    def _compute_energy_consumption(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute energy consumption (kWh) as a combination of all features
        with realistic coefficients and non-linear interactions.
        """
        base_load = 50  # kWh base load
        consumption = (
            base_load
            + df["sq_footage"].values * 0.001
            + df["hvac_usage"].values * 120
            + df["lighting_usage"].values * 40
            + df["equipment_usage"].values * 60
            + df["occupancy"].values * 80
            + np.abs(df["temperature"].values - 21) * 2.5
            + df["humidity"].values * 0.15
            + df["building_age"].values * 0.3
        )
        # Add multiplicative noise
        noise = self.rng.normal(1.0, 0.05, len(df))
        consumption = consumption * noise
        return np.round(np.clip(consumption, 10, None), 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate the full synthetic dataset for all buildings.

        Parameters
        ----------
        output_path : str, optional
            If provided, saves the dataset as CSV at this path.

        Returns
        -------
        pd.DataFrame
        """
        all_frames = []
        timestamps = pd.date_range(
            start=self.start_date, end=self.end_date, freq=self.frequency,
        )
        n = len(timestamps)

        for bid in range(self.num_buildings):
            logger.info("Generating data for building %d ...", bid)

            # Reset RNG per building for reproducibility
            self.rng = np.random.RandomState(DATA_GENERATION["random_seed"] + bid)

            time_df = self._generate_time_features(timestamps)
            weather_df = self._generate_weather_features(n, time_df["month"])
            building_df = self._generate_building_features(n, bid)

            hours = time_df["hour"].values
            is_weekend = time_df["is_weekend"].values
            occupancy = self._generate_occupancy(hours, is_weekend, n)

            appliance_df = self._generate_appliance_usage(
                hours, weather_df["temperature"].values, occupancy, n,
            )

            combined = pd.concat(
                [time_df, weather_df, building_df.reset_index(drop=True), appliance_df],
                axis=1,
            )
            combined["occupancy"] = occupancy
            combined["energy_consumption_kwh"] = self._compute_energy_consumption(combined)

            all_frames.append(combined)

        dataset = pd.concat(all_frames, ignore_index=True)
        dataset.sort_values("timestamp", inplace=True)
        dataset.reset_index(drop=True, inplace=True)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            dataset.to_csv(output_path, index=False)
            logger.info("Dataset saved to %s (%d rows)", output_path, len(dataset))

        return dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = EnergyDataGenerator()
    out = os.path.join(DATA_DIR, "energy_data.csv")
    df = gen.generate(output_path=out)
    print(f"Generated {len(df)} rows -> {out}")
