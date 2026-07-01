"""Shared inference layer for the microgrid energy platform.

This is the single source of truth for turning raw CSV rows into power
predictions using the pre-trained models in ``models/``. Every other module
(multi-agent system, optimization, xAI) must consume the microgrid power
forecasts through this module so that feature engineering stays consistent.

Key fact discovered during Phase 0: the provided models were trained with the
``Date`` column expanded into ``hour``, ``dayofweek`` and ``dayofyear`` features
(placed *after* the meteorological columns). The exact feature order required by
each model is hard-coded below from ``feature_names_in_`` and validated at load
time.

Models were serialized with scikit-learn 1.8.0 — pin that version (see
``requirements.txt``) to avoid silently invalid predictions.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

# Repo root = three levels up from this file (src/common/inference.py).
ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

WIND_MODEL_PATH = MODELS_DIR / "modelo_eolico.pkl"
SOLAR_MODEL_PATH = MODELS_DIR / "modelo_solar.pkl"

# Exact feature order expected by each model (from feature_names_in_).
WIND_FEATURES = [
    "temperature_2m",
    "relativehumidity_2m",
    "dewpoint_2m",
    "windspeed_10m",
    "windspeed_100m",
    "winddirection_10m",
    "winddirection_100m",
    "windgusts_10m",
    "hour",
    "dayofweek",
    "dayofyear",
]
SOLAR_FEATURES = [
    "WindSpeed",
    "Sunshine",
    "AirPressure",
    "Radiation",
    "RelativeAirHumidity",
    "hour",
    "dayofweek",
    "dayofyear",
]

# Date-derived features both models share.
DATE_FEATURES = ["hour", "dayofweek", "dayofyear"]


@lru_cache(maxsize=2)
def _load(path_str: str):
    """Load and cache a pickled model by its (string) path."""
    return joblib.load(path_str)


def load_wind_model():
    """Return the pre-trained wind power RandomForestRegressor."""
    return _load(str(WIND_MODEL_PATH))


def load_solar_model():
    """Return the pre-trained solar power RandomForestRegressor."""
    return _load(str(SOLAR_MODEL_PATH))


def add_date_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Return a copy of ``df`` with ``hour``/``dayofweek``/``dayofyear`` added.

    The ``date_col`` is parsed to datetime if it is not already.
    """
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["dayofyear"] = dt.dt.dayofyear
    return out


def _prepare(df: pd.DataFrame, features: list[str], date_col: str) -> pd.DataFrame:
    """Build the model-ready feature frame in the exact required column order."""
    needs_dates = any(f in DATE_FEATURES for f in features)
    if needs_dates and not set(DATE_FEATURES).issubset(df.columns):
        if date_col not in df.columns:
            raise KeyError(
                f"Column '{date_col}' required to derive {DATE_FEATURES}, "
                f"but it is missing and the features are not pre-computed."
            )
        df = add_date_features(df, date_col=date_col)

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    return df[features]


def predict_wind(df: pd.DataFrame, date_col: str = "Date"):
    """Predict wind turbine power for each row of ``df``.

    ``df`` must contain the meteorological columns of DatosEolicos.csv plus a
    ``Date`` column (or the pre-computed date features). Returns a numpy array.
    """
    X = _prepare(df, WIND_FEATURES, date_col)
    return load_wind_model().predict(X)


def predict_solar(df: pd.DataFrame, date_col: str = "Date"):
    """Predict PV system production for each row of ``df``.

    ``df`` must contain the meteorological columns of DatosSolares.csv plus a
    ``Date`` column (or the pre-computed date features). Returns a numpy array.
    """
    X = _prepare(df, SOLAR_FEATURES, date_col)
    return load_solar_model().predict(X)


def load_wind_data() -> pd.DataFrame:
    """Load the raw wind dataset (DatosEolicos.csv)."""
    return pd.read_csv(DATA_DIR / "DatosEolicos.csv")


def load_solar_data() -> pd.DataFrame:
    """Load the raw solar dataset (DatosSolares.csv)."""
    return pd.read_csv(DATA_DIR / "DatosSolares.csv")


if __name__ == "__main__":
    # Smoke test: predict the first rows of each dataset and compare with the
    # ground-truth target columns.
    wind = load_wind_data().head(5)
    solar = load_solar_data().head(5)

    wind_pred = predict_wind(wind)
    solar_pred = predict_solar(solar)

    print("WIND  | pred vs real (Power):")
    for p, r in zip(wind_pred, wind["Power"]):
        print(f"   pred={p:8.3f}   real={r:8.3f}")

    print("SOLAR | pred vs real (SystemProduction):")
    for p, r in zip(solar_pred, solar["SystemProduction"]):
        print(f"   pred={p:8.3f}   real={r:8.3f}")
