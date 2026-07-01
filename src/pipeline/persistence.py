"""Persistence layer for the microgrid pipeline.

Stores every pipeline run in a SQLite database (``results/microgrid.db`` by
default, overridable via the ``MICROGRID_DB_URL`` env var so Docker can point it
at a mounted volume or a Postgres service). Four tables keyed by ``run_id``:

  * ``runs``                 one row per pipeline execution (scenario + context)
  * ``predictions``          ML model outputs (solar/wind power forecasts) per run
  * ``optimization_metrics`` HV/GD/IGD/Epsilon per algorithm
  * ``mas_outcomes``         negotiated result per strategy combination

SQLAlchemy + pandas keep it portable: switching to Postgres only needs a
different URL, no code change.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = f"sqlite:///{(ROOT / 'results' / 'microgrid.db').as_posix()}"


def get_engine(url: str | None = None):
    url = url or os.environ.get("MICROGRID_DB_URL", DEFAULT_DB)
    if url.startswith("sqlite:///"):
        Path(url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url)


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S")


def save_run(run_id: str, cfg, engine=None) -> None:
    engine = engine or get_engine()
    row = pd.DataFrame([{
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": cfg.label,
        "demand_kw": cfg.demand,
        "gen_solar_kw": cfg.gen_solar,
        "gen_wind_kw": cfg.gen_wind,
        "gen_solar_raw_wh": cfg.meta.get("gen_solar_raw_Wh"),
        "gen_wind_raw_pct": cfg.meta.get("gen_wind_raw_pct"),
        "wind_rated_kw": cfg.meta.get("wind_rated_kw"),
    }])
    row.to_sql("runs", engine, if_exists="append", index=False)


def save_predictions(run_id: str, cfg, engine=None) -> None:
    engine = engine or get_engine()
    row = pd.DataFrame([{
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": cfg.label,
        "solar_pred_kw": cfg.gen_solar,
        "wind_pred_kw": cfg.gen_wind,
        "solar_raw_wh": cfg.meta.get("gen_solar_raw_Wh"),
        "wind_raw_pct": cfg.meta.get("gen_wind_raw_pct"),
        "wind_rated_kw": cfg.meta.get("wind_rated_kw"),
    }])
    row.to_sql("predictions", engine, if_exists="append", index=False)


def _add_missing_columns(table: str, df: pd.DataFrame, engine) -> None:
    """Append-friendly schema evolution: if ``df`` carries columns the existing
    table lacks (e.g. the ``Spread`` indicator added in Part II, or the MAS
    ``dominated_by_optimum``/``dist_to_pareto`` annotations), ALTER TABLE to add
    them instead of failing the insert. SQLite is dynamically typed, so a FLOAT
    declaration is harmless for the occasional text/bool column."""
    insp = inspect(engine)
    if not insp.has_table(table):
        return
    existing = {c["name"] for c in insp.get_columns(table)}
    with engine.begin() as conn:
        for col in df.columns:
            if col not in existing:
                conn.execute(text(f'ALTER TABLE {table} ADD COLUMN "{col}" FLOAT'))


def save_optimization(run_id: str, metrics: pd.DataFrame, engine=None) -> None:
    engine = engine or get_engine()
    df = metrics.copy()
    df.insert(0, "run_id", run_id)
    _add_missing_columns("optimization_metrics", df, engine)
    df.to_sql("optimization_metrics", engine, if_exists="append", index=False)


def save_mas(run_id: str, mas_df: pd.DataFrame, engine=None) -> None:
    engine = engine or get_engine()
    df = mas_df.copy()
    df.insert(0, "run_id", run_id)
    _add_missing_columns("mas_outcomes", df, engine)
    df.to_sql("mas_outcomes", engine, if_exists="append", index=False)


def read_table(name: str, engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    return pd.read_sql_table(name, engine)
