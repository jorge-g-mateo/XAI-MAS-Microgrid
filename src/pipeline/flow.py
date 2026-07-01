"""Prefect-orchestrated pipeline for the microgrid platform.

Wires the four layers into one monitored, retriable workflow:

    ingest_forecast -> optimize -> negotiate -> persist

Each module is wrapped as a Prefect ``@task`` so the whole workflow runs
(and is observable in the Prefect UI) as a single ``@flow``. Inference
results, optimization metrics and negotiation outcomes are persisted to SQLite
(see :mod:`src.pipeline.persistence`).

Usage:
    python -m src.pipeline.flow --timestamp "2017-06-15 19:00:00"
    python -m src.pipeline.flow --row 4000 --runs 2 --evals 6000 --rounds 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Make local runs self-contained: use a project-local Prefect home and allow the
# ephemeral API, so `python -m src.pipeline.flow` works with no server and no
# global profile interfering. In Docker, PREFECT_API_URL is set explicitly and
# these setdefaults don't override it.
_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PREFECT_HOME", str(_ROOT / ".prefect_home"))
os.environ.setdefault("PREFECT_SERVER_ALLOW_EPHEMERAL_MODE", "true")

import numpy as np
import pandas as pd
import polars as pl
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact

from src.mas.buyer_strategies import BUYER_STRATEGIES
from src.mas.run_mas import _dominated_by_front
from src.mas.simulation import run_negotiation
from src.mas.strategies import STRATEGIES
from src.optimization.market import scenario_from_timestamp

DATA_DIR = _ROOT / "data"
SOLAR_RAW = ["WindSpeed", "Sunshine", "AirPressure", "Radiation",
             "RelativeAirHumidity", "Date", "SystemProduction"]
WIND_RAW = ["temperature_2m", "relativehumidity_2m", "dewpoint_2m",
            "windspeed_10m", "windspeed_100m", "winddirection_10m",
            "winddirection_100m", "windgusts_10m", "Date", "Power"]
from src.optimization.problem import decode_objectives
from src.optimization.run_optimization import run as run_optimization
from src.pipeline import persistence
from src.xai.explain import build_solar_explainer, build_wind_explainer
from src.xai.run_xai import explain_one


@task
def check_data():
    """Validate the raw datasets with **Polars** before inference.

    Data Intelligence uses Polars as its standard dataframe engine, so the
    data-quality gate follows the lazy ``scan_csv -> filter -> collect`` idiom:
    we check the schema (required columns), count nulls with ``null_count`` and
    apply a physical-range filter (production/power non-negative, humidity in
    [0,100]); rows that fall outside are surfaced and a missing column fails the
    pipeline loudly. Only the meteorological + Date + target columns are checked;
    hour/dayofweek/dayofyear are derived later by the inference layer.
    """
    log = get_run_logger()
    issues = []
    solar_csv, wind_csv = DATA_DIR / "DatosSolares.csv", DATA_DIR / "DatosEolicos.csv"

    solar = pl.read_csv(solar_csv)
    wind = pl.read_csv(wind_csv)

    # 1) Schema: required raw columns must be present.
    missing_s = [c for c in SOLAR_RAW if c not in solar.columns]
    missing_w = [c for c in WIND_RAW if c not in wind.columns]
    if missing_s:
        issues.append(f"Solar missing columns: {missing_s}")
    if missing_w:
        issues.append(f"Wind missing columns: {missing_w}")

    # 2) Null counts (warn) via Polars null_count.
    if not missing_s:
        nan_s = {c: solar[c].null_count() for c in SOLAR_RAW
                 if c != "Date" and solar[c].null_count() > 0}
        if nan_s:
            log.warning(f"Solar null counts: {nan_s}")
    if not missing_w:
        nan_w = {c: wind[c].null_count() for c in WIND_RAW
                 if c != "Date" and wind[c].null_count() > 0}
        if nan_w:
            log.warning(f"Wind null counts: {nan_w}")

    # 3) Physical-range gate: lazy scan -> filter -> collect; report dropped rows.
    valid_solar = valid_wind = 0
    if not missing_s:
        valid_solar = (pl.scan_csv(solar_csv)
                       .filter(pl.col("SystemProduction") >= 0)
                       .collect().height)
        if valid_solar < solar.height:
            log.warning(f"Solar: {solar.height - valid_solar} rows fail the range gate "
                        "(negative production)")
    if not missing_w:
        valid_wind = (pl.scan_csv(wind_csv)
                      .filter((pl.col("Power") >= 0)
                              & pl.col("relativehumidity_2m").is_between(0, 100))
                      .collect().height)
        if valid_wind < wind.height:
            log.warning(f"Wind: {wind.height - valid_wind} rows fail the range gate "
                        "(negative power or humidity out of [0,100])")

    if issues:
        raise ValueError(f"Data quality check failed — {issues}")

    log.info(f"Data quality OK (Polars) — solar rows={solar.height} (valid {valid_solar}), "
             f"wind rows={wind.height} (valid {valid_wind})")
    return {"solar_rows": solar.height, "wind_rows": wind.height,
            "solar_valid": valid_solar, "wind_valid": valid_wind}


@task
def explain(sample: int = 300):
    """Run the xAI report for both power models (SHAP, PFI, PDP, LIME, what-if).

    The report explains the model's general behaviour on a sample of the
    dataset — it does not depend on the scenario/timestamp of this run, so it
    is skipped if already on disk (same cache check as the dashboard's XAI
    tab). Delete results/xai/ to force a fresh report (e.g. after retraining
    the models or wanting a different sample).
    """
    log = get_run_logger()
    xai_root = _ROOT / "results" / "xai"
    needs_solar = not (xai_root / "solar" / "shap_beeswarm.png").exists()
    needs_wind = not (xai_root / "wind" / "shap_beeswarm.png").exists()

    if needs_solar:
        explain_one(build_solar_explainer(sample), verbose=False)
    else:
        log.info("Solar xAI report already on disk, skipping")

    if needs_wind:
        explain_one(build_wind_explainer(sample), verbose=False)
    else:
        log.info("Wind xAI report already on disk, skipping")

    log.info(f"xAI report ready (sample={sample}) → results/xai/")


@task(retries=2, retry_delay_seconds=5)
def ingest_forecast(timestamp, demand):
    """Load the scenario and run the predictive models (generation forecast)."""
    cfg = scenario_from_timestamp(timestamp, demand=demand)
    get_run_logger().info(
        f"Scenario {cfg.label}: demand={cfg.demand:.2f} "
        f"solar={cfg.gen_solar:.2f} wind={cfg.gen_wind:.2f} kW")
    return cfg


@task
def optimize(cfg, runs, evals):
    """Multi-objective optimization -> metrics + Pareto reference front."""
    superfronts, metrics, combined = run_optimization(cfg, runs=runs, evals=evals)
    front = np.array([decode_objectives(s.objectives) for s in combined])
    get_run_logger().info(f"Pareto front: {len(front)} solutions")
    return metrics, front


@task
def negotiate(cfg, front, rounds):
    """Cross every (solar x wind x buyer) strategy combination and annotate
    each outcome vs. the optimum.

    Wires in the consumer-side negotiation (buyer strategies, M8/M9) that
    already exists in ``src.mas`` — this task only orchestrates the existing
    ``run_negotiation`` over the existing strategy sets, no MAS logic changes.
    """
    seller_names = list(STRATEGIES)
    buyer_names = list(BUYER_STRATEGIES)
    rows = []
    for b in buyer_names:
        for s in seller_names:
            for w in seller_names:
                res = run_negotiation(cfg, s, w, buyer_strategy=b, rounds=rounds, log=False)
                rows.append({
                    "solar_strategy": s, "wind_strategy": w, "buyer_strategy": b,
                    "profit_solar": res.profit_solar, "profit_wind": res.profit_wind,
                    "buyer_cost": res.buyer_cost, "shortfall": res.shortfall,
                    "price_solar": res.final_price_solar, "price_wind": res.final_price_wind,
                })
    df = pd.DataFrame(rows)
    dom, dist = [], []
    for _, r in df.iterrows():
        d, dd = _dominated_by_front((r.profit_solar, r.profit_wind, r.buyer_cost), front)
        dom.append(d); dist.append(round(dd, 3))
    df["dominated_by_optimum"] = dom
    df["dist_to_pareto"] = dist
    get_run_logger().info(f"Evaluated {len(df)} strategy combinations")
    return df


@task
def persist(cfg, metrics, mas_df):
    """Store scenario, predictions, optimization metrics and negotiation outcomes.
    Creates a Markdown artifact visible in the Prefect UI for each run.
    """
    run_id = persistence.new_run_id()
    eng = persistence.get_engine()
    persistence.save_run(run_id, cfg, eng)
    persistence.save_predictions(run_id, cfg, eng)
    persistence.save_optimization(run_id, metrics, eng)
    persistence.save_mas(run_id, mas_df, eng)

    # Build markdown summary visible in the Prefect UI (Artifacts tab).
    best = mas_df.loc[mas_df["profit_solar"].idxmax()]
    metrics_md = "\n".join(
        f"| {r['Algorithm']} | {r['HV']:.4f} | {r['IGD']:.4f} | {r['Epsilon']:.4f} |"
        for _, r in metrics.iterrows()
    )
    dominated_pct = int(mas_df["dominated_by_optimum"].mean() * 100)
    md = f"""## Run `{run_id}`

### Escenario energético
| Campo | Valor |
|---|---|
| Timestamp | `{cfg.label}` |
| Demanda | {cfg.demand:.2f} kW |
| Generación solar (pred.) | {cfg.gen_solar:.2f} kW |
| Generación eólica (pred.) | {cfg.gen_wind:.2f} kW |

### Optimización multi-objetivo
| Algoritmo | HV | IGD | Epsilon |
|---|---|---|---|
{metrics_md}

### Negociación MAS (vendedores x comprador)
| Campo | Valor |
|---|---|
| Combinaciones evaluadas | {len(mas_df)} |
| Dominadas por el óptimo | {dominated_pct}% |
| Mejor beneficio solar | {best.profit_solar:.3f} € (estrategia: `{best.solar_strategy}`, comprador: `{best.buyer_strategy}`) |
| Mejor beneficio eólico | {mas_df.loc[mas_df['profit_wind'].idxmax(), 'profit_wind']:.3f} € |
| Menor coste consumidor | {mas_df['buyer_cost'].min():.3f} € (comprador: `{mas_df.loc[mas_df['buyer_cost'].idxmin(), 'buyer_strategy']}`) |
"""
    create_markdown_artifact(markdown=md, key=run_id.lower(), description=f"Resumen del run {run_id}")
    get_run_logger().info(f"Persisted run {run_id}")
    return run_id


@flow(name="microgrid-pipeline")
def microgrid_pipeline(timestamp="2017-06-15 19:00:00", demand=None,
                       runs=2, evals=6000, rounds=10, xai_sample=300):
    check_data()
    cfg = ingest_forecast(timestamp, demand)
    metrics, front = optimize(cfg, runs, evals)
    mas_df = negotiate(cfg, front, rounds)
    run_id = persist(cfg, metrics, mas_df)
    explain(xai_sample)
    return run_id


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--timestamp", type=str)
    g.add_argument("--row", type=int)
    ap.add_argument("--demand", type=float, default=None)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--evals", type=int, default=6000)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--xai-sample", type=int, default=300)
    args = ap.parse_args()

    ts = args.timestamp if args.timestamp is not None else (
        args.row if args.row is not None else "2017-06-15 19:00:00")
    run_id = microgrid_pipeline(timestamp=ts, demand=args.demand,
                                runs=args.runs, evals=args.evals, rounds=args.rounds,
                                xai_sample=args.xai_sample)
    print(f"\nPipeline finished. run_id={run_id}")
    df = persistence.read_table("mas_outcomes")
    print(f"mas_outcomes rows in DB: {len(df)}")


if __name__ == "__main__":
    main()
