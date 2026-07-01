"""Compute-time benchmark of the data-intelligence gate: Polars vs pandas.

The Prefect ``check_data`` task is the data-quality + ingest gate of the pipeline
(read the raw CSVs, count nulls, apply the physical-range filter). It is written
with both pandas and Polars. This script
quantifies *why*: it runs the **same** quality gate with both engines over
increasingly large copies of the real datasets and times them, so we can show
how the runtime scales with the number of rows for each engine.

    .venv/Scripts/python.exe -m src.pipeline.benchmark

Outputs (in ``results/``):
    data_benchmark.csv  -- engine, scale, rows, time_ms_mean, time_ms_std
    data_benchmark.png  -- runtime vs. rows (Polars vs pandas) + speed-up panel
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

_ROOT = Path(__file__).resolve().parents[2]
DATA = _ROOT / "data"
RESULTS = _ROOT / "results"
SOLAR_CSV, WIND_CSV = DATA / "DatosSolares.csv", DATA / "DatosEolicos.csv"

SCALES = [1, 10, 25, 50, 100]   # row-count multipliers
REPEATS = 7
PETROL, TEAL = "#003B45", "#00ACBD"


# --------------------------------------------------------------------------- #
#  The quality gate, implemented identically in each engine                   #
# --------------------------------------------------------------------------- #
def gate_polars(solar_path: str, wind_path: str) -> int:
    """Polars lazy ``scan_csv -> filter -> collect`` gate + null counts."""
    solar = (pl.scan_csv(solar_path)
             .filter(pl.col("SystemProduction") >= 0)
             .collect())
    wind = (pl.scan_csv(wind_path)
            .filter((pl.col("Power") >= 0)
                    & pl.col("relativehumidity_2m").is_between(0, 100))
            .collect())
    _ = solar.null_count(), wind.null_count()
    return solar.height + wind.height


def gate_pandas(solar_path: str, wind_path: str) -> int:
    """Equivalent eager pandas gate: read -> boolean-mask filter + null counts."""
    solar = pd.read_csv(solar_path)
    solar = solar[solar["SystemProduction"] >= 0]
    wind = pd.read_csv(wind_path)
    wind = wind[(wind["Power"] >= 0)
                & wind["relativehumidity_2m"].between(0, 100)]
    _ = solar.isna().sum(), wind.isna().sum()
    return len(solar) + len(wind)


def _scaled_copy(df: pd.DataFrame, k: int, dst: Path) -> None:
    pd.concat([df] * k, ignore_index=True).to_csv(dst, index=False)


def _time(fn, *args, repeats=REPEATS) -> tuple[float, float]:
    fn(*args)  # warm-up (filesystem cache, lazy imports)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        samples.append((time.perf_counter() - t0) * 1000.0)  # ms
    samples.sort()
    trimmed = samples[:-1] if len(samples) > 3 else samples  # drop worst outlier
    mean = sum(trimmed) / len(trimmed)
    var = sum((s - mean) ** 2 for s in trimmed) / len(trimmed)
    return mean, var ** 0.5


def main() -> None:
    solar0 = pd.read_csv(SOLAR_CSV)
    wind0 = pd.read_csv(WIND_CSV)
    base_rows = len(solar0) + len(wind0)
    print(f"base rows: solar={len(solar0)}, wind={len(wind0)}, total={base_rows}")

    tmp = Path(tempfile.mkdtemp(prefix="microgrid_bench_"))
    rows, records = [], []
    try:
        for k in SCALES:
            sp, wp = tmp / f"solar_{k}.csv", tmp / f"wind_{k}.csv"
            _scaled_copy(solar0, k, sp)
            _scaled_copy(wind0, k, wp)
            n = base_rows * k
            pol_m, pol_s = _time(gate_polars, str(sp), str(wp))
            pan_m, pan_s = _time(gate_pandas, str(sp), str(wp))
            rows.append({"scale": k, "rows": n,
                         "polars_ms_mean": pol_m, "polars_ms_std": pol_s,
                         "pandas_ms_mean": pan_m, "pandas_ms_std": pan_s,
                         "speedup": pan_m / pol_m})
            print(f"  x{k:>3} ({n:>7} rows): polars {pol_m:7.1f}ms  "
                  f"pandas {pan_m:7.1f}ms  speed-up x{pan_m/pol_m:.2f}")
            records.extend([
                {"engine": "polars", "scale": k, "rows": n,
                 "time_ms_mean": pol_m, "time_ms_std": pol_s},
                {"engine": "pandas", "scale": k, "rows": n,
                 "time_ms_mean": pan_m, "time_ms_std": pan_s}])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(RESULTS / "data_benchmark.csv", index=False)

    # ---- figure ----------------------------------------------------------- #
    xs = [r["rows"] for r in rows]
    pol = [r["polars_ms_mean"] for r in rows]
    pol_e = [r["polars_ms_std"] for r in rows]
    pan = [r["pandas_ms_mean"] for r in rows]
    pan_e = [r["pandas_ms_std"] for r in rows]
    spd = [r["speedup"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.5), dpi=200)
    ax1.errorbar(xs, pan, yerr=pan_e, marker="s", color=PETROL, lw=2, capsize=3,
                 label="pandas (eager)")
    ax1.errorbar(xs, pol, yerr=pol_e, marker="o", color=TEAL, lw=2, capsize=3,
                 label="Polars (lazy scan)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("rows processed (solar + wind)")
    ax1.set_ylabel("data-quality gate time (ms)")
    ax1.set_title("Pipeline ingest + quality gate")
    ax1.grid(True, which="both", alpha=0.3); ax1.legend(frameon=False)

    ax2.plot(xs, spd, marker="D", color=TEAL, lw=2)
    ax2.axhline(1.0, color=PETROL, ls="--", lw=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("rows processed")
    ax2.set_ylabel("speed-up  (pandas / Polars)")
    ax2.set_title("Polars speed-up vs. pandas")
    ax2.grid(True, which="both", alpha=0.3)
    for x, s in zip(xs, spd):
        ax2.annotate(f"×{s:.1f}", (x, s), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=8, color=PETROL)

    fig.suptitle("Data-intelligence engine benchmark — Polars vs pandas",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULTS / "data_benchmark.png", dpi=200)
    print(f"saved {RESULTS/'data_benchmark.csv'} and data_benchmark.png")


if __name__ == "__main__":
    main()
