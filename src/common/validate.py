"""Predictive-model validation for the microgrid forecasting models.

The solar and wind random forests are used everywhere as the generators' capacity
estimators, so their accuracy underpins the whole system. This module quantifies
how well they fit the provided data: R^2, MAE and RMSE for each model, plus a
predicted-vs-actual scatter.

Honesty note: the models were supplied **pre-trained** on an unspecified split, so
we evaluate on the full provided dataset. These figures therefore characterize the
*fit quality* of the delivered models on the available data, not a clean
out-of-sample generalization estimate.

Outputs (under ``results/validation/``):
  - metrics.csv                model x {R2, MAE, RMSE, n, target range}
  - predicted_vs_actual.png    two-panel scatter (solar, wind) with the y=x line

Usage:
    python -m src.common.validate
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common.inference import (
    SOLAR_FEATURES,
    WIND_FEATURES,
    add_date_features,
    load_solar_data,
    load_solar_model,
    load_wind_data,
    load_wind_model,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "validation"


def _evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    pred = model.predict(X)
    return {
        "R2": r2_score(y, pred),
        "MAE": mean_absolute_error(y, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "n": int(len(y)),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "_pred": pred,
        "_true": y.to_numpy(),
    }


def _error_by_regime(res: dict):
    """Where do the models err? MAE per decile of the actual target + a residual
    plot. Honest diagnostic of which operating regimes the models fit worst."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    regime_rows = []
    for col, (name, r) in enumerate(res.items()):
        t, p = r["_true"], r["_pred"]
        resid = p - t

        # MAE per decile of the actual value
        order = np.argsort(t)
        bins = np.array_split(order, 10)
        centers, maes = [], []
        for k, b in enumerate(bins):
            if len(b) == 0:
                continue
            centers.append(float(t[b].mean()))
            maes.append(float(np.abs(resid[b]).mean()))
            regime_rows.append({"model": name, "decile": k + 1,
                                "actual_mean": round(centers[-1], 2),
                                "MAE": round(maes[-1], 3)})
        ax_bar = axes[0, col]
        ax_bar.bar(range(1, len(maes) + 1), maes, color="tab:purple", alpha=0.7)
        ax_bar.set_xlabel("decile of actual target (low→high)")
        ax_bar.set_ylabel("MAE")
        ax_bar.set_title(f"{name}: MAE by production regime")
        ax_bar.grid(True, axis="y", alpha=0.3)

        ax_res = axes[1, col]
        ax_res.scatter(p, resid, s=4, alpha=0.2, color="tab:green")
        ax_res.axhline(0, color="red", lw=1)
        ax_res.set_xlabel("predicted"); ax_res.set_ylabel("residual (pred − actual)")
        ax_res.set_title(f"{name}: residuals")
        ax_res.grid(True, alpha=0.3)

    fig.suptitle("Error by regime and residual structure")
    fig.tight_layout()
    fig.savefig(OUT / "error_by_regime.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(regime_rows).to_csv(OUT / "error_by_regime.csv", index=False)


def validate() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)

    solar = add_date_features(load_solar_data()).dropna(subset=SOLAR_FEATURES + ["SystemProduction"])
    wind = add_date_features(load_wind_data()).dropna(subset=WIND_FEATURES + ["Power"])

    res = {
        "solar": _evaluate(load_solar_model(), solar[SOLAR_FEATURES], solar["SystemProduction"]),
        "wind": _evaluate(load_wind_model(), wind[WIND_FEATURES], wind["Power"]),
    }

    # --- metrics table ---
    rows = []
    for name, r in res.items():
        rows.append({"model": name, "target": "SystemProduction" if name == "solar" else "Power",
                     "R2": round(r["R2"], 4), "MAE": round(r["MAE"], 3),
                     "RMSE": round(r["RMSE"], 3), "n": r["n"],
                     "target_range": f"[{r['target_min']:.1f}, {r['target_max']:.1f}]"})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)

    # --- predicted vs actual scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (name, r) in zip(axes, res.items()):
        t, p = r["_true"], r["_pred"]
        ax.scatter(t, p, s=4, alpha=0.2, color="tab:blue")
        lo, hi = float(min(t.min(), p.min())), float(max(t.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="ideal $y=x$")
        ax.set_xlabel(f"actual ({'SystemProduction' if name=='solar' else 'Power'})")
        ax.set_ylabel("predicted")
        ax.set_title(f"{name} model — $R^2$={r['R2']:.3f}, MAE={r['MAE']:.2f}")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Predictive-model fit on the provided data (in-sample)")
    fig.tight_layout()
    fig.savefig(OUT / "predicted_vs_actual.png", dpi=130)
    plt.close(fig)

    _error_by_regime(res)
    return metrics


def main():
    metrics = validate()
    print(metrics.to_string(index=False))
    print(f"\nOutputs -> {OUT}/")


if __name__ == "__main__":
    main()
