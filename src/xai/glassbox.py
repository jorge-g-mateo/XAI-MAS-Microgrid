"""X1 — Interpretable-by-design baselines + the Rashomon effect (xAI).

For each target (solar ``SystemProduction``, wind ``Power``) we fit two
**glass-box** models — a standardized linear regression (additive coefficients)
and a shallow decision tree (readable rules) — and compare them against the
supplied RandomForest on:

  (a) **accuracy** (R^2 / MAE / RMSE on a held-out split) → the
      performance-vs-transparency trade-off;
  (b) the **drivers** each model reports → standardized linear coefficients and
      the tree's importances vs. the RF's permutation importance. When models of
      comparable accuracy disagree on *why*, that is the **Rashomon effect**.

Honesty note: the RandomForests were supplied **pre-trained** on an unknown
split, so their held-out score here is optimistic (they likely saw these rows).
That makes the reported accuracy gap to the glass-box a *conservative* (upper)
bound on how much accuracy transparency costs — not an inflated one. The
glass-box models, by contrast, are trained only on the train split, so their
test score is a clean out-of-sample estimate.

Outputs (under ``results/xai/<model>/``):
  - glassbox_coeffs.png   standardized linear-regression coefficients
  - tree.png              the shallow decision tree (readable rules)
  - model_tradeoff.csv    R^2 / MAE / RMSE for RF vs. Linear vs. Tree (held-out)
  - rashomon.csv          top-k drivers per model, side by side (+ overlap)
  - surrogate.csv         X5: fidelity R^2 of a readable tree that mimics the RF

Usage:
    python -m src.xai.glassbox
    python -m src.xai.glassbox --model solar --depth 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree

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
OUT = ROOT / "results" / "xai"

# Per-model wiring: feature list, target column, data + model loaders.
SPEC = {
    "solar": (SOLAR_FEATURES, "SystemProduction", load_solar_data, load_solar_model),
    "wind": (WIND_FEATURES, "Power", load_wind_data, load_wind_model),
}


def _metrics(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "R2": round(float(r2_score(y, pred)), 4),
        "MAE": round(float(mean_absolute_error(y, pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, pred))), 3),
    }


def run_glassbox(name: str, depth: int = 4, test_size: float = 0.25,
                 pi_sample: int = 800, top_k: int = 5, seed: int = 42,
                 verbose: bool = True) -> dict:
    """Fit glass-box baselines for ``name`` and compare them with the RF.

    Returns a dict with the trade-off table and the Rashomon comparison.
    """
    features, target, load_data, load_model = SPEC[name]
    outdir = OUT / name
    outdir.mkdir(parents=True, exist_ok=True)

    df = add_date_features(load_data()).dropna(subset=features + [target])
    X, y = df[features], df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    rf = load_model()
    # Linear regression on standardized features → comparable additive coefficients.
    linear = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
    # Shallow tree on raw features → human-readable splits (no scaling needed).
    tree = DecisionTreeRegressor(max_depth=depth, random_state=seed).fit(X_tr, y_tr)

    # (a) Trade-off table, all evaluated on the SAME held-out split.
    tradeoff = pd.DataFrame(
        {
            "RandomForest (RF, black-box)": _metrics(rf, X_te, y_te),
            f"Linear (glass-box)": _metrics(linear, X_te, y_te),
            f"Tree depth {depth} (glass-box)": _metrics(tree, X_te, y_te),
        }
    ).T.reset_index(names="model")
    tradeoff.insert(0, "target", target)
    tradeoff.to_csv(outdir / "model_tradeoff.csv", index=False)

    # Standardized linear coefficients (importance comparable across features).
    coefs = pd.Series(linear.named_steps["linearregression"].coef_, index=features)
    coefs = coefs.reindex(coefs.abs().sort_values().index)  # ascending |coef| for barh
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(features) + 1.5))
    colors = ["tab:red" if c < 0 else "tab:blue" for c in coefs]
    ax.barh(coefs.index, coefs.values, color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("standardized coefficient (effect of +1 SD on the target)")
    r2_lin = tradeoff.set_index("model").loc["Linear (glass-box)", "R2"]
    ax.set_title(f"Linear glass-box coefficients — {name} model (test $R^2$={r2_lin})")
    fig.tight_layout()
    fig.savefig(outdir / "glassbox_coeffs.png", dpi=130)
    plt.close(fig)

    # Decision tree (readable rules).
    fig, ax = plt.subplots(figsize=(min(4 + 2 ** depth, 26), 4 + depth))
    plot_tree(tree, feature_names=features, filled=True, rounded=True,
              fontsize=8, precision=2, ax=ax, impurity=False)
    r2_tree = tradeoff.set_index("model").loc[f"Tree depth {depth} (glass-box)", "R2"]
    ax.set_title(f"Decision-tree glass-box (depth {depth}) — {name} model "
                 f"(test $R^2$={r2_tree})")
    fig.tight_layout()
    fig.savefig(outdir / "tree.png", dpi=120)
    plt.close(fig)

    # (c) X5 — global surrogate: a tree trained to MIMIC the RF's predictions
    # (not the true target). Its *fidelity* (how faithfully a readable tree
    # reproduces the black box) is a different question from train-on-y accuracy:
    # high fidelity → the RF is approximable by simple rules; low fidelity → its
    # extra accuracy genuinely comes from structure no shallow tree captures.
    surrogate = DecisionTreeRegressor(max_depth=depth + 2, random_state=seed).fit(
        X_tr, rf.predict(X_tr))
    fidelity = round(float(r2_score(rf.predict(X_te), surrogate.predict(X_te))), 4)
    pd.DataFrame([{"target": target, "surrogate_tree_depth": depth + 2,
                   "fidelity_R2_to_RF": fidelity}]).to_csv(
        outdir / "surrogate.csv", index=False)

    # (b) Rashomon: do comparable models agree on the drivers?
    Xpi = X_te.sample(min(pi_sample, len(X_te)), random_state=seed)
    ypi = y_te.loc[Xpi.index]
    pi = permutation_importance(rf, Xpi, ypi, n_repeats=5, random_state=seed)
    rf_rank = pd.Series(pi.importances_mean, index=features).sort_values(ascending=False)
    lin_rank = coefs.abs().sort_values(ascending=False)
    tree_rank = pd.Series(tree.feature_importances_, index=features).sort_values(ascending=False)

    rf_top, lin_top, tree_top = (list(r.head(top_k).index)
                                 for r in (rf_rank, lin_rank, tree_rank))
    overlap = len(set(rf_top) & set(lin_top))
    rashomon = pd.DataFrame({
        "rank": range(1, top_k + 1),
        "RF_perm_importance": rf_top,
        "Linear_|coef|": lin_top,
        "Tree_importance": tree_top,
    })
    rashomon.to_csv(outdir / "rashomon.csv", index=False)

    if verbose:
        print(f"\n===== {name.upper()} -- glass-box vs RF =====")
        print(tradeoff.to_string(index=False))
        print(f"Top-{top_k} drivers   RF={rf_top}")
        print(f"                   Linear={lin_top}")
        print(f"                     Tree={tree_top}")
        verdict = ("similar feature ranking (gap is functional form, not drivers)"
                   if overlap >= top_k - 1
                   else "Rashomon: same data, different explanation")
        print(f"RF vs Linear top-{top_k} overlap: {overlap}/{top_k} -> {verdict}")
        print(f"Global surrogate: depth-{depth + 2} tree mimics the RF with "
              f"fidelity R2={fidelity} (vs the RF's own true-target R2 "
              f"{tradeoff.set_index('model').loc['RandomForest (RF, black-box)', 'R2']})")
        print(f"Artifacts -> {outdir}")

    return {"tradeoff": tradeoff, "rashomon": rashomon,
            "rf_top": rf_top, "lin_top": lin_top, "tree_top": tree_top,
            "overlap": overlap, "k": top_k, "surrogate_fidelity": fidelity}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["solar", "wind", "both"], default="both")
    ap.add_argument("--depth", type=int, default=4, help="max depth of the glass-box tree")
    args = ap.parse_args()

    names = ["solar", "wind"] if args.model == "both" else [args.model]
    for n in names:
        run_glassbox(n, depth=args.depth)


if __name__ == "__main__":
    main()
