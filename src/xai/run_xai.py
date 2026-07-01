"""Generate the full xAI report for both the solar and the wind models.

Outputs (under ``results/xai/<model>/``):
  - permutation_importance.{csv,png}   global importance
  - shap_beeswarm.png                  global SHAP summary
  - pdp_ice.png                        PDP + ICE for the top features
  - pdp_vs_ale_<feature>.png           PDP vs. ALE (correlation diagnostic)
  - h_statistic.{csv,png}              Friedman H-statistic interaction strength
  - pdp2d_<a>_<b>.png                  2-D PDP of the strongest interaction pair (X3)
  - counterfactual_<feature>.png, counterfactuals.csv   contrastive explanation (X2)
  - waterfall_{high,medium,low}.png    3 local SHAP explanations
  - lime_{high,medium,low}.png         3 local LIME explanations
  - whatif_<feature>.png               what-if / ceteris-paribus curves
  - glassbox_coeffs.png, tree.png      interpretable-by-design baselines (X1)
  - model_tradeoff.csv, rashomon.csv   performance/transparency trade-off + Rashomon

Usage:
    python -m src.xai.run_xai
    python -m src.xai.run_xai --sample 1500 --model solar
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.xai.explain import build_solar_explainer, build_wind_explainer
from src.xai.glassbox import run_glassbox

# Features to highlight in PDP / what-if per model (physically meaningful drivers).
PDP_FEATURES = {
    "solar": ["Radiation", "Sunshine"],
    "wind": ["windspeed_100m", "windspeed_10m"],
}
WHATIF_FEATURES = {
    "solar": ["Radiation", "Sunshine"],
    "wind": ["windspeed_100m", "windgusts_10m"],
}


def explain_one(exp, verbose=True):
    print(f"\n===== {exp.name.upper()} MODEL =====")
    pi = exp.permutation_importance()
    if verbose:
        print("Top permutation importance:")
        print(pi.head(5).to_string(index=False))

    exp.plot_beeswarm()
    exp.plot_pdp(PDP_FEATURES[exp.name])

    # PDP vs. ALE for the top driver (correlation diagnostic) + ALE of top-3.
    top_feats = list(pi.feature.head(3))
    exp.plot_pdp_vs_ale(top_feats[0])
    exp.plot_ale_grid(top_feats)
    exp.plot_shap_dependence(top_feats[0])
    print(f"PDP-vs-ALE + ALE grid + SHAP dependence for: {top_feats}")

    inter = exp.h_statistic()
    # report the strongest interaction pair
    stacked = inter.stack()
    a, b = stacked.idxmax()
    print(f"Strongest interaction (H-stat): {a} x {b}  ({stacked.max():.3f})")
    # X3 — visualize that strongest pair with a 2-D PDP (measure→visualize).
    exp.plot_pdp_2d(a, b)

    instances = exp.plot_waterfalls()
    print(f"Local instances explained (SHAP): {instances}")

    exp.lime_explain(instances)
    agree = exp.local_method_agreement(instances["high"])
    print(f"Local instances explained (LIME): {list(instances)}")
    print(f"SHAP/LIME top-{agree['k']} overlap (high instance): "
          f"{agree['overlap']}/{agree['k']}  SHAP={agree['shap_top']}  LIME={agree['lime_top']}")

    for feat in WHATIF_FEATURES[exp.name]:
        exp.what_if(feat)
    print(f"What-if curves: {WHATIF_FEATURES[exp.name]}")

    # X2 — counterfactual (contrastive "why not Y?") on the top driver.
    cf = exp.counterfactual(top_feats[0])
    pd.DataFrame([cf]).to_csv(exp.outdir / "counterfactuals.csv", index=False)
    print(f"Counterfactual ({top_feats[0]}): {cf}")

    # X1 — interpretable-by-design baselines + Rashomon (independent train/test split).
    run_glassbox(exp.name, verbose=verbose)
    print(f"Artifacts -> {exp.outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=2000,
                    help="rows used for SHAP/PDP (speed vs. fidelity)")
    ap.add_argument("--model", choices=["solar", "wind", "both"], default="both")
    args = ap.parse_args()

    if args.model in ("solar", "both"):
        explain_one(build_solar_explainer(args.sample))
    if args.model in ("wind", "both"):
        explain_one(build_wind_explainer(args.sample))


if __name__ == "__main__":
    main()
