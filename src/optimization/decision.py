"""Pick ONE operating solution from the Pareto front (decision making).

The optimization uses post-hoc preference articulation: first approximate the
whole Pareto front and only then choose the
decision maker (the microgrid operator) chooses the solution to deploy. This
module implements the two standard selection mechanisms:

  * **Knee point** — normalize the objectives, take the extreme (best-per-
    objective) points of the front, and pick the solution with the largest
    orthogonal distance from the hyperplane through those extremes, on the
    ideal-point side. The knee is where giving up a little on one objective
    buys the most on the others, i.e. the natural "no-preference" compromise.
  * **Weighted Tschebycheff scalarization** — for an explicit stakeholder
    profile w, choose argmin over the front of  max_i w_i |f_i - z*_i| (z* =
    ideal point, objectives normalized). Larger weight = that objective is
    forced closer to its ideal. Profiles below: neutral, consumer-first and
    sellers-first.

If the MAS strategy-comparison results exist, the selected points are also
compared against the negotiated honest/honest outcome, closing the
optimization <-> negotiation loop with a concrete operating decision.

Outputs (results/): opt_decision.csv, opt_decision.png

Usage:
    python -m src.optimization.decision        # reads results/reference_front.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

OBJ_COLS = ["Profit Solar (AS)", "Profit Wind (AE)", "Buyer Cost (AC)"]
VAR_COLS = ["q_solar", "q_wind", "p_solar", "p_wind"]

# Stakeholder profiles for the Tschebycheff scalarization (weights over
# [profit AS, profit AE, cost AC], higher = more important).
PROFILES = {
    "tcheby_neutral": np.array([1 / 3, 1 / 3, 1 / 3]),
    "tcheby_consumer_first": np.array([0.2, 0.2, 0.6]),
    "tcheby_sellers_first": np.array([0.4, 0.4, 0.2]),
}


def _min_space(df: pd.DataFrame) -> np.ndarray:
    """Objectives as an all-minimization matrix, normalized to [0,1] per
    column over the front (ideal point -> origin)."""
    f = np.column_stack([-df[OBJ_COLS[0]], -df[OBJ_COLS[1]], df[OBJ_COLS[2]]])
    span = f.max(axis=0) - f.min(axis=0)
    span[span == 0] = 1.0
    return (f - f.min(axis=0)) / span


def knee_point(df: pd.DataFrame) -> int:
    """Index of the knee: max orthogonal distance below the hyperplane that
    passes through the m extreme (best-per-objective) points of the
    normalized front."""
    fn = _min_space(df)
    extremes = fn[[int(np.argmin(fn[:, k])) for k in range(fn.shape[1])]]
    # Hyperplane through the extremes: solve E n = 1 (then n.x = 1 on it).
    try:
        n = np.linalg.solve(extremes, np.ones(3))
    except np.linalg.LinAlgError:           # degenerate front -> fall back to
        d = np.linalg.norm(fn, axis=1)      # closest-to-ideal compromise
        return int(np.argmin(d))
    dist = (1.0 - fn @ n) / np.linalg.norm(n)   # >0 on the ideal-point side
    return int(np.argmax(dist))


def tschebycheff_pick(df: pd.DataFrame, weights: np.ndarray) -> int:
    """Index of the front solution minimizing the weighted Tschebycheff
    distance to the ideal point (normalized objective space)."""
    fn = _min_space(df)                     # ideal point is the origin
    return int(np.argmin((fn * weights).max(axis=1)))


def select(front_csv: Path = RESULTS / "reference_front.csv") -> pd.DataFrame:
    df = pd.read_csv(front_csv)
    picks = {"knee": knee_point(df)}
    for name, w in PROFILES.items():
        picks[name] = tschebycheff_pick(df, w)

    rows = []
    for method, idx in picks.items():
        row = {"method": method, **df.loc[idx, VAR_COLS + OBJ_COLS].to_dict()}
        rows.append(row)
    out = pd.DataFrame(rows)

    # Optional: distance from each pick to the negotiated honest/honest point.
    mas_csv = RESULTS / "mas_strategy_comparison.csv"
    if mas_csv.is_file():
        mas = pd.read_csv(mas_csv)
        hh = mas[(mas.solar_strategy == "honest") & (mas.wind_strategy == "honest")]
        if not hh.empty:
            target = hh.iloc[0][["profit_solar", "profit_wind", "buyer_cost"]].to_numpy(float)
            span = df[OBJ_COLS].max().to_numpy() - df[OBJ_COLS].min().to_numpy()
            span[span == 0] = 1.0
            sel = out[OBJ_COLS].to_numpy(float)
            out["dist_to_mas_honest"] = np.linalg.norm((sel - target) / span, axis=1).round(4)
    return out


def save_outputs(df_front: pd.DataFrame, decision: pd.DataFrame, suffix: str = ""):
    RESULTS.mkdir(exist_ok=True)
    decision.to_csv(RESULTS / f"opt_decision{suffix}.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df_front[OBJ_COLS[0]], df_front[OBJ_COLS[1]],
                    c=df_front[OBJ_COLS[2]], cmap="coolwarm", s=25, alpha=0.7)
    fig.colorbar(sc, ax=ax, label="Buyer Cost (AC) — minimize")
    style = {"knee": ("*", 380, "black", "knee point"),
             "tcheby_neutral": ("P", 200, "tab:green", "Tscheby. neutral"),
             "tcheby_consumer_first": ("X", 200, "tab:blue", "Tscheby. consumer-first"),
             "tcheby_sellers_first": ("^", 200, "tab:orange", "Tscheby. sellers-first")}
    for _, row in decision.iterrows():
        m, s, c, lbl = style[row["method"]]
        ax.scatter(row[OBJ_COLS[0]], row[OBJ_COLS[1]], marker=m, s=s,
                   facecolor=c, edgecolor="white", linewidth=1.2, zorder=5,
                   label=f"{lbl} (cost {row[OBJ_COLS[2]]:.0f})")
    ax.set_xlabel(OBJ_COLS[0])
    ax.set_ylabel(OBJ_COLS[1])
    ax.set_title("Decision making on the Pareto front\n"
                 "(a-posteriori preference articulation)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"opt_decision{suffix}.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", type=Path, default=RESULTS / "reference_front.csv",
                    help="CSV of the Pareto reference front")
    ap.add_argument("--suffix", default="",
                    help="filename suffix for outputs (e.g. _curved)")
    args = ap.parse_args()

    df_front = pd.read_csv(args.front)
    decision = select(args.front)
    save_outputs(df_front, decision, suffix=args.suffix)
    pd.set_option("display.width", 200)
    print(decision.to_string(index=False))
    print(f"\nOutputs -> {RESULTS}/ (opt_decision.{{csv,png}})")


if __name__ == "__main__":
    main()
