"""Explainability toolkit for the solar and wind power models (xAI).

Wraps a pre-trained tree model + its data into a single :class:`ModelExplainer`
that produces:

  * Global:        permutation feature importance, SHAP beeswarm, PDP/ICE.
  * Interactions:  Friedman's H-statistic over the top features.
  * Local:         SHAP waterfall for 3 representative instances.
  * What-if:       ceteris-paribus curves (vary one feature, keep the rest).

NOTE on performance: the provided RandomForests are large (300 trees, depth ~30-48),
so exact SHAP *interaction* values are impractically slow. We therefore use
SHAP on a small subsample for local/global attributions, and Friedman's
H-statistic (which only needs fast `predict` calls) for interactions — the same
tool used to compare local explanation methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

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


@dataclass
class ModelExplainer:
    name: str                 # "solar" or "wind"
    model: object
    X: pd.DataFrame           # feature frame (model-ready order)
    y: pd.Series              # ground-truth target
    target_name: str
    shap_n: int = 150         # rows used for the SHAP beeswarm (speed vs. fidelity)
    rng: int = 42

    def __post_init__(self):
        self.X = self.X.reset_index(drop=True)
        self.y = self.y.reset_index(drop=True)
        self.outdir = OUT / self.name
        self.outdir.mkdir(parents=True, exist_ok=True)
        self._explainer = shap.TreeExplainer(self.model)
        self._pi = None  # cached permutation-importance dataframe

    # ------------------------------------------------------------------ global
    def permutation_importance(self, n_repeats: int = 5) -> pd.DataFrame:
        if self._pi is not None:
            return self._pi
        r = permutation_importance(self.model, self.X, self.y,
                                   n_repeats=n_repeats, random_state=self.rng)
        df = (pd.DataFrame({"feature": self.X.columns,
                            "importance": r.importances_mean,
                            "std": r.importances_std})
              .sort_values("importance", ascending=False).reset_index(drop=True))
        df.to_csv(self.outdir / "permutation_importance.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(df.feature[::-1], df.importance[::-1], xerr=df["std"][::-1])
        ax.set_title(f"Permutation importance — {self.name} model")
        ax.set_xlabel("mean increase in error")
        fig.tight_layout(); fig.savefig(self.outdir / "permutation_importance.png", dpi=130)
        plt.close(fig)
        self._pi = df
        return df

    def plot_beeswarm(self):
        sv = self._explainer(self.X.iloc[: self.shap_n])
        plt.figure()
        shap.plots.beeswarm(sv, show=False, max_display=len(self.X.columns))
        plt.title(f"SHAP summary — {self.name} model")
        plt.tight_layout()
        plt.savefig(self.outdir / "shap_beeswarm.png", dpi=130, bbox_inches="tight")
        plt.close()

    def plot_pdp(self, features: list[str]):
        fig, ax = plt.subplots(figsize=(5 * len(features), 4))
        PartialDependenceDisplay.from_estimator(
            self.model, self.X, features, kind="both", ax=ax,
            subsample=150, random_state=self.rng)  # "both" = PDP + ICE
        fig.suptitle(f"PDP + ICE — {self.name} model")
        fig.tight_layout(); fig.savefig(self.outdir / "pdp_ice.png", dpi=130)
        plt.close(fig)

    # ------------------------------------------------------------ interactions
    def _centered_pd(self, S: list[str], Xs: pd.DataFrame) -> np.ndarray:
        """Centered partial-dependence values of feature set ``S`` over ``Xs``."""
        n = len(Xs)
        out = np.empty(n)
        for i in range(n):
            tmp = Xs.copy()
            for f in S:
                tmp[f] = Xs.iloc[i][f]
            out[i] = self.model.predict(tmp).mean()
        return out - out.mean()

    def h_statistic(self, top_k: int = 4, n: int = 60) -> pd.DataFrame:
        """Friedman's pairwise H-statistic for the ``top_k`` most important
        features (interaction strength in [0, 1]). Uses fast predict calls."""
        imp = self.permutation_importance()  # already cheap; reuse ordering
        feats = list(imp.feature.head(top_k))
        Xs = self.X.sample(min(n, len(self.X)), random_state=self.rng).reset_index(drop=True)

        pd_single = {f: self._centered_pd([f], Xs) for f in feats}
        m = pd.DataFrame(0.0, index=feats, columns=feats)
        for a, b in combinations(feats, 2):
            pd_pair = self._centered_pd([a, b], Xs)
            num = np.sum((pd_pair - pd_single[a] - pd_single[b]) ** 2)
            den = np.sum(pd_pair ** 2)
            h = float(np.sqrt(num / den)) if den > 1e-12 else 0.0
            m.loc[a, b] = m.loc[b, a] = h
        m.to_csv(self.outdir / "h_statistic.csv")

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(m.values, cmap="viridis", vmin=0, vmax=max(0.05, m.values.max()))
        ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=45, ha="right")
        ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats)
        for i in range(len(feats)):
            for j in range(len(feats)):
                ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center",
                        color="w", fontsize=8)
        ax.set_title(f"Friedman H-statistic (interactions) — {self.name}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(self.outdir / "h_statistic.png", dpi=130)
        plt.close(fig)
        return m

    def plot_pdp_2d(self, a: str, b: str, subsample: int = 200):
        """Two-way PDP heatmap for the ``(a, b)`` pair. The H-statistic *quantifies*
        the strongest interaction but does not *show* it; this closes the
        measure→visualize loop by rendering how the joint effect of ``a`` and ``b``
        on the prediction departs from the sum of their individual effects."""
        fig, ax = plt.subplots(figsize=(6, 5))
        # Cast to float: the strongest pair often involves integer features
        # (hour, dayofyear) which sklearn's 2-D PDP refuses on integer dtypes.
        PartialDependenceDisplay.from_estimator(
            self.model, self.X.astype(float), [(a, b)], ax=ax,
            subsample=subsample, random_state=self.rng)
        ax.set_title(f"2-D PDP ({a} × {b}) — {self.name}")
        fig.tight_layout()
        fig.savefig(self.outdir / f"pdp2d_{a}_{b}.png", dpi=130)
        plt.close(fig)

    # ------------------------------------------------------------------- local
    def representative_instances(self) -> dict[str, int]:
        """Pick 3 instances: highest, median(>0) and zero/low production."""
        pred = self.model.predict(self.X)
        order = np.argsort(pred)
        nonzero = np.where(pred > pred.max() * 0.01)[0]
        med = int(nonzero[len(nonzero) // 2]) if len(nonzero) else int(order[len(order) // 2])
        return {"high": int(order[-1]), "medium": med, "low": int(order[0])}

    def plot_waterfalls(self, instances: dict[str, int] | None = None):
        instances = instances or self.representative_instances()
        for label, idx in instances.items():
            sv = self._explainer(self.X.iloc[[idx]])  # SHAP for just this instance
            plt.figure()
            shap.plots.waterfall(sv[0], show=False)
            plt.title(f"SHAP waterfall — {self.name} [{label}] "
                      f"(pred={self.model.predict(self.X.iloc[[idx]])[0]:.1f})")
            plt.tight_layout()
            plt.savefig(self.outdir / f"waterfall_{label}.png", dpi=130,
                        bbox_inches="tight")
            plt.close()
        return instances

    # ----------------------------------------------------------------- what-if
    def what_if(self, feature: str, idx: int | None = None, n: int = 40) -> pd.DataFrame:
        """Ceteris paribus: vary one feature over its range for a base instance."""
        if idx is None:
            idx = self.representative_instances()["medium"]
        base = self.X.iloc[[idx]].copy()
        lo, hi = self.X[feature].quantile([0.01, 0.99])
        grid = np.linspace(lo, hi, n)
        rows = pd.concat([base] * n, ignore_index=True)
        rows[feature] = grid
        preds = self.model.predict(rows)
        df = pd.DataFrame({feature: grid, "prediction": preds})
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grid, preds, marker=".")
        ax.axvline(base[feature].iloc[0], color="red", ls="--", label="actual value")
        ax.set_xlabel(feature); ax.set_ylabel(f"predicted {self.target_name}")
        ax.set_title(f"What-if ({feature}) — {self.name} [instance {idx}]")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.outdir / f"whatif_{feature}.png", dpi=130)
        plt.close(fig)
        return df

    def _feature_grid(self, feature: str, n: int) -> np.ndarray:
        lo, hi = self.X[feature].quantile([0.001, 0.999])
        return np.linspace(lo, hi, n)

    def _ceilings(self, feature: str, idx: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Per-instance ceteris-paribus **ceiling**: the max prediction reachable by
        sweeping ``feature`` over ``grid`` while every other feature stays frozen at
        its value in that instance. One vectorized ``predict`` over all
        (instance × grid) combinations."""
        block = self.X.iloc[idx]
        rep = block.loc[block.index.repeat(len(grid))].reset_index(drop=True)
        rep[feature] = np.tile(grid, len(idx))
        return self.model.predict(rep).reshape(len(idx), len(grid)).max(axis=1)

    def _select_base(self, feature: str, target: float, preds_all: np.ndarray,
                     n_candidates: int, grid: np.ndarray) -> int:
        """Pick the base instance for the counterfactual on methodological grounds
        instead of taking an arbitrary one. Candidates are the **sub-target**
        instances (predicted below the high-regime target). Among those that this
        single feature can actually *rescue* (ceiling ≥ target) we take the one with
        the **lowest current prediction** — the most under-producing case that one
        lever still lifts into the high regime. If none is rescuable, we fall back to
        the sub-target instance with the **highest ceiling** (the closest possible
        approach), so the reported ``reachable=False`` is the genuine best case."""
        sub = np.where(preds_all < target)[0]
        if len(sub) == 0:
            return int(np.argmin(preds_all))
        rng = np.random.RandomState(self.rng)
        cand = sub if len(sub) <= n_candidates else rng.choice(sub, n_candidates, replace=False)
        ceil = self._ceilings(feature, cand, grid)
        rescuable = cand[ceil >= target]
        if len(rescuable):
            return int(rescuable[int(np.argmin(preds_all[rescuable]))])
        return int(cand[int(np.argmax(ceil))])

    def counterfactual(self, feature: str, target: float | None = None,
                       idx: int | None = None, n: int = 200,
                       target_quantile: float = 0.90,
                       n_candidates: int = 300) -> dict:
        """Contrastive ("why not Y?") explanation: the **minimal change** in
        ``feature`` (ceteris paribus) that moves a base instance's prediction to a
        principled high-production ``target``. Where the what-if curve answers "what
        happens if I vary x?", the counterfactual inverts it: "how much must x change
        to reach Y?".

        **Methodology (why the target/base are not arbitrary):**

        * ``target`` is the ``target_quantile`` (default **P90**) of the *model's own
          predicted-output distribution* — a defensible "high-production regime"
          derived from the data, not a hand-picked number.
        * ``idx`` (the base instance) is auto-selected by :meth:`_select_base`: the
          lowest-producing instance that this single feature can still lift to the
          target (ceiling ≥ target). This directly answers "what minimal change moves
          a low/medium case into a reasonably-defined high scenario?".

        Both can be overridden for manual/reproducible queries. When **no** value of
        this single feature reaches the target (the high regime needs several
        favorable conditions at once — exactly the interactions the H-statistic
        measures), ``reachable=False`` is reported together with the attainable
        ceiling and the remaining gap, rather than drawn as if it succeeded.
        """
        preds_all = self.model.predict(self.X)
        explicit_target = target is not None
        if target is None:
            target = float(np.quantile(preds_all, target_quantile))
        grid = self._feature_grid(feature, n)

        if idx is None:
            idx = self._select_base(feature, target, preds_all,
                                    n_candidates, self._feature_grid(feature, 60))

        base = self.X.iloc[[idx]].copy()
        base_val = float(base[feature].iloc[0])
        base_pred = float(self.model.predict(base)[0])
        rows = pd.concat([base] * n, ignore_index=True)
        rows[feature] = grid
        preds = self.model.predict(rows)
        ceiling = float(preds.max())                       # best this lever can do here
        j = int(np.argmin(np.abs(preds - target)))
        cf_val, cf_pred = float(grid[j]), float(preds[j])
        reachable = abs(cf_pred - target) <= 0.05 * max(1.0, abs(target))
        gap = round(float(target - ceiling), 2)            # >0 only when unreachable

        pct = f"P{int(round(target_quantile * 100))}" if not explicit_target else "target"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grid, preds, color="tab:blue", lw=2)
        ax.axhline(target, color="tab:green", ls=":", label=f"{pct} target = {target:.1f}")
        ax.plot([base_val], [base_pred], "ro",
                label=f"actual ({base_val:.1f} → {base_pred:.1f})")
        if reachable:
            ax.plot([cf_val], [cf_pred], "g*", ms=15,
                    label=f"counterfactual ({feature} = {cf_val:.1f}, Δ={cf_val - base_val:+.1f})")
            ax.annotate("", xy=(cf_val, cf_pred), xytext=(base_val, base_pred),
                        arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))
        else:
            ax.axhline(ceiling, color="tab:red", ls="--",
                       label=f"max reachable = {ceiling:.1f} (gap {gap:.1f})")
            ax.plot([cf_val], [cf_pred], "rx", ms=12)
        ax.set_xlabel(feature); ax.set_ylabel(f"predicted {self.target_name}")
        tag = "" if reachable else "  [target unreachable via this feature alone]"
        ax.set_title(f"Counterfactual ({feature}) — {self.name} [instance {idx}]{tag}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.outdir / f"counterfactual_{feature}.png", dpi=130)
        plt.close(fig)
        return {"instance": int(idx), "feature": feature,
                "target_quantile": None if explicit_target else round(target_quantile, 3),
                "target": round(float(target), 2),
                "actual_value": round(base_val, 3), "actual_pred": round(base_pred, 2),
                "cf_value": round(cf_val, 3), "cf_pred": round(cf_pred, 2),
                "delta": round(cf_val - base_val, 3),
                "max_reachable_pred": round(ceiling, 2), "gap_to_target": gap,
                "reachable": bool(reachable)}

    # ------------------------------------------------------------------- ALE
    def ale(self, feature: str, n_bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """1-D Accumulated Local Effects for ``feature``.

        Unlike PDP, ALE only perturbs each instance *within* the bin its feature
        value actually falls in, so it stays on the data manifold and is not
        distorted by correlated features. Returns (bin_edges, ale_at_edges); the
        curve is centered to mean zero over the data distribution.
        """
        x = self.X[feature].to_numpy()
        edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            return edges, np.zeros(len(edges))
        # bin index per row (1..len(edges)-1)
        idx = np.clip(np.searchsorted(edges, x, side="left"), 1, len(edges) - 1)

        local = np.zeros(len(edges) - 1)
        counts = np.zeros(len(edges) - 1)
        for b in range(1, len(edges)):
            mask = idx == b
            if not mask.any():
                continue
            Xb = self.X[mask]
            lo = Xb.copy(); lo[feature] = edges[b - 1]
            hi = Xb.copy(); hi[feature] = edges[b]
            local[b - 1] = (self.model.predict(hi) - self.model.predict(lo)).mean()
            counts[b - 1] = mask.sum()

        ale = np.concatenate([[0.0], np.cumsum(local)])  # value at each edge
        # center by the distribution-weighted mean of the segment midpoints
        mids = (ale[:-1] + ale[1:]) / 2
        w = counts / counts.sum() if counts.sum() > 0 else None
        ale = ale - (np.average(mids, weights=w) if w is not None else ale.mean())
        return edges, ale

    def plot_pdp_vs_ale(self, feature: str, n_bins: int = 20, grid: int = 40):
        """Overlay a centered PDP and the ALE for ``feature``. A gap between them
        signals correlation with other features (PDP uses unrealistic
        combinations; ALE does not) — the key PDP-vs-ALE diagnostic."""
        lo, hi = self.X[feature].quantile([0.01, 0.99])
        gx = np.linspace(lo, hi, grid)
        pdp = np.array([self._pred_with(feature, v).mean() for v in gx])
        pdp_c = pdp - pdp.mean()
        edges, ale = self.ale(feature, n_bins)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(gx, pdp_c, color="tab:orange", lw=2, label="PDP (centered)")
        ax.plot(edges, ale, color="tab:blue", marker=".", lw=2, label="ALE")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlabel(feature); ax.set_ylabel(f"effect on {self.target_name}")
        ax.set_title(f"PDP vs. ALE ({feature}) — {self.name}")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.outdir / f"pdp_vs_ale_{feature}.png", dpi=130)
        plt.close(fig)
        return edges, ale

    def _pred_with(self, feature: str, value: float) -> np.ndarray:
        tmp = self.X.copy()
        tmp[feature] = value
        return self.model.predict(tmp)

    def plot_ale_grid(self, features: list[str], n_bins: int = 20):
        """ALE curves for several features in one figure (one panel each)."""
        k = len(features)
        fig, axes = plt.subplots(1, k, figsize=(4.2 * k, 3.6))
        if k == 1:
            axes = [axes]
        for ax, feat in zip(axes, features):
            edges, ale = self.ale(feat, n_bins)
            ax.plot(edges, ale, color="tab:blue", marker=".", lw=2)
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_xlabel(feat); ax.set_ylabel(f"ALE on {self.target_name}")
            ax.set_title(feat)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"ALE of the top features — {self.name}")
        fig.tight_layout()
        fig.savefig(self.outdir / "ale_top.png", dpi=130)
        plt.close(fig)

    def plot_shap_dependence(self, feature: str):
        """SHAP dependence plot for ``feature``, colored by the feature SHAP
        interacts with most strongly (auto-selected) — shows how the feature's
        effect on the prediction is modulated by a second variable."""
        Xs = self.X.iloc[: self.shap_n]
        sv = self._explainer(Xs)
        plt.figure()
        shap.dependence_plot(feature, sv.values, Xs, interaction_index="auto",
                             show=False)
        plt.title(f"SHAP dependence ({feature}) — {self.name}")
        plt.tight_layout()
        plt.savefig(self.outdir / f"shap_dependence_{feature}.png", dpi=130,
                    bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------ LIME
    def lime_explain(self, instances: dict[str, int] | None = None,
                     num_features: int = 6):
        """LIME local surrogate explanations for representative instances.

        Complements SHAP: LIME fits a sparse linear model around each instance,
        giving an alternative, model-agnostic local attribution. Comparing the two
        is a robustness check on the local explanation."""
        instances = instances or self.representative_instances()
        explainer = LimeTabularExplainer(
            self.X.to_numpy(), feature_names=list(self.X.columns),
            mode="regression", random_state=self.rng, discretize_continuous=True)

        # wrap predict so LIME's numpy input keeps the model's feature names
        def predict_fn(arr):
            return self.model.predict(pd.DataFrame(arr, columns=self.X.columns))

        out = {}
        for label, idx in instances.items():
            exp = explainer.explain_instance(
                self.X.iloc[idx].to_numpy(), predict_fn, num_features=num_features)
            fig = exp.as_pyplot_figure()
            fig.set_size_inches(7, 4)
            fig.suptitle(f"LIME local explanation — {self.name} [{label}] "
                         f"(pred={self.model.predict(self.X.iloc[[idx]])[0]:.1f})")
            fig.tight_layout()
            fig.savefig(self.outdir / f"lime_{label}.png", dpi=130, bbox_inches="tight")
            plt.close(fig)
            out[label] = exp.as_list()
        return out

    def local_method_agreement(self, idx: int, k: int = 5) -> dict:
        """Top-``k`` features for instance ``idx`` by |SHAP| vs |LIME weight|, and
        their overlap (a robustness check that the two local methods agree)."""
        sv = self._explainer(self.X.iloc[[idx]])
        shap_rank = (pd.Series(np.abs(sv.values[0]), index=self.X.columns)
                     .sort_values(ascending=False).head(k).index.tolist())

        explainer = LimeTabularExplainer(
            self.X.to_numpy(), feature_names=list(self.X.columns),
            mode="regression", random_state=self.rng, discretize_continuous=True)

        def predict_fn(arr):
            return self.model.predict(pd.DataFrame(arr, columns=self.X.columns))

        exp = explainer.explain_instance(self.X.iloc[idx].to_numpy(), predict_fn,
                                         num_features=len(self.X.columns))
        # map each LIME condition string back to a feature name, keep top-k by |w|
        lime_rank = []
        for cond, _ in sorted(exp.as_list(), key=lambda kv: -abs(kv[1])):
            for col in self.X.columns:
                if col in cond and col not in lime_rank:
                    lime_rank.append(col)
                    break
            if len(lime_rank) == k:
                break
        overlap = len(set(shap_rank) & set(lime_rank))
        return {"shap_top": shap_rank, "lime_top": lime_rank,
                "overlap": overlap, "k": k}


def build_solar_explainer(sample: int = 600) -> ModelExplainer:
    df = add_date_features(load_solar_data()).dropna(subset=SOLAR_FEATURES)
    df = df.sample(min(sample, len(df)), random_state=42)
    return ModelExplainer("solar", load_solar_model(), df[SOLAR_FEATURES],
                          df["SystemProduction"], "SystemProduction")


def build_wind_explainer(sample: int = 600) -> ModelExplainer:
    df = add_date_features(load_wind_data()).dropna(subset=WIND_FEATURES)
    df = df.sample(min(sample, len(df)), random_state=42)
    return ModelExplainer("wind", load_wind_model(), df[WIND_FEATURES],
                          df["Power"], "Power")
