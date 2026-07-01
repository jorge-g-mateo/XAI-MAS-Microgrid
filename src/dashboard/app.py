"""Microgrid interactive dashboard.

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date

from src.mas.run_mas import _dominated_by_front, run_experiment
from src.optimization.market import scenario_from_timestamp
from src.optimization.problem import decode_objectives
from src.optimization.run_optimization import run as run_optimization
from src.pipeline import persistence

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Microgrid Dashboard",
    page_icon="⚡",
    layout="wide",
)
st.title("⚡ Microgrid Multi-Agent System — Dashboard")
st.caption("Ingesta · Optimización multi-objetivo · Negociación FIPA-ACL · Explainability")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Configuración del escenario")

date_input = st.sidebar.date_input(
    "Fecha",
    value=date(2017, 6, 15),
    min_value=date(2017, 1, 1),
    max_value=date(2020, 12, 31),
)
hour_input = st.sidebar.slider("Hora del día", 0, 23, 19)
timestamp = f"{date_input} {hour_input:02d}:00:00"
st.sidebar.markdown(f"**Timestamp:** `{timestamp}`")

st.sidebar.divider()
st.sidebar.subheader("Parámetros (solo para runs nuevos)")
runs_input = st.sidebar.slider("Runs por algoritmo", 1, 5, 2)
evals_input = st.sidebar.select_slider(
    "Evaluaciones por run", options=[1000, 3000, 6000, 10000], value=6000
)
rounds_input = st.sidebar.slider("Rondas de negociación", 5, 20, 10)

run_btn = st.sidebar.button("▶ Ejecutar", type="primary", use_container_width=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_from_db(ts: str):
    """Return (cfg, metrics_df, mas_df, run_id) if ts exists in DB, else None."""
    try:
        eng = persistence.get_engine()
        runs_df = persistence.read_table("runs", eng)
        match = runs_df[runs_df["scenario"] == ts]
        if match.empty:
            return None
        run_id = match.iloc[-1]["run_id"]
        metrics_df = persistence.read_table("optimization_metrics", eng)
        metrics_df = metrics_df[metrics_df["run_id"] == run_id]
        mas_df = persistence.read_table("mas_outcomes", eng)
        mas_df = mas_df[mas_df["run_id"] == run_id]
        cfg = scenario_from_timestamp(ts)
        return cfg, metrics_df, mas_df, run_id
    except Exception:
        return None


def _annotate_mas(mas_df: pd.DataFrame, front: np.ndarray) -> pd.DataFrame:
    dom, dist = [], []
    for _, r in mas_df.iterrows():
        d, dd = _dominated_by_front((r.profit_solar, r.profit_wind, r.buyer_cost), front)
        dom.append(d)
        dist.append(round(dd, 3))
    out = mas_df.copy()
    out["dominated_by_optimum"] = dom
    out["dist_to_pareto"] = dist
    return out


# ── Charts ────────────────────────────────────────────────────────────────────

def fig_scenario(cfg) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 3))
    labels = ["Demanda", "Solar", "Eólica"]
    values = [cfg.demand, cfg.gen_solar, cfg.gen_wind]
    colors = ["#e74c3c", "#f39c12", "#3498db"]
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.2f} kW", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("kW")
    ax.set_title(f"Generación vs Demanda\n{cfg.label}", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(values) * 1.25)
    fig.tight_layout()
    return fig


def fig_pareto(front: np.ndarray) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: profit_solar vs profit_wind, color = buyer_cost
    sc = axes[0].scatter(front[:, 0], front[:, 1], c=front[:, 2],
                         cmap="RdYlGn_r", s=18, alpha=0.75)
    plt.colorbar(sc, ax=axes[0], label="Coste consumidor (€)")
    axes[0].set_xlabel("Beneficio solar AS (€)")
    axes[0].set_ylabel("Beneficio eólico AE (€)")
    axes[0].set_title(f"Frente de Pareto — {len(front)} soluciones")
    axes[0].grid(True, alpha=0.3)

    # Right: profit_solar vs buyer_cost
    axes[1].scatter(front[:, 0], front[:, 2], c=front[:, 1],
                    cmap="viridis", s=18, alpha=0.75)
    axes[1].set_xlabel("Beneficio solar AS (€)")
    axes[1].set_ylabel("Coste consumidor AC (€)")
    axes[1].set_title("Trade-off solar vs coste consumidor")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def fig_metrics(metrics_df: pd.DataFrame) -> plt.Figure:
    indicators = [c for c in ["HV", "GD", "IGD", "Epsilon"] if c in metrics_df.columns]
    n = len(indicators)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    colors = ["#2ecc71", "#3498db", "#9b59b6"]
    for ax, ind in zip(axes, indicators):
        vals = metrics_df.groupby("Algorithm")[ind].mean()
        bars = ax.bar(vals.index, vals.values, color=colors[:len(vals)], alpha=0.85, edgecolor="white")
        ax.set_title(ind, fontsize=10)
        ax.set_ylabel(ind)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Calidad del frente de Pareto por algoritmo", fontsize=10)
    fig.tight_layout()
    return fig


def fig_mas(mas_df: pd.DataFrame) -> plt.Figure:
    strategies = mas_df["solar_strategy"] + "\nvs\n" + mas_df["wind_strategy"]
    x = np.arange(len(strategies))
    width = 0.28

    fig, ax = plt.subplots(figsize=(max(10, len(x) * 0.9), 4))
    ax.bar(x - width, mas_df["profit_solar"], width, label="Beneficio solar (€)", color="#f39c12", alpha=0.85)
    ax.bar(x,         mas_df["profit_wind"],  width, label="Beneficio eólico (€)", color="#3498db", alpha=0.85)
    ax.bar(x + width, mas_df["buyer_cost"],   width, label="Coste consumidor (€)", color="#e74c3c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=7)
    ax.set_ylabel("€")
    ax.set_title("Resultados por par de estrategias (solar × eólica)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_pareto_vs_mas(front: np.ndarray, mas_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(front[:, 0], front[:, 1], c="lightblue", s=15, alpha=0.6,
               label=f"Pareto ({len(front)} soluciones)", zorder=1)
    colors_mas = ["#e74c3c" if d else "#2ecc71"
                  for d in mas_df.get("dominated_by_optimum", [False] * len(mas_df))]
    sc2 = ax.scatter(mas_df["profit_solar"], mas_df["profit_wind"],
                     c=colors_mas, s=60, zorder=2, edgecolors="black", linewidths=0.5)
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="lightblue", label="Frente de Pareto"),
        Patch(facecolor="#e74c3c", label="MAS dominado por óptimo"),
        Patch(facecolor="#2ecc71", label="MAS no dominado"),
    ]
    ax.legend(handles=legend_elems, fontsize=8)
    ax.set_xlabel("Beneficio solar (€)")
    ax.set_ylabel("Beneficio eólico (€)")
    ax.set_title("MAS vs frente de Pareto óptimo")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if run_btn:
    cached = load_from_db(timestamp)

    if cached:
        cfg, metrics_df, mas_df, run_id = cached
        st.success(f"Escenario cargado desde DB — `{run_id}`")
        with st.spinner("Recalculando frente de Pareto para visualización..."):
            _, _, combined = run_optimization(cfg, runs=1, evals=3000)
            front = np.array([decode_objectives(s.objectives) for s in combined])
        mas_df = _annotate_mas(mas_df, front)
    else:
        st.info("Escenario nuevo — ejecutando pipeline en directo...")
        bar = st.progress(0, text="Cargando escenario y predicciones...")
        cfg = scenario_from_timestamp(timestamp)
        bar.progress(15, text="Optimizando (NSGA-II / NSGA-III / SPEA2)...")
        _, metrics_df, combined = run_optimization(cfg, runs=runs_input, evals=evals_input)
        front = np.array([decode_objectives(s.objectives) for s in combined])
        bar.progress(75, text="Negociando (FIPA-ACL Contract Net)...")
        mas_df = run_experiment(cfg, rounds=rounds_input, with_optimum=False)
        mas_df = _annotate_mas(mas_df, front)
        bar.progress(100, text="Completado")
        st.success(f"Pipeline completado — escenario `{timestamp}`")

    # ── Row 1: escenario + Pareto ──────────────────────────────────────────
    st.divider()
    col_scenario, col_pareto = st.columns([1, 2.5])

    with col_scenario:
        st.subheader("Escenario energético")
        st.pyplot(fig_scenario(cfg))
        c1, c2, c3 = st.columns(3)
        c1.metric("Demanda", f"{cfg.demand:.2f} kW")
        c2.metric("Solar", f"{cfg.gen_solar:.2f} kW")
        c3.metric("Eólica", f"{cfg.gen_wind:.2f} kW")

    with col_pareto:
        st.subheader("Frente de Pareto")
        st.pyplot(fig_pareto(front))

    # ── Row 2: métricas de optimización ───────────────────────────────────
    st.divider()
    st.subheader("Métricas de calidad del frente (por algoritmo)")
    col_m1, col_m2 = st.columns([2.5, 1])
    with col_m1:
        st.pyplot(fig_metrics(metrics_df))
    with col_m2:
        show_cols = [c for c in ["Algorithm", "Solutions", "HV", "GD", "IGD", "Epsilon"]
                     if c in metrics_df.columns]
        st.dataframe(metrics_df[show_cols].round(4), use_container_width=True, hide_index=True)

    # ── Row 3: MAS ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Negociación Multi-Agente (FIPA-ACL Contract Net)")
    col_mas1, col_mas2 = st.columns([2, 1])
    with col_mas1:
        st.pyplot(fig_mas(mas_df))
    with col_mas2:
        st.pyplot(fig_pareto_vs_mas(front, mas_df))

    with st.expander("Ver tabla completa de outcomes MAS"):
        display_cols = [c for c in ["solar_strategy", "wind_strategy", "profit_solar",
                                     "profit_wind", "buyer_cost", "shortfall",
                                     "dominated_by_optimum", "dist_to_pareto"]
                        if c in mas_df.columns]
        st.dataframe(mas_df[display_cols].round(3), use_container_width=True, hide_index=True)

    # ── Row 4: xAI ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Explainability (xAI) - forecasting models")
    xai_dir = ROOT / "results" / "xai"

    xai_images = {
        "SHAP Beeswarm": "shap_beeswarm.png",
        "PDP / ICE": "pdp_ice.png",
        "Importancia (PFI)": "permutation_importance.png",
    }

    for model_name in ["solar", "wind"]:
        st.markdown(f"**Modelo {model_name.upper()}**")
        cols = st.columns(len(xai_images))
        for col, (title, fname) in zip(cols, xai_images.items()):
            img_path = xai_dir / model_name / fname
            with col:
                if img_path.exists():
                    st.image(str(img_path), caption=f"{title} — {model_name}", use_container_width=True)
                else:
                    st.info(f"Ejecuta primero el pipeline completo para generar {fname}")

else:
    st.info("Selecciona una fecha y hora en el panel lateral y pulsa **▶ Ejecutar**.")
    st.markdown("""
    ### Qué muestra este dashboard

    | Sección | Qué hace |
    |---|---|
    | **Energy scenario** | Demand vs solar and wind generation predicted by the ML models |
    | **Frente de Pareto** | Soluciones no dominadas del problema tri-objetivo (NSGA-II/III/SPEA2) |
    | **Métricas de optimización** | Hypervolume, GD, IGD y Epsilon por algoritmo |
    | **Negociación MAS** | Beneficios y costes por combinación de estrategias (FIPA-ACL Contract Net) |
    | **MAS vs Pareto** | Cuánto se aleja la negociación emergente del óptimo matemático |
    | **xAI** | SHAP beeswarm, PDP/ICE and permutation importance for the generation models |

    Si el escenario ya está en la base de datos se carga en ~5 segundos.
    Si es nuevo, el pipeline completo tarda ~20 segundos (sin xAI en directo).
    """)
