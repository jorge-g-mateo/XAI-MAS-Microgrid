"""Microgrid optimization dashboard — Dash Mantine Components 0.14.

Run with:
    python src/dashboard/app_mantine.py

Opens at http://localhost:8050
"""

from __future__ import annotations

import base64
import copy
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html, no_update
from scipy.stats import wilcoxon

from src.dashboard.snapshots import run_with_snapshots
from src.mas.buyer_strategies import BUYER_STRATEGIES
from src.mas.simulation import run_negotiation
from src.mas.strategies import STRATEGIES
from src.optimization.market import MarketConfig, scenario_from_timestamp
from src.optimization.run_optimization import run as run_optimization
from src.pipeline import persistence
from src.pipeline.persistence import new_run_id, save_mas, save_optimization, save_run
from src.xai.explain import build_solar_explainer, build_wind_explainer

# ── App ───────────────────────────────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Microgrid Optimization Dashboard"

ALGORITHMS = ["NSGAII", "NSGAIII", "SPEA2", "MOEAD"]
ALGO_COLORS = {"NSGAII": "#2ecc71", "NSGAIII": "#3498db", "SPEA2": "#9b59b6", "MOEAD": "#e67e22"}
POINTS_PER_TICK = 1

PIPELINE_STEPS = [
    "check_data", "ingest_forecast", "optimize", "negotiate", "persist", "explain",
]

SPECIAL_DEFS = [
    ("Max solar profit",     "Max solar",   "#f39c12", "cross"),
    ("Max wind profit",      "Max wind",    "#3498db", "cross"),
    ("Min consumer cost",    "Min cost",    "#2ecc71", "cross"),
    ("Best trade-off (knee)","Equilibrium", "#9b59b6", "diamond"),
]

SENS_PARAMS = {
    "precio_mercado": "Market price (×)",
    "capacidad_solar": "Solar capacity (×)",
    "capacidad_eolica": "Wind capacity (×)",
    "demanda": "Demand (×)",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _img_b64(path: Path) -> str:
    try:
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    except FileNotFoundError:
        return ""


def _star_traces_3d(arr: np.ndarray) -> list:
    utopia = np.array([arr[:, 0].max(), arr[:, 1].max(), arr[:, 2].min()])
    scale = arr.std(axis=0) + 1e-9
    dists = np.linalg.norm((arr - utopia) / scale, axis=1)
    indices = [int(arr[:, 0].argmax()), int(arr[:, 1].argmax()),
               int(arr[:, 2].argmin()), int(dists.argmin())]
    traces = []
    for (label, tag, color, sym), idx in zip(SPECIAL_DEFS, indices):
        p = arr[idx]
        traces.append(go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", name=label,
            marker=dict(symbol=sym, size=12, color=color,
                        line=dict(width=2, color="white")),
            hovertemplate=(f"<b>{label}</b><br>Solar: {p[0]:.3f}€<br>"
                           f"Wind: {p[1]:.3f}€<br>Cost: {p[2]:.3f}€<extra></extra>"),
            showlegend=True,
        ))
    return traces


def _make_3d_fig(arr: np.ndarray, with_markers: bool = False,
                 algorithm: str = "", color_override: str | None = None) -> go.Figure:
    marker_kw = (dict(size=5, color=color_override, opacity=0.75,
                      line=dict(width=0.3, color="white"))
                 if color_override else
                 dict(size=5, color=arr[:, 2], colorscale="RdYlGn_r", showscale=True,
                      colorbar=dict(title="Consumer cost (€)", thickness=10, len=0.5, x=1.0),
                      line=dict(width=0.3, color="white"), opacity=0.85))
    traces = [go.Scatter3d(
        x=arr[:, 0], y=arr[:, 1], z=arr[:, 2], mode="markers",
        name="Pareto front", marker=marker_kw,
        text=[f"Solar: {p[0]:.3f}€<br>Wind: {p[1]:.3f}€<br>Cost: {p[2]:.3f}€" for p in arr],
        hovertemplate="%{text}<extra></extra>", showlegend=True, customdata=arr,
    )]
    if with_markers:
        traces.extend(_star_traces_3d(arr))
    title = f"{algorithm} — {len(arr)} solutions" if algorithm else f"{len(arr)} solutions"
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="Solar profit (€)", yaxis_title="Wind profit (€)",
            zaxis_title="Consumer cost (€)", bgcolor="white",
            xaxis=dict(backgroundcolor="#f8f9fa", gridcolor="#dee2e6"),
            yaxis=dict(backgroundcolor="#f8f9fa", gridcolor="#dee2e6"),
            zaxis=dict(backgroundcolor="#f8f9fa", gridcolor="#dee2e6"),
        ),
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.85)", font=dict(size=9)),
        margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="white",
        title=dict(text=title, font=dict(size=11), x=0.5),
    )
    return fig


def _status_done(timestamp: str, n_solutions: int, color: str = "green",
                 label: str = "Pipeline completed") -> html.Div:
    return html.Div([
        dmc.Group(gap="xs", mb="xs", children=[
            dmc.Badge(label, color=color, size="sm"),
            dmc.Text(f"{timestamp} · {n_solutions} non-dominated solutions", size="xs", c="dimmed"),
        ]),
        dmc.Group(gap=4, mt="xs", wrap="wrap", children=[
            dmc.Badge(f"✓ {s}", color="blue", variant="light", size="xs")
            for s in PIPELINE_STEPS
        ]),
    ])


def _scene_layout() -> dict:
    return dict(
        xaxis_title="Solar profit (€)", yaxis_title="Wind profit (€)",
        zaxis_title="Consumer cost (€)", bgcolor="white",
        xaxis=dict(backgroundcolor="#f8f9fa", gridcolor="#dee2e6"),
        yaxis=dict(backgroundcolor="#f8f9fa", gridcolor="#dee2e6"),
        zaxis=dict(backgroundcolor="#f8f9fa", gridcolor="#dee2e6"),
    )


# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = dmc.MantineProvider(children=[
    dmc.Container(fluid=True, p="md", children=[

        dmc.Group(justify="space-between", mb="md", children=[
            dmc.Stack(gap=0, children=[
                dmc.Title("Microgrid Optimization Dashboard", order=2),
                dmc.Text("3D Pareto · Comparison · XAI · Statistics · Sensitivity",
                         c="dimmed", size="sm"),
            ]),
            dmc.Badge("NSGA-II / III · SPEA2", color="blue", size="lg"),
        ]),
        dmc.Divider(mb="md"),

        dmc.Grid(gutter="md", children=[

            # ── Left panel ───────────────────────────────────────────────────
            dmc.GridCol(span=3, children=[
                dmc.Paper(shadow="xs", p="md", radius="md", children=[
                    dmc.Title("Scenario", order=4, mb="sm"),

                    dmc.Text("Date", size="sm", fw=500, mb=4),
                    dcc.DatePickerSingle(
                        id="date-picker", date="2017-06-15",
                        min_date_allowed="2017-01-01", max_date_allowed="2020-12-31",
                        display_format="DD/MM/YYYY",
                        style={"marginBottom": "12px", "width": "100%"},
                    ),
                    dmc.Text("Hour of day", size="sm", fw=500, mb=4),
                    dmc.Slider(id="hour-slider", min=0, max=23, step=1, value=19,
                               marks=[{"value": v, "label": str(v)} for v in [0, 6, 12, 18, 23]],
                               mb="xl"),

                    dmc.Divider(my="sm"),
                    dmc.Title("Optimization", order=4, mb="sm"),
                    dmc.Text("Algorithm (Animation tab)", size="sm", fw=500, mb=4),
                    dmc.Select(id="algo-select",
                               data=[{"value": a, "label": a} for a in ALGORITHMS],
                               value="NSGAII", mb="sm"),
                    dmc.Text("Evaluations", size="sm", fw=500, mb=4),
                    dmc.Slider(id="evals-slider", min=1000, max=10000, step=1000, value=6000,
                               marks=[{"value": v, "label": f"{v//1000}k"}
                                      for v in [1000, 3000, 6000, 10000]],
                               mb="xl"),

                    dmc.Divider(my="sm"),
                    dmc.Title("Negotiation", order=4, mb="sm"),
                    dmc.Text("MAS rounds", size="sm", fw=500, mb=4),
                    dmc.Slider(id="rounds-slider", min=5, max=20, step=5, value=10,
                               marks=[{"value": v, "label": str(v)} for v in [5, 10, 15, 20]],
                               mb="xl"),

                    dmc.Button("▶ Run pipeline", id="run-btn",
                               fullWidth=True, size="md", color="blue", mt="md"),
                    dmc.Button("⚖ Compare 4 algorithms", id="compare-btn",
                               fullWidth=True, size="sm", color="violet",
                               variant="outline", mt="xs"),
                    dmc.Text("Comparison and statistical analysis may take several minutes.",
                             size="xs", c="dimmed", mt=4),

                    html.Div(id="status-area", style={"marginTop": "12px"}),

                    dcc.Store(id="snapshots-store"),
                    dcc.Store(id="scenario-store"),
                    dcc.Store(id="mas-store"),
                    dcc.Store(id="metrics-store"),
                    dcc.Store(id="anim-state-store"),
                    dcc.Store(id="pareto-front-store"),
                    dcc.Interval(id="anim-interval", interval=50, disabled=True, n_intervals=0),
                    dcc.Download(id="download-csv"),
                ]),
            ]),

            # ── Right panel — tabs ───────────────────────────────────────────
            dmc.GridCol(span=9, children=[
                dmc.Tabs(value="anim", children=[
                    dmc.TabsList([
                        dmc.TabsTab("3D Animation",         value="anim"),
                        dmc.TabsTab("Comparison",           value="comp"),
                        dmc.TabsTab("XAI",                  value="xai"),
                        dmc.TabsTab("Statistical Analysis", value="stat"),
                        dmc.TabsTab("Sensitivity",          value="sens"),
                        dmc.TabsTab("History",              value="hist"),
                    ]),

                    # ── Tab 1: Animated 3D ───────────────────────────────────
                    dmc.TabsPanel(value="anim", pt="sm", children=[
                        dmc.Paper(shadow="xs", p="md", radius="md", children=[
                            dmc.Group(justify="space-between", mb="sm", children=[
                                dmc.Text("Points appear one by one during optimization. "
                                         "Optimal points are highlighted at the end. Drag to rotate.",
                                         size="sm", c="dimmed"),
                                dmc.Button("↓ Export CSV", id="export-csv-btn",
                                           size="xs", variant="outline", color="gray"),
                            ]),
                            dcc.Graph(id="pareto-anim", style={"height": "450px"},
                                      config={"displayModeBar": True,
                                              "toImageButtonOptions": {
                                                  "format": "png", "filename": "pareto_3d",
                                                  "height": 600, "width": 900}}),
                            html.Div(id="click-info", style={"marginTop": "8px"}),
                        ]),
                    ]),

                    # ── Tab 2: Comparison ────────────────────────────────────
                    dmc.TabsPanel(value="comp", pt="sm", children=[
                        dmc.Paper(shadow="xs", p="md", radius="md", children=[
                            dmc.Text("Press '⚖ Compare 4 algorithms' to see all 5 charts.",
                                     size="sm", c="dimmed", mb="sm"),
                            dmc.Grid(gutter="sm", children=[
                                dmc.GridCol(span=6, children=[
                                    dmc.Text("NSGA-II", size="xs", fw=600, ta="center", c="green"),
                                    dcc.Graph(id="comp-nsgaii", style={"height": "280px"},
                                              config={"displayModeBar": False}),
                                ]),
                                dmc.GridCol(span=6, children=[
                                    dmc.Text("NSGA-III", size="xs", fw=600, ta="center", c="blue"),
                                    dcc.Graph(id="comp-nsgaiii", style={"height": "280px"},
                                              config={"displayModeBar": False}),
                                ]),
                                dmc.GridCol(span=6, children=[
                                    dmc.Text("SPEA2", size="xs", fw=600, ta="center", c="violet"),
                                    dcc.Graph(id="comp-spea2", style={"height": "280px"},
                                              config={"displayModeBar": False}),
                                ]),
                                dmc.GridCol(span=6, children=[
                                    dmc.Text("MOEA/D", size="xs", fw=600, ta="center", c="orange"),
                                    dcc.Graph(id="comp-moead", style={"height": "280px"},
                                              config={"displayModeBar": False}),
                                ]),
                                dmc.GridCol(span=12, children=[
                                    dmc.Text("Combined (all 4)", size="xs", fw=600, ta="center", c="dimmed"),
                                    dcc.Graph(id="comp-combined", style={"height": "340px"},
                                              config={"displayModeBar": False}),
                                ]),
                            ]),
                        ]),
                    ]),

                    # ── Tab 3: XAI ──────────────────────────────────────────
                    dmc.TabsPanel(value="xai", pt="sm", children=[
                        dmc.Paper(shadow="xs", p="md", radius="md", children=[
                            dmc.Group(justify="space-between", mb="sm", children=[
                                dmc.Text("Feature importance and SHAP explanations "
                                         "for the solar and wind prediction models.",
                                         size="sm", c="dimmed"),
                                dmc.Button("▶ Run XAI", id="run-xai-btn",
                                           size="sm", color="orange"),
                            ]),
                            dmc.Tabs(value="xai-solar", children=[
                                dmc.TabsList([
                                    dmc.TabsTab("Solar", value="xai-solar"),
                                    dmc.TabsTab("Wind", value="xai-wind"),
                                ]),
                                dmc.TabsPanel(value="xai-solar", pt="sm", children=[
                                    dmc.Grid(gutter="md", children=[
                                        dmc.GridCol(span=6, children=[
                                            dmc.Title("Feature importance", order=5, mb="xs"),
                                            dcc.Graph(id="xai-solar-importance",
                                                      style={"height": "300px"}),
                                        ]),
                                        dmc.GridCol(span=6, children=[
                                            dmc.Title("SHAP Beeswarm", order=5, mb="xs"),
                                            html.Img(id="xai-solar-beeswarm",
                                                     style={"width": "100%", "maxHeight": "300px",
                                                            "objectFit": "contain"}),
                                        ]),
                                        dmc.GridCol(span=12, children=[
                                            dmc.Title("SHAP Waterfall — representative instances",
                                                      order=5, mb="xs"),
                                            dmc.Group(gap="sm", children=[
                                                html.Img(id="xai-solar-wf-high",
                                                         style={"width": "33%", "maxHeight": "240px",
                                                                "objectFit": "contain"}),
                                                html.Img(id="xai-solar-wf-med",
                                                         style={"width": "33%", "maxHeight": "240px",
                                                                "objectFit": "contain"}),
                                                html.Img(id="xai-solar-wf-low",
                                                         style={"width": "33%", "maxHeight": "240px",
                                                                "objectFit": "contain"}),
                                            ]),
                                        ]),
                                    ]),
                                ]),
                                dmc.TabsPanel(value="xai-wind", pt="sm", children=[
                                    dmc.Grid(gutter="md", children=[
                                        dmc.GridCol(span=6, children=[
                                            dmc.Title("Feature importance", order=5, mb="xs"),
                                            dcc.Graph(id="xai-wind-importance",
                                                      style={"height": "300px"}),
                                        ]),
                                        dmc.GridCol(span=6, children=[
                                            dmc.Title("SHAP Beeswarm", order=5, mb="xs"),
                                            html.Img(id="xai-wind-beeswarm",
                                                     style={"width": "100%", "maxHeight": "300px",
                                                            "objectFit": "contain"}),
                                        ]),
                                        dmc.GridCol(span=12, children=[
                                            dmc.Title("SHAP Waterfall — representative instances",
                                                      order=5, mb="xs"),
                                            dmc.Group(gap="sm", children=[
                                                html.Img(id="xai-wind-wf-high",
                                                         style={"width": "33%", "maxHeight": "240px",
                                                                "objectFit": "contain"}),
                                                html.Img(id="xai-wind-wf-med",
                                                         style={"width": "33%", "maxHeight": "240px",
                                                                "objectFit": "contain"}),
                                                html.Img(id="xai-wind-wf-low",
                                                         style={"width": "33%", "maxHeight": "240px",
                                                                "objectFit": "contain"}),
                                            ]),
                                        ]),
                                    ]),
                                ]),
                            ]),
                        ]),
                    ]),

                    # ── Tab 4: Statistical analysis ──────────────────────────
                    dmc.TabsPanel(value="stat", pt="sm", children=[
                        dmc.Paper(shadow="xs", p="md", radius="md", children=[
                            dmc.Grid(gutter="md", align="center", mb="sm", children=[
                                dmc.GridCol(span=4, children=[
                                    dmc.Text("Runs per algorithm", size="sm", fw=500, mb=4),
                                    dmc.Slider(id="stat-n-slider", min=5, max=20, step=5, value=10,
                                               marks=[{"value": v, "label": str(v)}
                                                      for v in [5, 10, 15, 20]]),
                                ]),
                                dmc.GridCol(span=3, children=[
                                    dmc.Text("Metric", size="sm", fw=500, mb=4),
                                    dmc.Select(id="stat-metric-select",
                                               data=[{"value": m, "label": l} for m, l in
                                                     [("HV", "Hypervolume (higher = better)"),
                                                      ("IGD", "Inverted GD (lower = better)"),
                                                      ("Epsilon", "Epsilon (lower = better)"),
                                                      ("Spread", "Spread / diversity (lower = better)")]],
                                               value="HV"),
                                ]),
                                dmc.GridCol(span=2, children=[
                                    dmc.Text(
                                        "Compares the chosen quality indicator across the 4 "
                                        "algorithms on the same scenario with different random seeds. "
                                        "Pairwise Wilcoxon signed-rank test.",
                                        size="xs", c="dimmed"),
                                ]),
                                dmc.GridCol(span=3, children=[
                                    dmc.Button("▶ Run analysis", id="run-stat-btn",
                                               fullWidth=True, color="teal"),
                                ]),
                            ]),
                            dmc.Grid(gutter="md", children=[
                                dmc.GridCol(span=7, children=[
                                    dcc.Graph(id="stat-boxplot", style={"height": "380px"}),
                                ]),
                                dmc.GridCol(span=5, children=[
                                    dmc.Title("Wilcoxon test (p-values)", order=5, mb="xs"),
                                    html.Div(id="stat-wilcoxon"),
                                ]),
                            ]),
                        ]),
                    ]),

                    # ── Tab 5: Sensitivity ───────────────────────────────────
                    dmc.TabsPanel(value="sens", pt="sm", children=[
                        dmc.Paper(shadow="xs", p="md", radius="md", children=[
                            dmc.Grid(gutter="md", align="center", mb="sm", children=[
                                dmc.GridCol(span=4, children=[
                                    dmc.Text("Parameter to vary", size="sm", fw=500, mb=4),
                                    dmc.Select(
                                        id="sens-param",
                                        data=[{"value": k, "label": v}
                                              for k, v in SENS_PARAMS.items()],
                                        value="precio_mercado", mb="sm",
                                    ),
                                    dmc.Text("Number of steps", size="sm", fw=500, mb=4),
                                    dmc.Select(
                                        id="sens-steps",
                                        data=[{"value": "3", "label": "3 steps (×0.5 / ×1.0 / ×2.0)"},
                                              {"value": "5", "label": "5 steps (×0.5 → ×2.0)"},
                                              {"value": "7", "label": "7 steps (×0.5 → ×2.0)"}],
                                        value="5",
                                    ),
                                ]),
                                dmc.GridCol(span=5, children=[
                                    dmc.Text(
                                        "Varies the selected parameter from ×0.5 to ×2.0 of its "
                                        "baseline value and shows the resulting Pareto front for each. "
                                        "Reveals how the solution space shifts with the parameter.",
                                        size="xs", c="dimmed"),
                                ]),
                                dmc.GridCol(span=3, children=[
                                    dmc.Button("▶ Run sensitivity", id="run-sens-btn",
                                               fullWidth=True, color="cyan"),
                                ]),
                            ]),
                            dcc.Graph(id="sens-graph", style={"height": "430px"}),
                            html.Div(id="sens-info", style={"marginTop": "8px"}),
                        ]),
                    ]),

                    # ── Tab 6: History ───────────────────────────────────────
                    dmc.TabsPanel(value="hist", pt="sm", children=[
                        dmc.Paper(shadow="xs", p="md", radius="md", children=[
                            dmc.Group(justify="space-between", mb="sm", children=[
                                dmc.Title("Runs saved in DB", order=4),
                                dmc.Button("↻ Load history", id="load-hist-btn",
                                           size="sm", variant="outline", color="blue"),
                            ]),
                            html.Div(id="hist-area"),
                        ]),
                    ]),
                ]),
            ]),
        ]),

        # Bottom row
        dmc.Grid(gutter="md", mt="md", children=[
            dmc.GridCol(span=4, children=[
                dmc.Paper(shadow="xs", p="md", radius="md", children=[
                    dmc.Title("Non-dominated solutions", order=4, mb="xs"),
                    dcc.Graph(id="solutions-line", style={"height": "230px"}),
                ]),
            ]),
            dmc.GridCol(span=4, children=[
                dmc.Paper(shadow="xs", p="md", radius="md", children=[
                    dmc.Title("Quality metrics", order=4, mb="xs"),
                    dcc.Graph(id="metrics-bar", style={"height": "230px"}),
                ]),
            ]),
            dmc.GridCol(span=4, children=[
                dmc.Paper(shadow="xs", p="md", radius="md", children=[
                    dmc.Title("MAS Negotiation", order=4, mb="xs"),
                    dcc.Graph(id="mas-bar", style={"height": "230px"}),
                ]),
            ]),
        ]),
    ]),
])


# ── Callbacks — pipeline ──────────────────────────────────────────────────────

@callback(
    Output("snapshots-store", "data"),
    Output("scenario-store", "data"),
    Output("mas-store", "data"),
    Output("metrics-store", "data"),
    Output("status-area", "children"),
    Output("anim-state-store", "data"),
    Output("anim-interval", "disabled"),
    Output("anim-interval", "n_intervals"),
    Output("pareto-front-store", "data"),
    Output("pareto-anim", "figure"),     # reset graph to blank on new run
    Input("run-btn", "n_clicks"),
    State("date-picker", "date"),
    State("hour-slider", "value"),
    State("algo-select", "value"),
    State("evals-slider", "value"),
    State("rounds-slider", "value"),
    prevent_initial_call=True,
)
def run_pipeline(n_clicks, date_str, hour, algorithm, evals, rounds):
    date_part = str(date_str).split("T")[0]
    timestamp = f"{date_part} {int(hour):02d}:00:00"
    cfg = scenario_from_timestamp(timestamp)
    snapshots, final_front = run_with_snapshots(
        cfg, algorithm=algorithm, evals=evals, every_n=max(200, evals // 30))
    _, metrics_df, _ = run_optimization(cfg, runs=1, evals=evals)

    # Seller-strategy matrix only (buyer fixed at price_taker) — keeps the
    # live dashboard run fast. The full seller x buyer cross (144 combos) runs
    # in the Prefect pipeline, where exhaustive analysis matters more than
    # interactive speed.
    mas_rows = []
    for s in STRATEGIES:
        for w in STRATEGIES:
            res = run_negotiation(cfg, s, w, buyer_strategy="price_taker",
                                  rounds=rounds, log=False)
            mas_rows.append({
                "solar_strategy": s, "wind_strategy": w, "buyer_strategy": "price_taker",
                "profit_solar": res.profit_solar, "profit_wind": res.profit_wind,
                "buyer_cost": res.buyer_cost, "shortfall": res.shortfall,
                "price_solar": res.final_price_solar, "price_wind": res.final_price_wind,
            })
    mas_df = pd.DataFrame(mas_rows)

    scenario_data = {"timestamp": timestamp, "demand": round(cfg.demand, 3),
                     "solar": round(cfg.gen_solar, 3), "wind": round(cfg.gen_wind, 3),
                     "algorithm": algorithm}
    status = _status_done(timestamp, len(final_front))

    run_id = new_run_id()
    save_run(run_id, cfg)
    save_optimization(run_id, metrics_df)
    save_mas(run_id, mas_df)

    # Reset animation: blank figure + fresh state + enable interval
    anim_state = {"snap_idx": 0, "pt_idx": 0, "cumulative": [], "done": False}
    blank_fig = go.Figure().update_layout(
        paper_bgcolor="white",
        scene=dict(bgcolor="white",
                   xaxis_title="Solar profit (€)",
                   yaxis_title="Wind profit (€)",
                   zaxis_title="Consumer cost (€)"),
    )
    return (snapshots, scenario_data, mas_df.to_dict("records"),
            metrics_df.to_dict("records"), status, anim_state,
            False, 0, final_front.tolist(), blank_fig)


@callback(
    Output("pareto-anim", "figure", allow_duplicate=True),
    Output("anim-state-store", "data", allow_duplicate=True),
    Output("anim-interval", "disabled", allow_duplicate=True),
    Input("anim-interval", "n_intervals"),
    State("snapshots-store", "data"),
    State("anim-state-store", "data"),
    State("scenario-store", "data"),
    prevent_initial_call=True,
)
def animate_tick(n_intervals, snapshots, state, scenario):
    if not snapshots or not state:
        return go.Figure(), state, True
    algorithm = (scenario or {}).get("algorithm", "NSGAII")
    cumulative: list = list(state["cumulative"])
    snap_idx, pt_idx, done = state["snap_idx"], state["pt_idx"], state["done"]
    if done:
        return _make_3d_fig(np.array(cumulative), with_markers=True,
                            algorithm=algorithm), state, True
    for _ in range(POINTS_PER_TICK):
        if snap_idx >= len(snapshots):
            done = True; break
        snap = snapshots[snap_idx]
        new_pts = (snap["front"] if snap_idx == 0 else
                   [p for p in snap["front"]
                    if (round(p[0], 6), round(p[1], 6)) not in
                    {(round(q[0], 6), round(q[1], 6)) for q in snapshots[snap_idx-1]["front"]}])
        if pt_idx < len(new_pts):
            cumulative.append(new_pts[pt_idx]); pt_idx += 1
        if pt_idx >= len(new_pts):
            snap_idx += 1; pt_idx = 0
        if snap_idx >= len(snapshots):
            done = True; break
    if not cumulative:
        return go.Figure(), {"snap_idx": snap_idx, "pt_idx": pt_idx,
                             "cumulative": cumulative, "done": done}, done
    arr = np.array(cumulative)
    new_state = {"snap_idx": snap_idx, "pt_idx": pt_idx, "cumulative": cumulative, "done": done}
    return _make_3d_fig(arr, with_markers=done, algorithm=algorithm), new_state, done


# ── Callbacks — click info & export ──────────────────────────────────────────

@callback(
    Output("click-info", "children"),
    Input("pareto-anim", "clickData"),
    prevent_initial_call=True,
)
def show_click_info(clickData):
    if not clickData:
        return ""
    pt = clickData["points"][0]
    x, y, z = pt.get("x"), pt.get("y"), pt.get("z")
    if x is None:
        return ""
    return dmc.Paper(p="xs", radius="sm", withBorder=True, style={"background": "#f8f9fa"},
        children=[dmc.Group(gap="xl", children=[
            dmc.Stack(gap=0, children=[dmc.Text("Solar profit AS", size="xs", c="dimmed"),
                                       dmc.Text(f"{x:.4f} €", size="sm", fw=600, c="#f39c12")]),
            dmc.Stack(gap=0, children=[dmc.Text("Wind profit AE", size="xs", c="dimmed"),
                                       dmc.Text(f"{y:.4f} €", size="sm", fw=600, c="#3498db")]),
            dmc.Stack(gap=0, children=[dmc.Text("Consumer cost AC", size="xs", c="dimmed"),
                                       dmc.Text(f"{z:.4f} €", size="sm", fw=600, c="#e74c3c")]),
            dmc.Stack(gap=0, children=[dmc.Text("Net balance", size="xs", c="dimmed"),
                                       dmc.Text(f"{x+y-z:.4f} €", size="sm", fw=600, c="#2ecc71")]),
        ])])


@callback(
    Output("download-csv", "data"),
    Input("export-csv-btn", "n_clicks"),
    State("pareto-front-store", "data"),
    State("scenario-store", "data"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, front_data, scenario):
    if not front_data:
        return no_update
    arr = np.array(front_data)
    df = pd.DataFrame(arr, columns=["profit_solar_eur", "profit_wind_eur", "buyer_cost_eur"])
    algo = (scenario or {}).get("algorithm", "algo")
    ts = (scenario or {}).get("timestamp", "").replace(" ", "T").replace(":", "")
    return dcc.send_data_frame(df.to_csv, f"pareto_{algo}_{ts}.csv", index=False)


# ── Callbacks — comparison ────────────────────────────────────────────────────

@callback(
    Output("comp-nsgaii",   "figure"),
    Output("comp-nsgaiii",  "figure"),
    Output("comp-spea2",    "figure"),
    Output("comp-moead",    "figure"),
    Output("comp-combined", "figure"),
    Output("status-area", "children", allow_duplicate=True),
    Input("compare-btn", "n_clicks"),
    State("date-picker", "date"),
    State("hour-slider", "value"),
    State("evals-slider", "value"),
    prevent_initial_call=True,
)
def run_comparison(n_clicks, date_str, hour, evals):
    date_part = str(date_str).split("T")[0]
    timestamp = f"{date_part} {int(hour):02d}:00:00"
    cfg = scenario_from_timestamp(timestamp)
    fronts = {}
    for algo in ALGORITHMS:
        _, front = run_with_snapshots(cfg, algorithm=algo, evals=evals, every_n=evals)
        fronts[algo] = front

    indiv_figs = [_make_3d_fig(fronts[a], with_markers=True, algorithm=a) for a in ALGORITHMS]

    combined = go.Figure()
    for algo in ALGORITHMS:
        arr = np.array(fronts[algo])
        combined.add_trace(go.Scatter3d(
            x=arr[:, 0], y=arr[:, 1], z=arr[:, 2], mode="markers", name=algo,
            marker=dict(size=5, color=ALGO_COLORS[algo], opacity=0.75,
                        line=dict(width=0.3, color="white")),
            text=[f"{algo}<br>Solar:{p[0]:.3f}€ Wind:{p[1]:.3f}€ Cost:{p[2]:.3f}€" for p in arr],
            hovertemplate="%{text}<extra></extra>",
        ))
    combined.update_layout(
        scene=_scene_layout(),
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.85)", font=dict(size=10)),
        margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="white",
        title=dict(text="All 4 algorithms", font=dict(size=11), x=0.5),
    )

    n_total = sum(len(fronts[a]) for a in ALGORITHMS)
    status = html.Div([
        dmc.Group(gap="xs", mb="xs", children=[
            dmc.Badge("Comparison complete", color="violet", size="sm"),
            dmc.Text(f"{timestamp} · {n_total} total solutions", size="xs", c="dimmed"),
        ]),
        dmc.Group(gap=4, mt="xs", wrap="wrap", children=[
            dmc.Badge(f"{a}: {len(fronts[a])} sol.", color="blue", variant="light", size="xs")
            for a in ALGORITHMS
        ]),
    ])
    return indiv_figs[0], indiv_figs[1], indiv_figs[2], indiv_figs[3], combined, status


# ── Callbacks — XAI ──────────────────────────────────────────────────────────

@callback(
    Output("xai-solar-importance", "figure"),
    Output("xai-wind-importance",  "figure"),
    Output("xai-solar-beeswarm",   "src"),
    Output("xai-wind-beeswarm",    "src"),
    Output("xai-solar-wf-high",    "src"),
    Output("xai-solar-wf-med",     "src"),
    Output("xai-solar-wf-low",     "src"),
    Output("xai-wind-wf-high",     "src"),
    Output("xai-wind-wf-med",      "src"),
    Output("xai-wind-wf-low",      "src"),
    Input("run-xai-btn", "n_clicks"),
    prevent_initial_call=True,
)
def run_xai(n_clicks):
    def _importance_fig(pi_df: pd.DataFrame, title: str) -> go.Figure:
        top = pi_df.head(10).sort_values("importance")
        fig = go.Figure(go.Bar(
            x=top["importance"], y=top["feature"], orientation="h",
            error_x=dict(type="data", array=top["std"].tolist()),
            marker_color="#3498db", opacity=0.85,
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=12), x=0.5),
            xaxis_title="Error increase (importance)",
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#f0f0f0"),
            xaxis=dict(gridcolor="#f0f0f0"),
        )
        return fig

    xai_root = ROOT / "results" / "xai"

    # If PNGs already exist, load them without rerunning XAI (instant)
    solar_beeswarm = xai_root / "solar" / "shap_beeswarm.png"
    solar_pi_csv   = xai_root / "solar" / "permutation_importance.csv"
    wind_beeswarm  = xai_root / "wind"  / "shap_beeswarm.png"
    wind_pi_csv    = xai_root / "wind"  / "permutation_importance.csv"

    needs_solar = not solar_beeswarm.exists() or not solar_pi_csv.exists()
    needs_wind  = not wind_beeswarm.exists()  or not wind_pi_csv.exists()

    if needs_solar:
        solar_exp = build_solar_explainer(sample=150)
        solar_pi  = solar_exp.permutation_importance(n_repeats=2)
        solar_exp.plot_beeswarm()
        solar_exp.plot_waterfalls()
    else:
        solar_pi = pd.read_csv(solar_pi_csv)

    if needs_wind:
        wind_exp = build_wind_explainer(sample=150)
        wind_pi  = wind_exp.permutation_importance(n_repeats=2)
        wind_exp.plot_beeswarm()
        wind_exp.plot_waterfalls()
    else:
        wind_pi = pd.read_csv(wind_pi_csv)

    return (
        _importance_fig(solar_pi, "Importance — Solar Model"),
        _importance_fig(wind_pi,  "Importance — Wind Model"),
        _img_b64(xai_root / "solar" / "shap_beeswarm.png"),
        _img_b64(xai_root / "wind"  / "shap_beeswarm.png"),
        _img_b64(xai_root / "solar" / "waterfall_high.png"),
        _img_b64(xai_root / "solar" / "waterfall_medium.png"),
        _img_b64(xai_root / "solar" / "waterfall_low.png"),
        _img_b64(xai_root / "wind"  / "waterfall_high.png"),
        _img_b64(xai_root / "wind"  / "waterfall_medium.png"),
        _img_b64(xai_root / "wind"  / "waterfall_low.png"),
    )


# ── Callbacks — statistical analysis ─────────────────────────────────────────

METRIC_HIGHER_IS_BETTER = {"HV"}  # all other indicators (GD/IGD/Epsilon/Spread) are minimized


@callback(
    Output("stat-boxplot",  "figure"),
    Output("stat-wilcoxon", "children"),
    Input("run-stat-btn", "n_clicks"),
    State("stat-n-slider", "value"),
    State("stat-metric-select", "value"),
    State("date-picker",   "date"),
    State("hour-slider",   "value"),
    State("evals-slider",  "value"),
    prevent_initial_call=True,
)
def run_statistical(n_clicks, n_runs, metric, date_str, hour, evals):
    date_part = str(date_str).split("T")[0]
    timestamp = f"{date_part} {int(hour):02d}:00:00"
    cfg = scenario_from_timestamp(timestamp)
    higher_is_better = metric in METRIC_HIGHER_IS_BETTER

    results: dict[str, list[float]] = {a: [] for a in ALGORITHMS}
    for seed in range(n_runs):
        _, metrics_df, _ = run_optimization(cfg, runs=1, evals=evals, seed=seed)
        for algo in ALGORITHMS:
            val = metrics_df.loc[metrics_df["Algorithm"] == algo, metric]
            if not val.empty:
                results[algo].append(float(val.iloc[0]))

    # Boxplot
    fig = go.Figure()
    for algo in ALGORITHMS:
        fig.add_trace(go.Box(
            y=results[algo], name=algo,
            marker_color=ALGO_COLORS[algo], boxmean="sd",
            hovertemplate=f"<b>{algo}</b><br>{metric}: %{{y:.4f}}<extra></extra>",
        ))
    better_hint = "higher = better" if higher_is_better else "lower = better"
    fig.update_layout(
        title=dict(text=f"{metric} distribution — {n_runs} runs per algorithm ({better_hint})",
                   font=dict(size=12), x=0.5),
        yaxis_title=metric, plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        margin=dict(l=50, r=10, t=50, b=40),
    )

    # Wilcoxon pairwise — every pair among the algorithms actually compared
    pairs = list(itertools.combinations(ALGORITHMS, 2))
    th_s = {"padding": "6px 10px", "textAlign": "center",
             "borderBottom": "2px solid #dee2e6", "fontSize": "12px", "fontWeight": "600"}
    td_s = {"padding": "5px 10px", "textAlign": "center", "fontSize": "12px"}

    rows = []
    for a, b in pairs:
        va, vb = results[a], results[b]
        if len(va) >= 5 and len(vb) >= 5:
            try:
                _, p = wilcoxon(va, vb)
                sig = "✓ Yes (p<0.05)" if p < 0.05 else "✗ No"
                sig_c = "#2ecc71" if p < 0.05 else "#e74c3c"
                a_wins = np.mean(va) > np.mean(vb) if higher_is_better else np.mean(va) < np.mean(vb)
                better = a if a_wins else b
            except Exception:
                p, sig, sig_c, better = float("nan"), "—", "#888", "—"
        else:
            p, sig, sig_c, better = float("nan"), "Insufficient data", "#888", "—"
        rows.append(html.Tr([
            html.Td(f"{a} vs {b}", style=td_s),
            html.Td(f"{p:.4f}" if not np.isnan(p) else "—", style=td_s),
            html.Td(sig, style={**td_s, "color": sig_c, "fontWeight": "600"}),
            html.Td(better, style=td_s),
        ]))

    table = html.Table(
        [html.Thead(html.Tr([html.Th(h, style=th_s)
                              for h in ["Pair", "p-value", "Sig. diff.", f"Best {metric}"]]))] +
        [html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse",
               "border": "1px solid #dee2e6", "marginTop": "12px"},
    )
    medias = html.Div([
        dmc.Text(f"Mean {metric} per algorithm:", size="xs", fw=600, mt="sm"),
        dmc.Group(gap="xs", mt=4, children=[
            dmc.Badge(f"{a}: {np.mean(results[a]):.4f}", color="blue", variant="light")
            for a in ALGORITHMS if results[a]
        ]),
    ])
    return fig, html.Div([table, medias])


# ── Callbacks — sensitivity ───────────────────────────────────────────────────

@callback(
    Output("sens-graph", "figure"),
    Output("sens-info",  "children"),
    Input("run-sens-btn", "n_clicks"),
    State("sens-param",  "value"),
    State("sens-steps",  "value"),
    State("date-picker", "date"),
    State("hour-slider", "value"),
    State("algo-select", "value"),
    State("evals-slider","value"),
    prevent_initial_call=True,
)
def run_sensitivity(n_clicks, param, steps_str, date_str, hour, algorithm, evals):
    date_part = str(date_str).split("T")[0]
    timestamp = f"{date_part} {int(hour):02d}:00:00"
    base_cfg = scenario_from_timestamp(timestamp)

    n_steps = int(steps_str)
    multipliers = np.linspace(0.5, 2.0, n_steps)
    colorscale = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
    colors = [colorscale[int(i * (len(colorscale) - 1) / max(n_steps - 1, 1))]
              for i in range(n_steps)]

    fig = go.Figure()
    summary_rows = []

    for mult, color in zip(multipliers, colors):
        cfg = copy.copy(base_cfg)
        if param == "precio_mercado":
            cfg.price_min = base_cfg.price_min * mult
            cfg.price_max = base_cfg.price_max * mult
        elif param == "capacidad_solar":
            cfg.gen_solar = base_cfg.gen_solar * mult
        elif param == "capacidad_eolica":
            cfg.gen_wind = base_cfg.gen_wind * mult
        elif param == "demanda":
            cfg.demand = base_cfg.demand * mult

        _, front = run_with_snapshots(cfg, algorithm=algorithm, evals=evals, every_n=evals)
        arr = np.array(front)
        label = f"×{mult:.2f}"
        fig.add_trace(go.Scatter3d(
            x=arr[:, 0], y=arr[:, 1], z=arr[:, 2],
            mode="markers", name=label,
            marker=dict(size=4, color=color, opacity=0.75,
                        line=dict(width=0.3, color="white")),
            text=[f"{label}<br>Solar:{p[0]:.3f}€ Wind:{p[1]:.3f}€ Cost:{p[2]:.3f}€"
                  for p in arr],
            hovertemplate="%{text}<extra></extra>",
        ))
        summary_rows.append({
            "mult": label, "n_sol": len(arr),
            "hv_solar": float(arr[:, 0].max()), "hv_wind": float(arr[:, 1].max()),
            "min_cost": float(arr[:, 2].min()),
        })

    param_label = SENS_PARAMS.get(param, param)
    fig.update_layout(
        scene=_scene_layout(),
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.85)", font=dict(size=10),
                    title=dict(text=param_label)),
        margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="white",
        title=dict(text=f"Sensitivity: {param_label} — {algorithm}", font=dict(size=12), x=0.5),
    )

    th_s = {"padding": "5px 8px", "textAlign": "center",
             "borderBottom": "2px solid #dee2e6", "fontSize": "11px", "fontWeight": "600"}
    td_s = {"padding": "4px 8px", "textAlign": "center", "fontSize": "11px"}
    info = html.Table(
        [html.Thead(html.Tr([html.Th(h, style=th_s)
                              for h in ["Mult.", "N sol.", "Max solar", "Max wind", "Min cost"]]))] +
        [html.Tbody([html.Tr([
            html.Td(r["mult"], style=td_s),
            html.Td(str(r["n_sol"]), style=td_s),
            html.Td(f"{r['hv_solar']:.3f}€", style=td_s),
            html.Td(f"{r['hv_wind']:.3f}€", style=td_s),
            html.Td(f"{r['min_cost']:.3f}€", style=td_s),
        ]) for r in summary_rows])],
        style={"width": "100%", "borderCollapse": "collapse",
               "border": "1px solid #dee2e6", "marginTop": "8px"},
    )
    return fig, info


# ── Callbacks — bottom charts ─────────────────────────────────────────────────

@callback(
    Output("solutions-line", "figure"),
    Input("snapshots-store", "data"),
    State("scenario-store", "data"),
    prevent_initial_call=True,
)
def update_solutions_line(snapshots, scenario):
    if not snapshots:
        return go.Figure()
    algorithm = (scenario or {}).get("algorithm", "NSGAII")
    color = ALGO_COLORS.get(algorithm, "#3498db")
    fig = go.Figure(go.Scatter(
        x=[s["evals"] for s in snapshots], y=[s["n_solutions"] for s in snapshots],
        mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=5),
        fill="tozeroy", fillcolor=_hex_to_rgba(color),
    ))
    fig.update_layout(xaxis_title="Evaluations", yaxis_title="Non-dominated solutions",
                      margin=dict(l=40, r=10, t=10, b=40),
                      plot_bgcolor="white", paper_bgcolor="white",
                      xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                      yaxis=dict(showgrid=True, gridcolor="#f0f0f0"))
    return fig


@callback(
    Output("metrics-bar", "figure"),
    Input("metrics-store", "data"),
    prevent_initial_call=True,
)
def update_metrics(metrics_data):
    if not metrics_data:
        return go.Figure()
    algos = [r["Algorithm"] for r in metrics_data]
    colors = [ALGO_COLORS.get(a, "#888") for a in algos]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="HV", x=algos, y=[r["HV"] for r in metrics_data],
                         marker_color=colors, opacity=0.85))
    fig.add_trace(go.Bar(name="IGD", x=algos, y=[r["IGD"] for r in metrics_data],
                         marker_color=colors, opacity=0.6, marker_pattern_shape="/"))
    fig.add_trace(go.Bar(name="Epsilon", x=algos, y=[r["Epsilon"] for r in metrics_data],
                         marker_color=colors, opacity=0.45, marker_pattern_shape="x"))
    fig.add_trace(go.Bar(name="Spread", x=algos, y=[r["Spread"] for r in metrics_data],
                         marker_color=colors, opacity=0.3, marker_pattern_shape="."))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.15, font=dict(size=9)),
                      margin=dict(l=40, r=10, t=20, b=40),
                      plot_bgcolor="white", paper_bgcolor="white",
                      yaxis=dict(showgrid=True, gridcolor="#f0f0f0"))
    return fig


@callback(
    Output("mas-bar", "figure"),
    Input("mas-store", "data"),
    prevent_initial_call=True,
)
def update_mas(mas_data):
    if not mas_data:
        return go.Figure()
    # mas_data crosses every (solar x wind) pair with all buyer strategies —
    # average over the buyer dimension so the chart stays readable as one bar
    # per seller-strategy pair (the buyer's own behavior is now data, not a
    # fixed choice; full detail is in the CSV export / History tab).
    df = pd.DataFrame(mas_data)
    n_buyers = df["buyer_strategy"].nunique()
    agg = (df.groupby(["solar_strategy", "wind_strategy"], sort=False)
             .agg(profit_solar=("profit_solar", "mean"),
                  profit_wind=("profit_wind", "mean"),
                  buyer_cost=("buyer_cost", "mean"))
             .reset_index())
    labels = [f"{r.solar_strategy[:4]}×{r.wind_strategy[:4]}" for r in agg.itertuples()]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Solar", x=labels, y=agg["profit_solar"],
                         marker_color="#f39c12", opacity=0.85))
    fig.add_trace(go.Bar(name="Wind", x=labels, y=agg["profit_wind"],
                         marker_color="#3498db", opacity=0.85))
    fig.add_trace(go.Bar(name="Consumer cost", x=labels, y=agg["buyer_cost"],
                         marker_color="#e74c3c", opacity=0.85))
    title_text = (f"Buyer strategy: {df['buyer_strategy'].iloc[0]}" if n_buyers == 1
                 else f"Averaged over {n_buyers} buyer strategies")
    fig.update_layout(barmode="group",
                      title=dict(text=title_text, font=dict(size=11), x=0.5),
                      legend=dict(orientation="h", y=1.1, font=dict(size=9)),
                      xaxis=dict(tickfont=dict(size=8)),
                      margin=dict(l=40, r=10, t=30, b=60),
                      plot_bgcolor="white", paper_bgcolor="white",
                      yaxis=dict(showgrid=True, gridcolor="#f0f0f0"))
    return fig


@callback(
    Output("hist-area", "children"),
    Input("load-hist-btn", "n_clicks"),
    prevent_initial_call=True,
)
def load_history(n_clicks):
    try:
        runs = persistence.read_table("runs")
        metrics = persistence.read_table("optimization_metrics")
    except Exception as e:
        return dmc.Text(f"Error reading DB: {e}", c="red", size="sm")
    if runs.empty:
        return dmc.Text("No runs saved yet.", c="dimmed", size="sm")
    if not metrics.empty and "HV" in metrics.columns:
        hv = metrics.groupby("run_id")["HV"].mean().reset_index()
        hv.columns = ["run_id", "HV_medio"]
        runs = runs.merge(hv, on="run_id", how="left")
    show_cols = [c for c in ["run_id", "scenario", "demand_kw", "gen_solar_kw",
                              "gen_wind_kw", "HV_medio"] if c in runs.columns]
    th_s = {"padding": "4px 8px", "textAlign": "left",
             "borderBottom": "1px solid #dee2e6", "fontSize": "12px"}
    td_s = {"padding": "3px 8px", "fontSize": "12px"}
    table = html.Table(
        [html.Thead(html.Tr([html.Th(c, style=th_s) for c in show_cols]))] +
        [html.Tbody([html.Tr([html.Td(
            str(row[c]) if pd.notna(row[c]) else "—", style=td_s) for c in show_cols])
            for _, row in runs.iterrows()])],
        style={"width": "100%", "borderCollapse": "collapse"},
    )
    children = [table]
    if "HV_medio" in runs.columns and runs["HV_medio"].notna().any():
        fig_hv = go.Figure(go.Scatter(
            x=list(range(1, len(runs) + 1)), y=runs["HV_medio"].tolist(),
            mode="lines+markers", line=dict(color="#3498db", width=2), marker=dict(size=7),
            text=runs["scenario"].tolist() if "scenario" in runs.columns else None,
            hovertemplate="Run %{x}<br>Scenario: %{text}<br>HV: %{y:.4f}<extra></extra>",
        ))
        fig_hv.update_layout(xaxis_title="Run #", yaxis_title="HV medio",
                              margin=dict(l=40, r=10, t=30, b=40),
                              plot_bgcolor="white", paper_bgcolor="white",
                              title="Mean Hypervolume per run")
        children.append(dcc.Graph(figure=fig_hv,
                                  style={"height": "260px", "marginTop": "16px"}))
    return html.Div(children)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
