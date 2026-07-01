# XAI-MAS Microgrid: Explainable Multi-Agent Optimization for Renewable Energy Markets

**Authors:**  
Gontzal Arregi Aranoa<sup>1</sup>, Jorge García-Mateo<sup>1</sup>, Miguel Iriso Soret<sup>1</sup>, Julen Larrañaga Karrera<sup>1</sup>

<sup>1</sup> Mondragon Unibertsitatea, AS Fabrik -- Goi Eskola Politeknikoa, Bilbao, Spain

---

## 1. Overview

This repository presents **XAI-MAS Microgrid**, a reproducible Python project for explainable renewable microgrid management.

The system combines **solar and wind power forecasting**, **multi-objective market optimization**, **FIPA-ACL Contract Net negotiation**, **per-agent battery simulation**, **explainable AI**, **data-quality validation**, **orchestration**, **persistence** and **interactive dashboards**.

The microgrid is modeled with three agents:

* **Solar Agent (AS)**: photovoltaic generation seller.
* **Wind Agent (AE)**: wind generation seller.
* **Consumer Agent (AC)**: active energy buyer and negotiator.

The central idea is to use a **shared market-clearing rule** for both the centralized optimizer and the decentralized multi-agent negotiation. This allows negotiated outcomes to be directly compared with the centralized Pareto-optimal reference.

The proposed system integrates four technical layers:

* **Forecasting** with pretrained Random Forest models.
* **Multi-objective optimization** of a cost-aware renewable energy market.
* **Multi-agent negotiation** through FIPA-ACL Contract Net messages.
* **Explainability and data intelligence** through SHAP, PDP, ALE, LIME, H-statistics, counterfactuals, Polars, Prefect, SQLite and dashboards.

The result is not just a collection of independent modules, but a single experimental platform where forecasting, optimization, negotiation, storage and explainability are evaluated under the same market definition.

---

## 2. Method

The system first forecasts the available renewable generation for the solar and wind agents. Since the two original targets use different units, the project harmonizes them into kW:

$$
P_{solar} = \frac{SystemProduction}{1000}
$$

$$
P_{wind} = \frac{Power}{100} \cdot P_{rated}
$$

where \(P_{rated}=10\) kW is used for the wind generator so that both renewable agents have comparable market capacity.

The market is formulated as a single-timestep multi-objective problem with decision variables:

$$
x = (q_s, q_w, p_s, p_w)
$$

where:

* \(q_s\) and \(q_w\) are the offered quantities of the solar and wind agents,
* \(p_s\) and \(p_w\) are their unit prices,
* the consumer buys the cheapest accepted energy first,
* unmet demand is penalized through a shortfall term.

Each generator has a convex economic-dispatch cost:

$$
c_i(q) = m_i q + \frac{1}{2} b_i q^2
$$

The optimization objectives are:

$$
\max \ \pi_s = q_s^{sold}p_s - c_s(q_s^{sold})
$$

$$
\max \ \pi_w = q_w^{sold}p_w - c_w(q_w^{sold})
$$

$$
\min \ C_{AC} = \sum_i q_i^{sold}p_i + \lambda s
$$

where:

* \(\pi_s\) is the solar seller profit,
* \(\pi_w\) is the wind seller profit,
* \(C_{AC}\) is the consumer cost,
* \(s\) is the unmet-demand shortfall,
* \(\lambda\) is the shortfall penalty.

A key methodological point is that **zero marginal cost makes the objectives linearly dependent**, producing a degenerate Pareto plane. The project therefore introduces **convex dispatch costs** and **dispatch freedom**, which recover a curved Pareto front with a meaningful knee point.

The decentralized negotiation layer uses the same market-clearing rule. The consumer sends a **CFP**, generators answer with **PROPOSE** or **REFUSE**, and the consumer replies through **ACCEPT/REJECT-PROPOSAL** messages. This makes the multi-agent outcome measurable against the centralized optimum.

---

## 3. Key Contributions

* Unified market formulation shared by centralized optimization and decentralized negotiation.
* Cost-aware renewable microgrid model with convex economic-dispatch costs.
* Identification of the degenerate zero-cost Pareto plane and recovery of a curved Pareto front.
* Comparison of NSGA-II, NSGA-III, SPEA2 and MOEA/D under a non-degenerate objective space.
* FIPA-ACL Contract Net negotiation with explicit seller and buyer strategies.
* Analysis of how active consumer bargaining reshapes Nash equilibria.
* Per-agent battery simulation with a 20%–80% state-of-charge operating band.
* Explainability analysis of the solar and wind Random Forest predictors.
* Reproducible orchestration with Prefect, Docker, Polars data validation and SQLite persistence.
* Interactive dashboards for Pareto exploration, algorithm comparison, xAI reports, sensitivity analysis and execution history.

---

## 4. Experimental Setup

The experimental system uses two pretrained forecasting models:

| Model | Target | Main drivers |
|---|---|---|
| Solar Random Forest | Solar production | `Radiation`, `hour`, `dayofyear` |
| Wind Random Forest | Wind production | `windspeed_100m` |

The optimization layer compares four evolutionary multi-objective algorithms:

| Algorithm | Paradigm |
|---|---|
| NSGA-II | Dominance-based |
| NSGA-III | Reference-direction-based |
| SPEA2 | Strength Pareto and density-based |
| MOEA/D | Decomposition-based |

The multi-agent layer evaluates six seller strategies:

| Strategy | Type |
|---|---|
| Honest | Reactive |
| Information hiding | Reactive |
| Deception | Reactive |
| Opponent modeling | Deliberative |
| Tit-for-tat | Deliberative |
| Bayesian opponent modeling | Deliberative |

The consumer is also modeled as an active participant through four buyer strategies:

| Buyer strategy | Behaviour |
|---|---|
| Price taker | Passive buyer |
| Honest buyer | High reservation price to guarantee supply |
| Hard bargainer | Aggressive low-price negotiation |
| Opponent-modeling buyer | Estimates seller marginal floors |

The explainability layer combines:

* permutation importance,
* SHAP summaries and local explanations,
* PDP and ALE curves,
* Friedman H-statistics,
* LIME explanations,
* glassbox baselines,
* surrogate trees,
* counterfactual explanations.

---

## 5. Results Summary

The experiments show that the proposed system becomes meaningful only when the market includes realistic dispatch economics. With zero marginal generation cost, the objectives become linearly dependent and the Pareto front degenerates into a flat surplus-division plane. Adding convex economic-dispatch costs and dispatch freedom produces a curved Pareto front with a real knee point.

### Multi-objective optimization

On the curved front, the algorithms separate clearly by paradigm. Higher HV is better, while lower GD, IGD, ε and Spread indicate better convergence, approximation or diversity.

| Algorithm | HV | GD | IGD | ε | Spread Δ |
|---|---:|---:|---:|---:|---:|
| NSGA-II | 82,791 | 0.55 | 0.086 | 8.82 | 0.55 |
| NSGA-III | 81,484 | 1.41 | 0.079 | 7.72 | 0.65 |
| SPEA2 | **86,396** | 1.05 | **0.063** | **6.03** | **0.18** |
| MOEA/D | 80,567 | **0.35** | 0.072 | 7.51 | 0.62 |

**SPEA2** provides the strongest coverage-oriented performance, achieving the best HV, IGD, ε and Spread. **MOEA/D** achieves the best GD, showing tighter convergence to the reference front but weaker coverage.

This distinction is only visible once the market is non-degenerate. On the zero-cost plane, GD is identically zero for all algorithms and the comparison is not informative.

### Multi-agent negotiation

The negotiation experiments show that the consumer strategy is central to the market outcome.

| Finding | Evidence | Interpretation |
|---|---|---|
| Passive consumers enable inefficient seller equilibria | Under a price-taker consumer, sellers can keep prices high | The buyer cannot be treated as a passive sink |
| Active bargaining reduces consumer cost | Opponent-modeling buyers force seller concessions | Buyer intelligence reshapes the market |
| Over-aggressive bargaining is risky | Hard bargainers can trigger supply shortfall | Minimizing price too aggressively may destroy feasibility |
| Smart consumers promote cooperation | Under an opponent-modeling buyer, honest-honest becomes a Nash equilibrium aligned with the social optimum | Cooperation becomes individually rational |

The main result is that market inefficiency is not an unavoidable consequence of multi-agent competition. It depends strongly on whether the consumer is modeled as an active strategic agent.

### Storage and temporal behavior

Adding per-agent batteries changes the daily dynamics of the microgrid. Batteries operate within a **20%–80% SoC band**, store midday renewable surplus and discharge during the evening peak.

This reduces unmet demand from **26.7 kWh** to **16.7 kWh**, corresponding to a reduction of approximately **37%**. The result shows that storage turns static scarcity into a temporally manageable resource.

### Explainability

The xAI analysis confirms that the forecasting layer is physically coherent:

* solar predictions are mainly driven by **`Radiation`**, **`hour`** and seasonal variables;
* wind predictions are mainly driven by **`windspeed_100m`**;
* ALE curves reveal correlation effects that PDP tends to smooth;
* local explanations and counterfactuals rely on the same dominant variables identified by global explanations.

Overall, the system is not only optimized and negotiated, but also inspectable: the forecasts, market decisions and pipeline outputs can be explained and audited.

---

## 6. Installation

Recommended environment:

* Python 3.11 or 3.12
* scikit-learn version compatible with the serialized Random Forest models
* Docker and Docker Compose for containerized execution
* Prefect for orchestration

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Windows PowerShell setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. Usage

### Preparing local inputs

The repository does not version datasets or trained models. Before running the full workflow, place the required files in the following paths:

```text
data/DatosEolicos.csv
data/DatosSolares.csv
models/modelo_eolico.pkl
models/modelo_solar.pkl
```

Without these files, commands that depend on data or pretrained models are expected to fail with `FileNotFoundError`.

### Running the first sanity check

```powershell
python -m src.common.inference
```

This validates that the datasets are available, that the expected columns are present and that the serialized models can be loaded.

### Validating the forecasting models

```powershell
python -m src.common.validate
```

Generated validation outputs are stored in:

```text
results/validation/
```

### Running optimization experiments

```powershell
python -m src.optimization.run_optimization --timestamp "2017-06-15 19:00:00"
python -m src.optimization.stats --timestamp "2017-06-15 19:00:00" --seeds 15
python -m src.optimization.decision --timestamp "2017-06-15 19:00:00"
```

These commands generate Pareto-front results, algorithm comparisons and operating-point analyses.

### Running multi-agent experiments

```powershell
python -m src.mas.run_mas --timestamp "2017-06-15 19:00:00"
python -m src.mas.sweep --no-optimum
python -m src.mas.dynamics --solar opponent_modeling --wind opponent_modeling
python -m src.mas.game_analysis
python -m src.mas.reciprocity
python -m src.mas.opponent_modeling_experiment
python -m src.mas.information_experiment
python -m src.mas.battery
python -m src.mas.ontology
```

These commands cover seller strategy tournaments, buyer bargaining, Nash-equilibrium analysis, information hiding, reciprocity, storage behavior and message-ontology checks.

### Running explainability experiments

```powershell
python -m src.xai.run_xai
```

Useful variants:

```powershell
python -m src.xai.run_xai --sample 1500 --model solar
python -m src.xai.run_xai --sample 1500 --model wind
```

Generated xAI outputs are stored in:

```text
results/xai/
```

### Running the full Prefect pipeline

```powershell
python -m src.pipeline.flow --timestamp "2017-06-15 19:00:00"
```

The full flow is:

```text
check_data -> ingest_forecast -> optimize -> negotiate -> persist -> explain
```

By default, results are persisted to:

```text
results/microgrid.db
```

### Running the dashboards

Streamlit dashboard:

```powershell
streamlit run src/dashboard/app.py
```

Dash Mantine dashboard:

```powershell
python src/dashboard/app_mantine.py
```

The dashboards expect generated artifacts to exist under `results/`.

### Running with Docker and Prefect

```powershell
docker compose up -d prefect-server
docker compose run --rm pipeline
```

The Prefect UI is available at:

```text
http://localhost:4200
```

---

## 8. Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- .dockerignore
|-- Dockerfile
|-- docker-compose.yml
|-- prefect.yaml
|-- src/
|   |-- common/          # data loading, feature engineering, inference and validation
|   |-- optimization/    # market model, Pareto optimization and quality indicators
|   |-- mas/             # FIPA-ACL negotiation, agent strategies and market experiments
|   |-- xai/             # model explainability and interpretability experiments
|   |-- pipeline/        # Prefect flow, orchestration and SQLite persistence
|   `-- dashboard/       # Streamlit and Dash dashboards
|-- data/                # local datasets, ignored except for .gitkeep
|-- models/              # local trained models, ignored except for .gitkeep
`-- results/             # generated outputs, ignored except for .gitkeep
```

---

## 9. Reproducibility Notes

This repository intentionally contains only source code, configuration files and the minimum folder structure.

Versioned files include:

* Python source code,
* configuration files,
* dependency specification,
* Docker and Prefect setup,
* empty `data/`, `models/` and `results/` folders preserved through `.gitkeep`.

Non-versioned files include:

* real or downloaded datasets,
* trained model files,
* generated metrics,
* generated figures,
* SQLite databases,
* logs,
* local virtual environments,
* caches and checkpoints.

This means that the repository can be installed and inspected immediately, but the full end-to-end workflow requires the external datasets and pretrained models listed above.

The expected clean-state behavior is:

* commands that do not require local datasets or trained models can be inspected or executed normally;
* commands that load `data/` or `models/` will fail until the required external files are copied into place;
* generated artifacts are written under `results/` and are not versioned.

---

## 10. Main Takeaways

* A renewable market with zero marginal cost collapses into a degenerate Pareto plane; realistic dispatch costs are necessary for meaningful optimization.
* Convex economic-dispatch costs and dispatch freedom produce a curved front with a real compromise region.
* SPEA2 is the strongest algorithm for front coverage, while MOEA/D is strongest for convergence.
* The consumer strategy is a structural component of the market: active bargaining can turn cooperation into an individually rational equilibrium.
* Distributed storage reduces scarcity by shifting renewable surplus across the day.
* The forecasting layer is interpretable and physically coherent.
* The complete system is reproducible through a single orchestrated and containerized pipeline.

Overall, **XAI-MAS Microgrid** demonstrates how forecasting, optimization, multi-agent negotiation, storage and explainability can be integrated into a single renewable energy market simulator.

---

## 11. License

No explicit open-source license has been provided yet. The repository is shared for academic and reproducibility purposes.
