# XAI-MAS Microgrid

**Explainable Multi-Agent Optimization for Renewable Energy Markets**

XAI-MAS Microgrid is a reproducible Python project for renewable microgrid management. It combines solar and wind power forecasting, multi-objective market optimization, FIPA-ACL Contract Net negotiation, per-agent battery simulation, explainable AI, orchestration, persistence and interactive dashboards.

The system models a small renewable microgrid with three agents:

- **Solar Agent (AS)**: photovoltaic generation seller.
- **Wind Agent (AE)**: wind generation seller.
- **Consumer Agent (AC)**: active energy buyer and negotiator.

The central idea is that the optimizer and the decentralized negotiation share the same market-clearing rule. This makes the negotiated outcomes directly comparable with the centralized Pareto-optimal reference.

## Project Scope

This repository implements the experimental system described in:

> **XAI-MAS Microgrid: Explainable Multi-Agent Optimization for Renewable Energy Markets**

The project integrates four technical layers:

1. **Forecasting**: pretrained Random Forest models estimate available solar and wind power.
2. **Optimization**: evolutionary multi-objective algorithms solve the cost-aware market-clearing problem.
3. **Multi-agent negotiation**: FIPA-ACL Contract Net interactions simulate decentralized bargaining between producers and consumer.
4. **Explainability and data intelligence**: SHAP, PDP, ALE, LIME, H-statistics, counterfactuals, Polars validation, Prefect orchestration and dashboards make the system inspectable and reproducible.

## Main Findings Reproduced by the Code

The experiments implemented in this repository support the following conclusions:

- With zero marginal generation cost, the objective space degenerates into a flat Pareto plane. Adding convex economic-dispatch costs and dispatch freedom produces a genuinely curved Pareto front with a meaningful knee point.
- On the curved front, the algorithms separate by paradigm: **SPEA2** provides the best coverage indicators, while **MOEA/D** achieves the tightest convergence.
- Modeling the consumer as an active bargaining agent reshapes the seller game. Under a smart opponent-modeling consumer, mutual honesty becomes a Nash equilibrium aligned with the social optimum.
- Per-agent batteries operated within a **20%--80% SoC band** reduce unmet demand over the daily horizon and shave the evening peak.
- Successfully served decentralized negotiations lie very close to the centralized cost-aware Pareto front.
- The forecasting models remain physically interpretable: solar predictions are mainly driven by radiation and temporal features, while wind predictions are mainly driven by wind speed.

## Reproducibility Status

The repository intentionally contains only source code, configuration files and the minimum folder structure.

With the current repository contents, you can:

- install the Python environment;
- inspect and run code that does not require local data or trained models;
- recreate the expected project structure;
- execute the full workflow once the required local artifacts are copied into place.

With the current repository contents, you cannot reproduce the full end-to-end flow immediately, because the following external files are not versioned:

```text
data/DatosEolicos.csv
data/DatosSolares.csv
models/modelo_eolico.pkl
models/modelo_solar.pkl
```

This is intentional. These files are local inputs or trained artifacts and are ignored by Git. To reproduce the complete project on another machine, copy the four files into the paths shown above before running inference, validation, optimization, MAS experiments, xAI, the Prefect pipeline or the dashboards.

## Repository Structure

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

## Environment Setup

Python 3.13 is the recommended version. The `scikit-learn` version is pinned in `requirements.txt` because the serialized Random Forest models must be loaded with a compatible version.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Prepare Local Inputs

Copy the datasets into `data/`:

```text
data/
|-- DatosEolicos.csv
`-- DatosSolares.csv
```

Copy the trained models into `models/`:

```text
models/
|-- modelo_eolico.pkl
`-- modelo_solar.pkl
```

Without these files, commands that load scenarios, forecasts or trained models will fail with `FileNotFoundError`.

## First Sanity Check

After placing the datasets and trained models, run the inference check first:

```powershell
python -m src.common.inference
```

This validates that the CSV files exist, that the expected columns are present and that the `.pkl` models can be loaded.

## Validate Forecasting Models

```powershell
python -m src.common.validate
```

This generates validation metrics and plots under:

```text
results/validation/
```

## Run Optimization Experiments

Run the modules from the project root:

```powershell
python -m src.optimization.run_optimization --timestamp "2017-06-15 19:00:00"
python -m src.optimization.stats --timestamp "2017-06-15 19:00:00" --seeds 15
python -m src.optimization.decision --timestamp "2017-06-15 19:00:00"
```

The optimization layer evaluates the market-clearing problem, computes Pareto fronts and compares multi-objective algorithms such as NSGA-II, NSGA-III, SPEA2 and MOEA/D.

Generated CSV files, figures and logs are stored under `results/`.

## Run Multi-Agent System Experiments

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

These commands cover:

- seller strategy tournaments;
- active consumer bargaining;
- Nash equilibrium analysis;
- information hiding and deception experiments;
- temporal battery arbitrage;
- FIPA-ACL message and ontology checks;
- robustness sweeps across time-of-day and seasonal scenarios.

## Run Explainability Experiments

```powershell
python -m src.xai.run_xai
```

Useful variants:

```powershell
python -m src.xai.run_xai --sample 1500 --model solar
python -m src.xai.run_xai --sample 1500 --model wind
```

The xAI layer produces explanations for the solar and wind Random Forest models, including:

- permutation importance;
- SHAP global and local explanations;
- PDP and ALE feature effects;
- LIME local explanations;
- Friedman H-statistics;
- glassbox baselines;
- surrogate trees;
- counterfactual explanations.

Generated outputs are saved under:

```text
results/xai/
```

## Run the Full Pipeline

```powershell
python -m src.pipeline.flow --timestamp "2017-06-15 19:00:00"
```

The Prefect flow executes the full sequence:

```text
check_data -> ingest_forecast -> optimize -> negotiate -> persist -> explain
```

By default, the pipeline stores results in:

```text
results/microgrid.db
```

The SQLite database contains the persisted runs used by the dashboards and history views.

## Dashboards

Streamlit dashboard:

```powershell
streamlit run src/dashboard/app.py
```

Dash Mantine dashboard:

```powershell
python src/dashboard/app_mantine.py
```

The dashboards expect the generated artifacts in `results/`. They allow users to explore scenarios, inspect Pareto fronts, compare algorithms, review xAI reports, run sensitivity checks and replay stored executions.

## Docker and Prefect

The Docker setup also requires the local `data/` and `models/` folders to contain the four required files before running the complete workflow.

```powershell
docker compose up -d prefect-server
docker compose run --rm pipeline
```

The Prefect UI is available at:

```text
http://localhost:4200
```

## Non-Versioned Files

The following files and folders are generated or copied locally and are ignored by Git:

- `data/*`: real or downloaded datasets.
- `models/*`: trained models (`.pkl`, `.joblib`, `.pt`, `.h5`, etc.).
- `results/*`: metrics, figures, SQLite databases and generated reports.
- `.venv/`, `.prefect/`, `.prefect_home/`, caches, checkpoints and logs.

The `.gitkeep` files preserve the empty folder structure after cloning.

## Reproducibility Notes

After the repository cleanup:

- `src/` contains reusable Python source code.
- `data/`, `models/` and `results/` are empty except for `.gitkeep`.
- No datasets, trained models, generated results or delivery-only auxiliary files are versioned.
- The repository avoids local absolute paths.
- The Python files in `src/` should be syntax-valid before execution.
- `python -m src.common.inference` is expected to fail with `FileNotFoundError` until the required datasets and trained models are copied into place.

## Code Availability

Repository URL:

```text
https://github.com/jorge-g-mateo/XAI-MAS-Microgrid.git
```
