<p align="center">
  <img src="assets/apex-pulse-logo.png" alt="Apex Pulse logo" width="420">
</p>

<p align="center">
  Machine-learning system for Formula 1 qualifying prediction from practice-session data.
</p>

<p align="center">
  <strong>Checkpoint-safe forecasting of Formula 1 qualifying pace from free-practice evidence.</strong><br />
  An event-level machine-learning pipeline using FastF1 session data.
</p>

<p align="center">
  <a href="#prediction-problem">Prediction problem</a> ·
  <a href="#feature-design-and-f1-rationale">Feature design</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#quick-start">Quick start</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white" alt="Python 3.13" />
  <img src="https://img.shields.io/badge/FastF1-session%20data-E10600" alt="FastF1" />
  <img src="https://img.shields.io/badge/pandas-data%20engineering-150458?logo=pandas&logoColor=white" alt="pandas" />
  <img src="https://img.shields.io/badge/scikit--learn-machine%20learning-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Typer-CLI-009688" alt="Typer" />
  <img src="https://img.shields.io/badge/Pytest-tested-0A9EDC?logo=pytest&logoColor=white" alt="pytest" />
  <img src="https://img.shields.io/badge/Ruff-linted-D7FF64?logo=ruff&logoColor=black" alt="Ruff" />
</p>

---

## Overview

**Apex Pulse** estimates Formula 1 qualifying performance using only signals observable by a selected free-practice checkpoint: after FP1, FP2, or FP3. The central output is a driver's **qualifying gap to pole** in seconds; qualifying position and Q2/Q3 progression are supported as secondary targets.

The task is deliberately framed as a constrained forecast rather than a post-hoc performance explanation. A prediction produced after FP2, for example, can depend on FP1 and FP2 data plus pre-existing historical information, but never on FP3 or qualifying outcomes. This restriction is enforced during dataset construction and validation.

```text
FastF1 session data
    ↓
Lap cleaning, session normalisation, and quality flags
    ↓
Driver × weekend × checkpoint feature table
    ↓
Checkpoint-safe baselines and ML candidates
    ↓
Chronological backtests, champion selection, and diagnostics
    ↓
Metrics, failure-case analysis, uncertainty estimates, reports
```

## Prediction problem

Each row represents one **driver–race weekend–forecast checkpoint**. The model must infer relative qualifying pace from practice runs whose underlying programmes are only partially observed.

| Element | Definition |
|---|---|
| Unit of prediction | Driver × event × checkpoint |
| Primary target | Qualifying gap to pole, in seconds |
| Secondary targets | Qualifying position; reached Q2; reached Q3 |
| Available checkpoints | After FP1, FP2, FP3 |
| Data source | FastF1 public session data with local cache |
| Experimental scope | Conventional 2023–2024 F1 weekends; expanded iteratively |
| Generalisation unit | A future race weekend, not a random driver row |

Qualifying is an appropriate target because it is a relatively concentrated performance setting: drivers target a short, low-fuel, high-grip run on soft tyres, with track evolution and session conditions becoming critical. Free practice is informative but imperfect: teams trade off setup exploration, tyre work, long-run simulation, traffic management, and race preparation. The modelling objective is therefore not to assume that the fastest practice lap *is* qualifying pace, but to recover a robust estimate from several imperfect observations.

## Feature design and F1 rationale

### Why raw fastest laps are insufficient

A single fastest lap is an unstable measurement. It can be inflated or suppressed by tyre compound, tyre age, fuel load, track evolution, traffic, red flags, aborted laps, changing weather, and an individual team's run plan. Comparing raw times across drivers also mixes car performance with circuit length and session-specific conditions.

Apex Pulse therefore turns lap-level records into checkpoint-safe **driver-level summaries** and contextual comparisons. The design goal is to preserve the parts of practice that are informative about qualifying while reducing sensitivity to one anomalous lap or one non-representative run.

### Feature families

| Feature family | Technical treatment | F1 interpretation |
|---|---|---|
| **Absolute pace** | Robust lap-time summaries, best/representative laps, quantiles, valid-lap counts, and recent-session signals | Captures the driver's observed one-lap ceiling and the consistency of the available pace sample. |
| **Tyre and stint context** | Compound usage, tyre-life summaries, stint structure, and counts conditional on available compounds | A soft-tyre, fresh-tyre effort is generally more indicative of qualifying intent than a worn medium/hard long run. Stint context prevents treating them as equivalent. |
| **Session-relative pace** | Gaps and ranks relative to session benchmarks; normalised pace features | Removes much of the circuit-length and session-level timing scale, making signals more comparable across venues and weekends. |
| **Teammate comparison** | Driver-minus-teammate pace and availability features within the same session | Teammates share the same car concept and operate under closely related track conditions, creating a useful local reference for driver execution and setup direction. |
| **Team signals** | Team-level pace aggregates and within-team context | Team pace is a partial proxy for machinery competitiveness and helps distinguish a strong individual lap in a weak package from a broadly competitive car. |
| **Historical form** | Lagged driver/team aggregates built only from prior eligible events | Carries stable information that a short or disrupted practice session may fail to reveal, without importing future-event outcomes. |
| **Data-quality and anomaly flags** | Missingness, low sample size, incomplete session, and extreme-signal indicators | Makes unreliable evidence visible to models and diagnostics rather than silently imputing confidence into a poor session. |

### Aggregation strategy

The pipeline does not pool every lap indiscriminately. It first applies cleaning and validity rules, then produces feature aggregates separately by session and checkpoint. A row at `after_fp2` can use FP1 and FP2 aggregates; an `after_fp1` row cannot inherit later-session information. Feature names and build paths retain checkpoint provenance so that data availability can be audited.

Robust summaries are preferred over a single minimum lap because the minimum is highly exposed to outliers. At the same time, best-lap-style features are retained because qualifying rewards peak one-lap pace. This combination lets the model weigh ceiling pace against sample reliability rather than hard-coding either interpretation.

Relative features are especially important for cross-event learning. A 0.3-second gap has different practical meaning at Monaco and Silverstone in raw absolute terms; pace relative to the session field, teammate, and team benchmark is more portable. Historical features are lagged at the event level to prevent an event from influencing its own forecast.

### Leakage controls

The data contract follows a simple rule: **a feature may only encode information that would have existed when the prediction was issued.** In practice, this means:

- qualifying results are targets only and never predictors;
- checkpoint datasets exclude later practice sessions;
- historical aggregates use previous eligible events only;
- splits are event-based and chronological, so rows from a held-out weekend never appear in training;
- model selection procedures are evaluated within backtest structure rather than tuned against the same future events being reported.

## Modelling and baselines

The system benchmarks regularised linear models, random forests, and boosted tabular candidates against credible pace-based baselines. Baselines are not included merely as a formality: early practice is often weakly informative, so a complex model must demonstrate out-of-sample value over simple, transparent estimates.

Model and feature-policy selection is checkpoint-specific. This reflects the fact that the information regime changes materially from FP1 to FP3: FP1 is sparse and exploratory, FP2 often contains the most representative qualifying simulations, and FP3 is closer to qualifying but can be unusually sensitive to interruptions or track-condition changes.

A champion-policy layer evaluates candidate model/feature combinations under static, nested, and replay-oriented selection procedures. Reports include residual and interval diagnostics, selection metadata, and event-level failure cases.

## Evaluation

Random train/test splits would leak weekend-level structure: drivers, teams, circuit conditions, and shared session context would be represented on both sides of the split. Apex Pulse instead evaluates at the event level.

- **Repeated event holdout** tests whether patterns transfer to unseen weekends.
- **Walk-forward backtesting** trains on earlier events and predicts subsequent events in chronological order.
- **Fold-consistent comparisons** ensure baselines and candidates are assessed on the same event set.
- **Ablations and diagnostics** identify whether gains come from pace, relative, historical, or contextual feature groups.
- **Uncertainty estimates** assess interval coverage and width rather than reporting only point-error metrics.

This is intentionally a difficult setting. A model that performs well only on clean, representative practice sessions is not enough; the reporting layer surfaces weak-signal weekends and conditions in which the forecast should be treated cautiously.

## Repository layout

```text
src/f1_prediction/          # Package source code and Typer CLI
configs/                    # Runtime and model configuration
tests/                      # Automated tests
data/                       # Local raw/interim/processed artefacts (ignored)
models/                     # Serialized local models (ignored)
reports/                    # Generated metrics and diagnostics
```

## Quick start

```bash
git clone git@github.com:matteosgobba/f1_project.git
cd f1_project
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python -m f1_prediction.cli --help
```

A representative multi-season workflow:

```bash
# Build driver-weekend-checkpoint datasets
python -m f1_prediction.cli build-season-dataset --seasons 2023 2024

# Audit feature availability and dataset quality
python -m f1_prediction.cli dataset-report

# Benchmark tabular candidates chronologically
python -m f1_prediction.cli backtest-tabular-models --strategy walk_forward

# Select and evaluate a checkpoint-specific champion policy
python -m f1_prediction.cli champion-backtest --strategy walk_forward

# Produce consolidated metrics and diagnostics
python -m f1_prediction.cli backtest-report
```

Use `--help` for command options available in the current checkout.

## Current status

**Active development.** The repository currently includes multi-season dataset construction, lap/session normalisation, checkpoint-specific feature policies, baseline and tabular-model backtesting, champion selection, uncertainty diagnostics, data-quality reporting, ablation analysis, and prospective replay eligibility checks.

Reported results are research-style backtest evidence, not a claim of reliable qualifying prediction in every Formula 1 context. Performance is expected to vary substantially with unobserved fuel loads, run plans, session disruption, weather, and changes in technical or sporting regimes.

## Limitations and planned extensions

- FastF1-accessible public data do not expose full fuel-load, setup, tyre-degradation, traffic, or proprietary simulation information.
- Practice programmes are latent: a lap can be quick for reasons unrelated to qualifying intent, or slow despite strong ultimate pace.
- The sample remains small relative to the diversity of circuits, regulations, weather states, driver line-ups, and team development trajectories.
- Next steps include broader seasonal coverage, richer track-condition proxies, calibration refinement, and continuing prospective event-by-event evaluation.

## Tech stack

**Python · FastF1 · pandas · NumPy · scikit-learn · Typer · joblib · Pytest · Ruff · Parquet**

## Acknowledgements

This project uses the [FastF1](https://github.com/theOehrly/Fast-F1) ecosystem for Formula 1 session data access. Formula 1 names, marks, and data remain the property of their respective owners. Apex Pulse is an independent educational project.

## Author

Matteo Sgobba

- M.Sc. Data Science and Engineering @ Politecnico di Torino
- Contact: matteo.sgobba@studenti.polito.it
- LinkedIn: https://www.linkedin.com/in/matteosgobba/
