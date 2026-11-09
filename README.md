<p align="center">
  <img src="assets/apex-pulse-logo.png" alt="Apex Pulse logo" width="420">
</p>

<p align="center">
  <strong>Production-deployed machine-learning system for Formula 1 qualifying prediction.</strong><br>
  Checkpoint-safe forecasting from free-practice data, with chronological validation, prospective monitoring and automated live operations.
</p>

<p align="center">
  <a href="https://apex-pulse-ten.vercel.app"><strong>Live Demo</strong></a>
  ·
  <a href="#results">Results</a>
  ·
  <a href="#architecture">Architecture</a>
  ·
  <a href="#quick-start">Quick Start</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastF1-data-E10600" alt="FastF1" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Next.js-frontend-000000?logo=nextdotjs&logoColor=white" alt="Next.js" />
  <img src="https://img.shields.io/badge/Railway-backend-0B0D0E?logo=railway&logoColor=white" alt="Railway" />
  <img src="https://img.shields.io/badge/Vercel-frontend-000000?logo=vercel&logoColor=white" alt="Vercel" />
</p>

---

## Overview

**Apex Pulse** predicts Formula 1 qualifying performance using only information that would have been available when the forecast was issued.

Predictions are supported after **FP1, FP2 and FP3**. The primary target is each driver's **qualifying gap to pole in seconds**, from which the predicted qualifying order is derived.

The project extends beyond offline model training: it includes guarded data ingestion, chronological backtesting, policy governance, prospective replay, immutable forecast and settlement records, an autonomous race-weekend orchestrator, a read-only API and a deployed public frontend.


|                       |                                                                |
| --------------------- | -------------------------------------------------------------- |
| Historical scope      | **2023–2025**                                                  |
| Conventional weekends | **44**                                                         |
| Modeling rows         | **2,634**                                                      |
| Dataset columns       | **182**                                                        |
| Drivers / teams       | **28 / 11**                                                    |
| Walk-forward folds    | **39 / 39**                                                    |
| Data sources          | **OpenF1 free historical sessions; FastF1 calendar and prior data** |
| Public app            | [apex-pulse-ten.vercel.app](https://apex-pulse-ten.vercel.app) |

<p align="center">
  <img
    src="assets/screenshots/apex-pulse-dashboard.png"
    alt="Apex Pulse qualifying forecast dashboard"
    width="1000"
  >
</p>

---

## Architecture

```mermaid
flowchart LR
    A[OpenF1 historical sessions / preserved FastF1 data] --> B[Lap cleaning & validation]
    B --> C[Checkpoint-safe features]
    C --> D[Historical modeling dataset]
    D --> E[Walk-forward evaluation]
    E --> F[Champion policy & governance]
    F --> G[Prospective monitoring]

    G --> H[Pre-Q forecast]
    H --> I[Immutable snapshot]
    I --> J[Post-Q validation]
    J --> K[Settlement & integrity checks]
    K --> L[Dashboard artifacts]

    L --> M[FastAPI · Railway]
    M --> N[Next.js · Vercel]

    O[5-minute scheduler] --> P[Weekend orchestrator]
    P --> G
```

The public API is intentionally **read-only**. Forecasting and settlement mutations remain behind the same guarded backend workflows used by the autonomous scheduler.

---

## Technical highlights

### Leakage-safe forecasting

The central data contract is simple:

> **A feature may encode only information available when the prediction would have been issued.**

This is enforced through:

* checkpoint-specific FP1 / FP2 / FP3 feature provenance;
* qualifying outcomes excluded from predictors;
* historical features built only from earlier events;
* complete-event train/test separation;
* chronological walk-forward backtesting;
* prior-fold-only model selection and uncertainty calibration.

An `after_fp2` prediction can therefore use FP1 and FP2, but never FP3 or qualifying information.

### F1-specific feature engineering

Raw fastest laps are noisy proxies for qualifying pace because practice programmes differ in fuel load, tyres, traffic, setup work and run intent.

Apex Pulse combines:

| Feature family       | Examples                                                  |
| -------------------- | --------------------------------------------------------- |
| **Practice pace**    | best/median lap, push-lap pace, theoretical best, sectors |
| **Relative pace**    | gap/rank to session best                                  |
| **Teammate context** | driver-to-teammate pace deltas                            |
| **Team context**     | team-best pace and within-team comparisons                |
| **Tyre context**     | compound, tyre life and stint information                 |
| **Historical form**  | prior driver/team qualifying performance                  |
| **Data quality**     | missingness, lap counts and extreme-signal flags          |

Push-like laps are identified with deterministic validity and pace rules rather than assuming every timed lap represents qualifying intent.

### Qualifying roster and practice evidence

The pre-qualifying workflow keeps two event-scoped populations separate:

* `forecast_eligible_drivers` is the authoritative weekend roster that receives predictions;
* `practice_evidence_drivers` contains everyone who produced usable FP1/FP2/FP3 data.

On a conventional weekend the live workflow checks the latest pre-qualifying FIA entry list,
including revisions. An explicit event roster takes precedence. The latest completed practice
session supplies the roster when an official list is unavailable. When the FIA announces a driver
change and its revised list is pending, the scheduler retries the roster check.

All practice participants contribute to the pace aggregates for their team and session. The scoring
rows cover only the entrants. A confirmed replacement with no personal laps can receive a prediction
from usable practice pace for their team. Personal lap, sector and tyre features are populated only
from that driver's own running for the team. A team-based estimate has less driver-specific evidence.

The immutable forecast records the roster, source and scoring inputs available at publication.
Changes announced afterward appear as partial coverage when qualifying results arrive. Evaluation
uses the drivers with matching predictions and outcomes. A missed publication window stays recorded
as missed.

### Model and policy governance

The evaluation framework includes:

* robust practice baselines;
* Ridge Regression;
* Random Forest;
* HistGradientBoostingRegressor;
* feature-group ablations;
* temporal weighting;
* static and nested champion policies;
* guarded policy switching;
* conformal uncertainty estimates.

Model selection is checkpoint-specific because the information regime changes substantially between FP1 and FP3.

Most importantly, **the model with the strongest retrospective aggregate metric is not automatically promoted to production**.

### Prospective evaluation

Historical walk-forward backtesting is complemented by frozen-policy and retrain-based prospective replay.

For each held-out event, the system can reconstruct the legal training history available before that event, generate predictions, preserve diagnostic shadow candidates separately and reveal qualifying outcomes only at settlement time.

This separates retrospective model discovery from deployment-policy evidence.

---

## Results

FP3 is the strongest ML checkpoint: by then practice contains substantially more representative qualifying signal than FP1 or FP2.

### Season-aware FP3 experiment

A fixed comparison evaluated the same Random Forest with `base_plus_relative` features under two training policies on identical rows:

| Training policy               |      FP3 MAE | Scope                            |
| ----------------------------- | -----------: | -------------------------------- |
| Uniform historical training   |  **0.920 s** | 774 rows · 39 folds              |
| Current-season-aware training |  **0.729 s** | 774 rows · 39 folds              |
| Difference                    | **−0.191 s** | retrospective aligned comparison |

The gain is regime-dependent. Current-season weighting is most useful after enough races from the active season have accumulated; cold-start events show little benefit.

### Why the best retrospective candidate is not automatically deployed

Frozen season-held-out evaluation produced mixed evidence:

| Split                       | Static FP3 MAE | Season-aware FP3 MAE |
| --------------------------- | -------------: | -------------------: |
| Train 2023 → Test 2024      |    **0.946 s** |              0.951 s |
| Train 2023–2024 → Test 2025 |        0.788 s |          **0.528 s** |

A stricter event-by-event retrain replay was more conservative and did not provide sufficient evidence to replace the default production policy.

**Historical production decision:** retain the stable policy while continuing prospective evaluation.

This distinction between *best historical candidate* and *deployed policy* is deliberate.

### Conditional FP3 policy V1 (implementation completed September 20, 2026)

The separately versioned `FP3_CONDITIONAL_POLICY_V1` uses Uniform with fewer than
five eligible prior current-season events and Season-aware from five onward. Both
candidates remain in shadow monitoring. Five is the existing training boundary;
it was not optimized for this deployment rule. The frozen 126-feature candidates
reproduce aligned 2024 MAEs of 0.945950/0.716471 and 2025 MAEs of 0.788273/0.527566
(Uniform/Season-aware). These fixed-candidate scores differ from the earlier
deployment-selector evaluation above.

The implementation corrects missing current-season training history, preserves
the historical feature contract, and separates new post-freeze evidence from
historical replay. The initial September implementation was not deployed; the
October 27 investigation found the September 5 image still running. Runtime
activation and per-forecast selection are now exposed independently. See the
[audit, activation semantics, fallback and rollback instructions](docs/fp3_conditional_policy_v1.md).
The [October availability audit](docs/prediction_incident_2026_10_27.md) explains
the Singapore exclusion, missing USA timing, integrity correction and deployment
checks. Git commit/push alone does not deploy the CLI-managed Railway backend.
Inspect the configured policy without modifying artifacts with
`python -m f1_prediction.cli fp3-policy-report`.

---

## Production workflow

Apex Pulse runs as a single-writer production system.

### Automated race-weekend lifecycle

Every **5 minutes**, the scheduler invokes a deterministic one-shot weekend orchestrator.

It:

1. identifies the current/next Formula 1 event;
2. distinguishes scheduled session completion from actual public-data availability;
3. waits for the required practice evidence;
4. invokes the guarded pre-qualifying workflow;
5. preserves the forecast as an immutable snapshot;
6. waits for qualifying data;
7. validates targets and driver-set parity;
8. settles the forecast;
9. refreshes validated dashboard artifacts.

Repeated ticks are idempotent. Existing forecasts and settlements are reused rather than regenerated.

Sprint/non-standard weekends are currently treated as unsupported safe no-ops rather than being remapped onto incompatible FP sessions.

### Integrity guarantees

Historical monitored events are protected with event-scoped semantic fingerprints covering forecast, settlement, registry, entry-list, feature/target and dashboard evidence.

Legitimate new events may be appended. Mutation or deletion of previously recorded events is treated as a blocking integrity failure.

A 14-step network-independent rehearsal tests the full state-machine lifecycle without mutating production records.

### Deployment

**Backend**

* FastAPI
* Docker
* Railway
* persistent runtime volume
* 5-minute scheduler
* read-only public API

**Frontend**

* Next.js
* React / TypeScript
* Tailwind CSS
* Vercel
* automatic server-data refresh
* responsive dark/light interface

The browser never downloads FastF1 data, trains models or creates forecasts.

---

## Quick start

### Install

```bash
git clone https://github.com/matteosgobba/apex-pulse.git
cd apex-pulse

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Inspect available commands:

```bash
python -m f1_prediction.cli --help
```

### Build the historical dataset

```bash
python -m f1_prediction.cli build-season-dataset \
  --seasons 2023 2024 2025 \
  --preset conventional
```

Generated data and model artifacts are intentionally excluded from Git.

### Run a chronological evaluation

```bash
python -m f1_prediction.cli dataset-report
python -m f1_prediction.cli evaluate-baselines

python -m f1_prediction.cli ablation-backtest \
  --strategy walk_forward \
  --temporal-weighting uniform \
  --min-events 10 \
  --min-train-events 5

python -m f1_prediction.cli backtest-report
```

### Run the public app locally

Backend:

```bash
python -m f1_prediction.cli dashboard-export
python -m f1_prediction.cli dashboard-api
```

Frontend:

```bash
cd web
npm install
cp .env.example .env.local
npm run dev
```

Then open:

```text
http://localhost:3000
```

### Inspect the autonomous weekend state

```bash
python -m f1_prediction.cli autopilot-tick --dry-run --json
```

Dry-run mode performs event discovery and readiness inspection without mutating operational artifacts.

### Free session ingestion after the archive access change

Since October 2026, the F1 timing archive used by FastF1 requires a paid F1 TV
subscription ([upstream incident](https://github.com/theOehrly/Fast-F1/issues/978)).
The project remains subscription-free: `configs/data.yaml` selects
`session_data_source: openf1` for new downloads. FastF1 still supplies the calendar;
existing training data, forecasts and cached raw sessions keep their original source.

```bash
python -m f1_prediction.cli load-session --season 2026 --event "Mexico City Grand Prix" --session FP3
python -m f1_prediction.cli ingest-event --season 2026 --event "Mexico City Grand Prix" --sessions FP1 FP2 FP3
```

These commands save historical laps only. They cannot recover a missed prospective
forecast. Live forecasts remain restricted to the window before qualifying.

OpenF1 historical access is free from 30 minutes after a session ends. The adapter
waits 31 minutes and the scheduler starts its probes after 35 minutes. Requests are
spaced below the free rate limit, cached locally, and never authenticated. Live-window
restrictions and unpublished data remain retryable; errors/empty responses are not cached.

The versioned `OPENF1_HISTORICAL_LAPS_V1` adapter matches year, location, session type
and UTC start exactly. It preserves lap/sector times, maps tyre life from observed stint
age, and excludes pit laps, identified deleted laps, red-flag/SC/VSC laps, and laps
with inconsistent sector sums. Missing driver identity can be recovered only from an
earlier practice session in the same meeting. Practice loads never request qualifying
or race performance. Every ingested session records its source and adapter version;
new forecast snapshots also freeze source metadata and input hashes.

`IsAccurate` in adapted data is explicitly a sector-consistency/quality **proxy**,
not FastF1's own accuracy signal. If a deletion cannot be assigned to one lap, that
driver's timing is conservatively excluded from valid/push pace, while participation
is preserved. Missing mini-sector samples alone do not invalidate complete sectors.
Coverage and aggregates may differ from the FastF1 training distribution; prior
backtest accuracy does not establish accuracy after this source migration.
See [the investigation and validation report](docs/prediction_incident_2026_11_02.md).

---

## Repository layout

```text
.
├── configs/                    # Data, feature, model and scheduler configuration
├── data/                       # Generated raw/interim/processed data
├── deploy/                     # Production runtime setup
├── docs/                       # Operational runbooks
├── models/                     # Generated model artifacts
├── reports/
│   ├── dashboard/              # Validated public JSON artifacts
│   ├── figures/                # Evaluation figures
│   └── metrics/                # Backtests, diagnostics and monitoring state
├── src/f1_prediction/
│   ├── data/                   # FastF1 ingestion and onboarding
│   ├── features/               # Cleaning and feature engineering
│   ├── modeling/               # Models, backtests and live workflows
│   ├── dashboard/              # Public artifact export
│   └── dashboard_api/          # Read-only FastAPI service
├── tests/
├── web/                        # Next.js frontend
├── Dockerfile
└── pyproject.toml
```

---

## Verification

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .

cd web
npm test
npm run lint
npm run typecheck
npm run build
```

Latest production-hardening verification:

```text
653 Python tests passed
98 frontend tests passed
Ruff / ESLint / TypeScript checks passed
Production builds passed
14-step autonomous workflow rehearsal passed
```

---

## Current status

**Production deployed · Phase A live validation complete.**

The scheduler, persistent runtime, API, frontend and integrity observers are operational.

The remaining validation phase intentionally waits for the first naturally selected **conventional FP1/FP2/FP3/Q weekend** to pass through the complete production lifecycle. No artificial live forecast or settlement is created merely to mark validation as complete.

---

## Limitations

Apex Pulse relies on public Formula 1 data and cannot observe several important latent variables, including:

* fuel load;
* exact engine and energy-deployment modes;
* setup configuration;
* private tyre temperatures;
* proprietary simulation data;
* team strategy instructions.

Practice run intent is also latent, and apparently similar laps may have been produced under very different programmes.

Current system limitations include:

* no Sprint-format forecasting;
* historical modeling focused on conventional 2023–2025 weekends;
* simplified qualifying-progression targets;
* weaker early-season evidence for season-aware policies;
* uncertainty calibration remains sensitive to outlier regimes.

These constraints are surfaced explicitly rather than corrected retrospectively.

---

## Tech stack

**ML & data:** Python · FastF1 · pandas · NumPy · scikit-learn · PyArrow · Parquet · joblib · matplotlib

**Backend & operations:** FastAPI · Uvicorn · Typer · Docker · Railway

**Frontend:** Next.js · React · TypeScript · Tailwind CSS · Vercel

**Quality:** Pytest · Ruff · ESLint · TypeScript

---

## Data and trademark notice

Apex Pulse uses [OpenF1](https://openf1.org/) for free historical session downloads
and [FastF1](https://github.com/theOehrly/Fast-F1) for calendars and preserved prior data.
No paid API or F1 TV subscription is required by the production workflow.

Formula 1, team names, driver names, logos and related marks belong to their respective owners. Apex Pulse is an independent educational and research project and is not affiliated with Formula 1, the FIA or any Formula 1 team.

---

## Author

**Matteo Sgobba**
M.Sc. Data Science and Engineering — Politecnico di Torino

[GitHub](https://github.com/matteosgobba) ·
[LinkedIn](https://www.linkedin.com/in/matteosgobba/) ·
[Live Apex Pulse](https://apex-pulse-ten.vercel.app/)
