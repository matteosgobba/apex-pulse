Formula 1 Qualifying Performance Prediction

This repository contains a machine learning project for predicting Formula 1 qualifying performance from free practice data, historical timing data, telemetry-derived features, weather information, tyre information, and circuit/session context.

The project is designed as an end-to-end, portfolio-quality machine learning system rather than a simple Formula 1 visualization dashboard.

The final objective is to build a system that can progressively update predictions after FP1, FP2, and FP3, estimating how each driver is likely to perform in qualifying.

The project name is intentionally left unspecified for now.

⸻

Project Goal

The goal is to predict qualifying outcomes using only information that would be available before qualifying.

The system should eventually estimate:

* predicted qualifying ranking;
* predicted gap to pole;
* probability of reaching Q2;
* probability of reaching Q3;
* probability of top 10, top 5, top 3, and pole;
* driver/team strengths and weaknesses;
* telemetry-based performance explanations;
* prediction confidence as more practice data becomes available.

The initial MVP focuses on:

1. collecting and caching historical F1 data;
2. building a clean practice-to-qualifying modeling dataset;
3. creating strong baseline models;
4. training tabular ML models;
5. evaluating predictions with historical backtesting.

⸻

Why This Project Matters

Formula 1 qualifying performance depends on many interacting factors:

* raw car pace;
* driver performance;
* teammate-relative pace;
* tyre compound and tyre age;
* track evolution;
* weather;
* circuit characteristics;
* sector-specific strengths;
* practice programme differences;
* traffic and lap representativeness.

Free practice sessions contain useful signals, but they are noisy. Not every practice lap is representative of qualifying pace. Some laps are push laps, others are race simulations, cooldown laps, out laps, in laps, laps affected by traffic, or laps run with different fuel loads and tyre conditions.

This project aims to extract useful predictive information from that noisy practice data and convert it into probabilistic qualifying forecasts.

⸻

Data Sources

The project uses only free or publicly accessible data sources.

Primary Source

FastF1 is the primary data source. It provides access to Formula 1 timing data, session data, lap data, weather data, results, and public telemetry.

FastF1 will be used for:

* session loading;
* lap times;
* sector times;
* tyre compounds;
* tyre age;
* session results;
* weather data;
* telemetry-derived features.

Optional Secondary Source

OpenF1 historical API may be used where helpful, but only in its free historical-access form.

The project must not depend on paid OpenF1 real-time access.

The initial “real-time” functionality will be implemented as historical replay: past practice sessions are replayed lap by lap, and predictions are updated as if the data were arriving live.

⸻

Important Limitations

This project uses public data. It does not have access to private team data.

The following signals are generally not available from public APIs and should not be treated as directly observable:

* real fuel load;
* exact engine mode;
* battery state of charge;
* MGU-K deployment;
* MGU-H behaviour;
* brake temperature;
* brake pressure;
* brake bias;
* tyre carcass temperature;
* tyre surface temperature;
* aerodynamic setup;
* team strategy instructions.

Energy management or brake management can only be studied through proxy signals such as speed traces, acceleration, throttle, brake flag, DRS usage, and straight-line speed deltas. Any such analysis must be explicitly described as proxy analysis under partial observability.

⸻

Core Prediction Tasks

1. Qualifying Gap Regression

Predict:

quali_gap_to_pole

This is the difference in seconds between a driver’s best qualifying lap and the pole lap.

Example:

Driver A: +0.000
Driver B: +0.084
Driver C: +0.231

2. Qualifying Ranking

Predict the final qualifying order.

This can be treated as:

* regression followed by sorting;
* direct ranking problem;
* pairwise ranking problem;
* learning-to-rank task.

3. Q3 Classification

Predict whether each driver reaches Q3.

Target:

reached_q3 = 1 or 0

Optional future classification targets:

reached_q2
top_10
top_5
top_3
pole

⸻

Prediction Checkpoints

The system should support predictions at different points of a race weekend:

pre_weekend
after_fp1
after_fp2
after_fp3

The MVP should implement:

after_fp1
after_fp2
after_fp3

Each modeling row should represent:

season + event + prediction_checkpoint + driver

Example:

2024 | Monza | after_fp2 | NOR | features... | targets...

⸻

Methodology Overview

The project follows this workflow:

1. Load historical practice and qualifying sessions.
2. Extract lap-level, session-level, weather, tyre, and telemetry-derived data.
3. Identify representative practice laps.
4. Build driver-level features for FP1, FP2, and FP3.
5. Build historical rolling features without data leakage.
6. Build qualifying targets.
7. Train baseline models.
8. Train machine learning models.
9. Evaluate predictions with historical backtesting.
10. Add explainability and uncertainty estimates.
11. Add historical replay and dashboard functionality.

⸻

Planned Repository Structure

.
├── README.md
├── AGENTS.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── configs/
│   ├── data.yaml
│   ├── features.yaml
│   └── model.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── notebooks/
│   └── exploratory/
├── reports/
│   ├── figures/
│   └── metrics/
├── models/
├── src/
│   └── f1_prediction/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data/
│       ├── features/
│       ├── modeling/
│       ├── replay/
│       ├── visualization/
│       └── utils/
└── tests/

⸻

Suggested Tech Stack

Main language:

Python

Core libraries:

fastf1
pandas
polars
numpy
pyarrow
duckdb
scikit-learn
xgboost
lightgbm
matplotlib
plotly
streamlit
typer
pydantic
pyyaml
joblib
pytest
ruff
black

The project should start simple. Advanced models and dashboards should be added only after the data pipeline and baseline models work correctly.

⸻

Setup

1. Clone the repository

git clone <repository-url>
cd <repository-name>

2. Create a virtual environment

python -m venv .venv
source .venv/bin/activate

On Windows:

.venv\Scripts\activate

3. Install the project and development dependencies

```bash
python -m pip install -e ".[dev]"
```

The project uses a `src/` layout, so the editable install is required before running the module command.

4. Load a historical practice session

```bash
python -m f1_prediction.cli load-session --season 2024 --event Monza --session FP2
```

The command initializes the FastF1 cache, downloads or reads the requested historical session, and writes a stable subset of lap-level data to:

```text
data/raw/laps/2024/monza/fp2_laps.parquet
```

Successful output has this form:

```text
Session loaded successfully
Season: 2024
Event: Italian Grand Prix
Session: Practice 2 (FP2)
Drivers: 20
Laps: <loaded lap count>
Output: <repository>/data/raw/laps/2024/monza/fp2_laps.parquet
```

FastF1 requires an internet connection the first time a session is loaded. Later runs can reuse cached responses.

### Ingest a race weekend

Milestone 2 adds batch ingestion for one historical event. By default, the command loads FP1, FP2, FP3, and Qualifying:

```bash
python -m f1_prediction.cli ingest-event --season 2024 --event Monza
```

Override the requested sessions by listing them after `--sessions`:

```bash
python -m f1_prediction.cli ingest-event --season 2024 --event Monza --sessions FP1 FP2 Q
```

Existing successful lap and metadata outputs are skipped. Use `--force` to reload and overwrite them:

```bash
python -m f1_prediction.cli ingest-event --season 2024 --event Monza --force
```

By default, a failed session is recorded and the remaining sessions continue. Use `--fail-fast` to stop after the first failure:

```bash
python -m f1_prediction.cli ingest-event --season 2024 --event Monza --fail-fast
```

Outputs are stored under:

```text
data/raw/laps/{season}/{event_slug}/{session_slug}_laps.parquet
data/raw/session_metadata/{season}/{event_slug}/{session_slug}_metadata.json
```

For example:

```text
data/raw/laps/2024/monza/fp2_laps.parquet
data/raw/session_metadata/2024/monza/fp2_metadata.json
```

Generated Parquet and metadata JSON files are ignored by Git. The `.gitkeep` files preserve the raw data directory structure.

### Build cleaned practice features

Milestone 3 builds a pre-modeling feature layer from the raw FP1, FP2, and FP3 Parquet files:

```bash
python -m f1_prediction.cli build-session-features --season 2024 --event Monza
```

Process a subset of practice sessions by listing them after `--sessions`:

```bash
python -m f1_prediction.cli build-session-features --season 2024 --event Monza --sessions FP2
```

Use `--force` to rebuild the requested cleaned session files and replace their rows in the event aggregate:

```bash
python -m f1_prediction.cli build-session-features --season 2024 --event Monza --force
```

Cleaned laps preserve the useful raw FastF1 columns and add normalized identifiers, lap and sector times in seconds, pit-lap flags, `is_valid_lap`, and `is_push_lap`. They are written to:

```text
data/interim/clean_laps/{season}/{event_slug}/{session_slug}_clean_laps.parquet
```

A lap is valid when its lap and sector times are present, it is accurate and not deleted when those raw flags are available, it is not an in-lap or out-lap, and its driver and lap number are present. Missing optional accuracy/deletion/pit fields use permissive fallbacks documented in the cleaning code.

Push-like laps must be valid, use an allowed compound, be within 103% of the driver's best valid lap, and be within 107% of the session-best valid lap. These thresholds and compounds are configured in `configs/features.yaml`.

Driver/session aggregates are written to:

```text
data/processed/session_features/{season}/{event_slug}/practice_session_features.parquet
```

The aggregate contains one row per practice session and driver, with lap counts, best and median pace, theoretical best sectors, session-relative gaps and ranks, and basic tyre summaries. This remains a pre-modeling data preparation stage: it does not create qualifying targets or train models.

### Build the modeling dataset

Milestone 4 combines practice-session features with targets derived from raw qualifying laps:

```bash
python -m f1_prediction.cli build-modeling-dataset --season 2024 --event Monza
```

Existing output is skipped by default. Use `--force` to rebuild it:

```bash
python -m f1_prediction.cli build-modeling-dataset --season 2024 --event Monza --force
```

Qualifying targets include final position ranked from each driver's best valid qualifying lap, best qualifying lap time, gap to pole, and approximate Q2/Q3 advancement flags. For this MVP, Q2 and Q3 are inferred from positions 1-15 and 1-10 respectively; penalties, unusual qualifying formats, and session-specific elimination details are not modeled yet.

The dataset contains one row per qualifying driver at each prediction checkpoint:

* `after_fp1` uses FP1 performance features only;
* `after_fp2` uses FP1 and FP2 features only;
* `after_fp3` uses FP1, FP2, and FP3 features only.

Practice feature names are prefixed by session, such as `fp1_n_push_laps` and `fp2_best_push_lap_time_sec`. Future-session values are null on earlier checkpoint rows. Qualifying columns are targets only and are excluded by the feature-column helper.

The output is written to:

```text
data/processed/modeling/{season}/{event_slug}/modeling_dataset.parquet
```

This milestone prepares training rows but does not implement baselines or machine learning models.

### Build multiple events or seasons

Milestone 5 discovers events through the FastF1 schedule and reuses the existing ingestion, practice-feature, and event-modeling builders:

```bash
python -m f1_prediction.cli build-season-dataset --season 2024
```

Repeat `--season` to combine multiple seasons:

```bash
python -m f1_prediction.cli build-season-dataset --season 2023 --season 2024
```

Use an event filter for a small or incremental build:

```bash
python -m f1_prediction.cli build-season-dataset --season 2024 --events Monza
```

Event filters accept circuit locations, countries, event names, official event names, and common
aliases. For example, `Bahrain`, `Bahrain Grand Prix`, and `Sakhir` select the same weekend;
`Abu Dhabi`, `Abu Dhabi Grand Prix`, and `Yas Island` are also equivalent. Multiple event values
may follow one `--events` option.

For the first practical historical evaluation, use the conventional 2024 convenience preset:

```bash
python -m f1_prediction.cli build-season-dataset --season 2024 --preset conventional_2024
```

The preset contains Bahrain, Australia, Japan, Imola, Monaco, Canada, Spain, Silverstone, Hungary,
the Netherlands, Monza, and Abu Dhabi. It is an MVP convenience list of conventional FP1/FP2/FP3
weekends, not a complete 2024 calendar. Explicit `--events` selection remains available separately.

Milestone 22 adds a documented `conventional_2025` preset for expanding temporal validation:

```bash
python -m f1_prediction.cli build-season-dataset --season 2025 --preset conventional_2025
```

The 2025 preset selects conventional weekends that satisfy the current FP1/FP2/FP3/Q contract:
Australia, Japan, Bahrain, Saudi Arabia, Imola, Monaco, Spain, Canada, Austria, Silverstone,
Hungary, Netherlands, Monza, Baku, Singapore, Mexico City, Las Vegas, and Abu Dhabi. Sprint or
non-standard weekends are excluded from the preset rather than remapped to incompatible sessions:
China, Miami, Belgium, Austin, Sao Paulo, and Qatar.

Use the multi-season conventional preset to build the current expanded validation scope:

```bash
python -m f1_prediction.cli build-season-dataset --seasons 2023 2024 2025 --preset conventional
```

`--force` rebuilds existing event outputs. By default failures are recorded and remaining events continue; `--fail-fast` stops after the first failed event. The combined outputs are:

```text
data/processed/modeling/combined/modeling_dataset.parquet
reports/metrics/dataset_build_report.json
```

The current checkpoint contract requires FP1, FP2, and FP3. Sprint-format weekends without those sessions are reported as failures rather than silently mapped to incompatible sessions.

### Evaluate non-ML baselines

Evaluate the combined dataset with:

```bash
python -m f1_prediction.cli evaluate-baselines
```

An alternative dataset can be supplied with `--dataset`. Three transparent references are evaluated at each checkpoint:

* best push lap from the latest available practice session, falling back to earlier sessions;
* best valid lap with the same fallback rule;
* theoretical best practice lap with the same fallback rule.

Drivers are ranked within each event and checkpoint, and predicted gaps are measured relative to the best selected practice metric. These baselines use only practice sessions available at the checkpoint. Outputs are written to:

```text
reports/metrics/baseline_metrics.json
reports/metrics/baseline_predictions.parquet
```

These are reproducible non-ML references for future model evaluation; no machine-learning estimator is trained in this milestone.

### Inspect dataset quality

Create a coverage and missingness report before training:

```bash
python -m f1_prediction.cli dataset-report
```

Use `--dataset` to inspect another modeling Parquet file. The report includes event, season,
driver, and checkpoint coverage; missing targets and features; missing checkpoint detection;
and rows where only practice or qualifying information can be detected. It is written to:

```text
reports/metrics/dataset_quality_report.json
```

### Create time-aware splits

Hold out complete events or seasons so the same race weekend cannot appear in training and test:

```bash
python -m f1_prediction.cli split-dataset --strategy event_holdout --test-events Monza
python -m f1_prediction.cli split-dataset --strategy season_holdout --test-seasons 2024
python -m f1_prediction.cli split-dataset --strategy walk_forward --min-train-events 5
```

Walk-forward folds train only on earlier events and test on the next event. Event chronology uses
FastF1 round order in newly combined datasets, with deterministic dataset appearance as a fallback
for older files. Split metadata is written to:

```text
reports/metrics/dataset_splits.json
```

### Train first tabular models

The first ML command trains checkpoint-specific mean and median references, Ridge regression, and
a conservative Random Forest regressor:

```bash
python -m f1_prediction.cli train-tabular-models --test-events Monza --min-events 5
```

An entire season can instead be held out with `--test-season 2024`. Training is skipped with a
clear report when fewer than five unique events are available by default. Only numeric practice
features are used: identifiers, qualifying targets, `quali_` columns, and future-session features
are excluded. Each checkpoint is fitted separately to make the FP1/FP2/FP3 boundary explicit.

Outputs are written to:

```text
reports/metrics/tabular_model_metrics.json
reports/metrics/tabular_model_predictions.parquet
models/ridge_gap_model.joblib
models/random_forest_gap_model.joblib
```

When baseline metrics are available, model comparisons are recomputed on the same held-out events
instead of comparing test metrics against full-dataset baseline scores. These are intentionally
simple first models; advanced boosting, telemetry, and dashboard work remain out of scope.

### Produce the first backtesting report

After building enough events, run the complete evaluation sequence:

```bash
python -m f1_prediction.cli build-season-dataset --season 2024 --preset conventional_2024
python -m f1_prediction.cli dataset-report
python -m f1_prediction.cli evaluate-baselines
python -m f1_prediction.cli train-tabular-models --test-events Monza --min-events 5
python -m f1_prediction.cli backtest-report
```

The final command combines dataset coverage, baseline metrics, and tabular holdout metrics into:

```text
reports/metrics/backtest_report.json
```

It identifies the best practice baseline and best simple tabular model at each checkpoint, then
reports model-minus-baseline deltas for qualifying-gap MAE and mean absolute position error.
Negative deltas indicate improvement. If training was skipped or its metrics are absent, the report
is still generated with the available dataset and baseline results. This is the first meaningful
model-versus-baseline evaluation stage; it is not yet a claim of production-level generalization.

### Run repeated tabular backtests

Repeated event holdout trains on every event except one and repeats that process for each test
weekend. Restrict the evaluated weekends with `--test-events` when useful:

```bash
python -m f1_prediction.cli backtest-tabular-models --strategy repeated_event_holdout --min-events 5
python -m f1_prediction.cli backtest-tabular-models --strategy repeated_event_holdout --test-events Monza Silverstone
```

Walk-forward evaluation is stricter and closer to deployment chronology. It trains only on events
before the test weekend, starting once the requested training history is available:

```bash
python -m f1_prediction.cli backtest-tabular-models --strategy walk_forward --min-events 8 --min-train-events 5
```

Every fold evaluates the three practice baselines on exactly the same test rows as Ridge and Random
Forest. Reports include global metrics, fold-level metrics, fold mean and standard deviation, and
model-minus-best-baseline deltas. Negative deltas mean the ML model performed better.

Repeated holdout outputs:

```text
reports/metrics/repeated_event_holdout_metrics.json
reports/metrics/repeated_event_holdout_predictions.parquet
reports/metrics/repeated_event_holdout_folds.json
```

Walk-forward outputs:

```text
reports/metrics/walk_forward_metrics.json
reports/metrics/walk_forward_predictions.parquet
reports/metrics/walk_forward_folds.json
```

The recommended evaluation sequence is:

```bash
python -m f1_prediction.cli build-season-dataset --season 2024 --preset conventional_2024
python -m f1_prediction.cli dataset-report
python -m f1_prediction.cli evaluate-baselines
python -m f1_prediction.cli backtest-tabular-models --strategy repeated_event_holdout --min-events 5
python -m f1_prediction.cli backtest-tabular-models --strategy walk_forward --min-events 8 --min-train-events 5
python -m f1_prediction.cli backtest-report
```

`backtest-report` prefers walk-forward metrics when available, followed by repeated holdout and then
the original single-event holdout. Fold failures are recorded and remaining folds continue unless
`--fail-fast` is supplied.

### Diagnose errors and add relative practice context

Modeling rows include leakage-safe relative features derived only from practice aggregates. For
each FP1/FP2/FP3 checkpoint these describe:

* gap and percentage gap to the session-best push, valid, and theoretical lap;
* session rank for the same three pace measures;
* gap to the best available teammate and rank within the team;
* team-best practice pace and the driver's gap to that team benchmark.

Teammate gaps are null when no second team driver is available. Lower lap-time metrics are always
treated as better. Qualifying targets and `quali_` columns are never read when these features are
constructed, and future practice sessions remain absent from earlier checkpoint rows.

Rebuild event and combined modeling outputs from existing raw FastF1 files with:

```bash
python -m f1_prediction.cli build-season-dataset --season 2024 --preset conventional_2024 --force
```

Season-level `--force` rebuilds cleaned practice, aggregate, and modeling outputs while reusing
successful raw session files. It does not force a FastF1 re-download.

Create event- and driver-level error diagnostics with:

```bash
python -m f1_prediction.cli diagnostics-report
```

The command prefers walk-forward predictions, then repeated holdout, then full-dataset baseline
predictions. It identifies the worst events, drivers, checkpoint/model combinations, and baseline
or model event errors above the configured MAE threshold in `configs/features.yaml`. Outputs are:

```text
reports/metrics/diagnostics_report.json
reports/metrics/event_error_summary.parquet
reports/metrics/driver_error_summary.parquet
```

Recommended post-feature evaluation sequence:

```bash
python -m f1_prediction.cli build-season-dataset --season 2024 --preset conventional_2024 --force
python -m f1_prediction.cli dataset-report
python -m f1_prediction.cli evaluate-baselines
python -m f1_prediction.cli backtest-tabular-models --strategy repeated_event_holdout --min-events 5
python -m f1_prediction.cli backtest-tabular-models --strategy walk_forward --min-events 8 --min-train-events 5
python -m f1_prediction.cli backtest-report
python -m f1_prediction.cli diagnostics-report
```

### Add historical form and robust practice signals

Milestone 10 adds qualifying history that is available strictly before each event. Driver and team
features include rolling three-event, rolling five-event, and expanding averages for qualifying gap,
position, and Q3 rate. Driver history also includes teammate-relative qualifying gap. The first event
for an entity has a zero prior-event count and null historical averages. Current and future qualifying
targets are never used for the row being predicted. During event-holdout evaluation, historical
features are recomputed so the held-out event cannot influence later training rows either.

Each checkpoint row also carries transparent practice-signal quality fields: available session and
lap counts, missing latest-session pace indicators, a zero-to-six signal quality score, and flags for
push, valid, or theoretical practice gaps more than the configured threshold from the session best.
Rows are retained; these fields expose weak or suspicious practice evidence to models and reports.
Thresholds live in `configs/features.yaml`.

Three robust baseline variants complement the original practice baselines:

* `robust_best_push_lap`;
* `robust_best_valid_lap`;
* `robust_theoretical_best_lap`.

They use the latest checkpoint-safe session unless its driver gap to the session best exceeds the
configured threshold. In that case they fall back to an earlier available session. A driver with no
non-extreme signal remains missing and is ranked after valid predictions. Original baselines remain
unchanged for direct comparison.

Rebuild and evaluate the expanded dataset with:

```bash
python -m f1_prediction.cli build-season-dataset --season 2024 --preset conventional_2024 --force
python -m f1_prediction.cli dataset-report
python -m f1_prediction.cli evaluate-baselines
python -m f1_prediction.cli backtest-tabular-models --strategy repeated_event_holdout --min-events 5
python -m f1_prediction.cli backtest-tabular-models --strategy walk_forward --min-events 8 --min-train-events 5
python -m f1_prediction.cli backtest-report
python -m f1_prediction.cli diagnostics-report
```

The dataset quality report includes historical and data-quality feature counts, low-quality practice
rows, and rows with an extreme latest-session signal. Baseline and fold reports list original and
robust baselines separately, making the effect of outlier fallback visible.

### Build multiple seasons and run feature ablation

Milestone 11 adds a practical conventional-event preset for 2023:

```bash
python -m f1_prediction.cli build-season-dataset --season 2023 --preset conventional_2023
```

Build the documented 2023 and 2024 presets together with:

```bash
python -m f1_prediction.cli build-season-dataset --seasons 2023 2024 --preset conventional
```

The 2023 preset deliberately omits known sprint weekends that do not provide the required
FP1/FP2/FP3/Q contract. It is a practical MVP subset, not a complete championship calendar. Event
failures remain isolated in `reports/metrics/dataset_build_report.json`, and successful events from
both seasons are combined into the existing modeling dataset path.

Feature ablation compares the current Ridge and Random Forest models on identical event-safe folds:

```bash
python -m f1_prediction.cli ablation-backtest --strategy walk_forward --min-events 10 --min-train-events 5
```

The default comparison includes base lap features, base plus relative features, base plus historical
features, base plus data-quality features, and all numeric features. Additional component and
combination groups can be selected with `--feature-groups`. Target, identifier, current qualifying,
and non-numeric columns are excluded by the same safe feature helper used by normal training.

Outputs are written to:

```text
reports/metrics/ablation_metrics.json
reports/metrics/ablation_predictions.parquet
reports/metrics/ablation_feature_groups.json
```

`backtest-report` includes the preferred ablation feature group and its delta against the best robust
or original baseline on the same folds. Ablation is used to decide whether feature engineering helps
before stronger model families are introduced.

Recommended multi-season evaluation sequence:

```bash
python -m f1_prediction.cli build-season-dataset --seasons 2023 2024 --preset conventional
python -m f1_prediction.cli dataset-report
python -m f1_prediction.cli evaluate-baselines
python -m f1_prediction.cli backtest-tabular-models --strategy walk_forward --min-events 10 --min-train-events 5
python -m f1_prediction.cli ablation-backtest --strategy walk_forward --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-report
python -m f1_prediction.cli diagnostics-report
```

### Normalize identities across seasons

Milestone 12 adds explicit identity columns while preserving the original FastF1 labels:

* `driver_code` retains the FastF1 three-letter code where available;
* `driver_name` preserves the available display name, falling back to the code;
* `driver_key` is a stable lowercase code-based identifier;
* `team_name` preserves the original team label;
* `team_key` maps known team aliases and renames to a stable identifier.

For example, `Oracle Red Bull Racing` and `Red Bull Racing` map to `red_bull`; `Alfa Romeo`,
`Kick Sauber`, and `Stake F1 Team Kick Sauber` map to `sauber`; and `AlphaTauri`, `RB`, and
`Visa Cash App RB` map to `rb`. Unknown teams use a deterministic slug fallback. Teammate and
team-relative practice features group by `team_key`, while rolling history groups by `driver_key`
and `team_key`. Raw `driver` and `team` columns remain available for display and auditing.

The combined-dataset builder also upgrades older cached event files in memory, so identity
normalization does not require raw sessions to be downloaded again. Build both practical presets
with:

```bash
python -m f1_prediction.cli build-season-dataset --seasons 2023 2024 --preset conventional
```

The build report includes rows by season and event, normalized team count, and per-event failures.
When schedule lookup is unavailable but explicit preset events are known, the builder falls back to
those names and records individual FastF1 failures instead of aborting before a report is written.

`dataset-report` adds multi-season identity checks, including events, drivers, and teams by season;
missing key counts; key distributions; drivers represented under multiple teams; team/event groups
with only one driver; and events with fewer than twenty qualifying drivers. These checks should be
reviewed before introducing stronger model families.

Recommended validated workflow:

```bash
python -m f1_prediction.cli build-season-dataset --seasons 2023 2024 --preset conventional
python -m f1_prediction.cli dataset-report
python -m f1_prediction.cli evaluate-baselines
python -m f1_prediction.cli backtest-tabular-models --strategy walk_forward --min-events 10 --min-train-events 5
python -m f1_prediction.cli ablation-backtest --strategy walk_forward --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-report
python -m f1_prediction.cli diagnostics-report
```

### Backtest gradient-boosted models

Milestone 13 adds `HistGradientBoostingRegressor`, a scikit-learn gradient-boosted tree model that
handles missing numeric values without an external native dependency. Its conservative settings are
stored in `configs/model.yaml`; this milestone does not perform hyperparameter tuning.

Run leakage-safe walk-forward evaluation with the ablation-derived checkpoint policy:

```bash
python -m f1_prediction.cli backtest-boosted-models --strategy walk_forward --feature-policy checkpoint_best --min-events 10 --min-train-events 5
```

The default `checkpoint_best` policy uses:

* `base_lap_features` after FP1;
* `base_plus_quality` after FP2;
* `base_plus_relative` after FP3.

Use `--feature-policy all_features` to select every safe numeric feature at each checkpoint, or
`--feature-group base_plus_relative` to apply one registered group to every checkpoint. The normal
checkpoint filter still excludes future practice sessions, identifiers, and current-event
qualifying targets. Historical qualifying form remains eligible only because it was computed from
strictly prior events.

Boosted evaluation uses the existing event folds and computes robust baseline predictions on the
exact same test rows. Outputs are written to:

```text
reports/metrics/boosted_metrics.json
reports/metrics/boosted_predictions.parquet
reports/metrics/boosted_folds.json
```

`backtest-report` compares boosted MAE against the best fold-consistent baseline, the previous
tabular models, and the best ablation result. It also reports the preferred model family at each
checkpoint.

Recommended boosted-model evaluation sequence:

```bash
python -m f1_prediction.cli build-season-dataset --seasons 2023 2024 --preset conventional
python -m f1_prediction.cli dataset-report
python -m f1_prediction.cli evaluate-baselines
python -m f1_prediction.cli ablation-backtest --strategy walk_forward --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-boosted-models --strategy walk_forward --feature-policy checkpoint_best --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-report
python -m f1_prediction.cli diagnostics-report
```

Boosted models are introduced only after multi-season identity validation and identical-fold feature
ablation. Telemetry and deeper model families remain outside this milestone.

### Evaluate a checkpoint champion policy

Milestone 14 combines the strongest available prediction methods instead of forcing one family to
serve every checkpoint. The configured static policy currently uses:

* `robust_best_push_lap` after FP1;
* `robust_theoretical_best_lap` after FP2;
* Random Forest with `base_plus_relative` after FP3.

Evaluate that fixed engineering policy with:

```bash
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode static --min-events 10 --min-train-events 5
```

Nested mode is the leakage-safe evaluation mode:

```bash
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode nested --min-events 10 --min-train-events 5
```

For each test event and checkpoint, nested mode scores candidate methods using only earlier
out-of-sample walk-forward folds. The current event never participates in its own method selection.
The first fold, or any fold without usable prior method history, falls back to the configured static
method. Selection defaults to qualifying-gap MAE.

Milestone 15 adds a stabilized nested mode to reduce noisy method switching:

```bash
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested --min-events 10 --min-train-events 5
```

Stabilized nested mode is still leakage-safe: it uses only prior folds/events. Before switching away
from the static checkpoint default, it requires enough prior history and a meaningful improvement.
The defaults are configured in `configs/model.yaml`:

* at least 5 prior folds;
* at least 100 prior predictions;
* candidate MAE must beat the static/default method by at least 0.05 seconds.

If those conditions are not met, the checkpoint falls back to the static default and records a
`fallback_reason` such as `insufficient_history` or `hysteresis_margin_not_met`.

Milestone 19 adds an opt-in guarded stabilized mode:

```bash
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested_guarded --uncertainty conformal --min-events 10 --min-train-events 5
```

`stabilized_nested_guarded` behaves like stabilized nested except at `after_fp3`. If the configured
static/default FP3 method is Random Forest with `base_plus_relative`, and stabilized nested would
switch FP3 to a practice baseline such as `best_valid_lap`, `best_push_lap`, or
`theoretical_best_lap`, the guarded mode keeps the static FP3 Random Forest method instead. The
guardrail is deployable and leakage-safe because it uses only checkpoint, method identity, static
policy identity, and prior-selection metadata; it does not use realized test errors. It does not
affect FP1 or FP2, and it does not block non-baseline ML methods that clear the normal stabilized
nested gates.

Guarded selection records include audit columns:

```text
guardrail_applied
guardrail_name
guardrail_reason
pre_guardrail_selected_family
pre_guardrail_selected_model_name
pre_guardrail_selected_feature_group
post_guardrail_selected_family
post_guardrail_selected_model_name
post_guardrail_selected_feature_group
```

Champion runs now write both latest-run outputs and mode-specific snapshots:

```text
reports/metrics/champion_metrics.json
reports/metrics/champion_predictions.parquet
reports/metrics/champion_selection.parquet
reports/metrics/champion_static_metrics.json
reports/metrics/champion_nested_metrics.json
reports/metrics/champion_stabilized_nested_metrics.json
reports/metrics/champion_stabilized_nested_guarded_metrics.json
```

Champion predictions include approximate 90% intervals. The default residual-standard-deviation
interval uses only residuals from earlier folds:

```text
predicted gap +/- 1.64 * prior residual standard deviation
```

Conformal intervals are also available:

```bash
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested --uncertainty conformal --min-events 10 --min-train-events 5
```

The conformal interval uses the empirical 90th percentile of absolute residuals from prior folds
only:

```text
predicted gap +/- prior absolute residual quantile
```

Intervals remain null and are labeled `insufficient_history` until at least 20 prior residuals are
available. Champion metrics include interval coverage, mean/median interval width, and interval
availability by checkpoint. These intervals are diagnostic approximations, not fully calibrated
probabilistic guarantees.

Milestone 20 adds an opt-in predicted-gap-regime conformal method:

```bash
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested_guarded --uncertainty conformal_predicted_gap_bucket --min-events 10 --min-train-events 5
```

`conformal_predicted_gap_bucket` assigns each prediction to a bucket using only the predicted
qualifying gap:

* `pole_contender`: predicted gap `<= 0.5`
* `close_midfield`: `0.5 < predicted gap <= 1.5`
* `midfield`: `1.5 < predicted gap <= 3.0`
* `backmarker_or_outlier`: predicted gap `> 3.0`

It then calibrates the conformal residual quantile from prior folds/events only. The fallback
hierarchy is:

```text
checkpoint_method_bucket
checkpoint_bucket
checkpoint_method
checkpoint
global
```

This method never uses the actual qualifying gap to choose a bucket or calibrate its own interval.
Actual-gap bucket calibration remains an oracle/evaluation-only simulation and must not be used for
live prediction. The trade-off observed in simulation is better FP3 coverage with materially wider
intervals, so the method is opt-in and identified by the `uncertainty_method` column in saved
predictions and metrics.
Mode-specific champion artifact filenames remain keyed by selection mode, such as
`champion_stabilized_nested_guarded_predictions.parquet`; the `uncertainty_method` field inside the
metrics and predictions identifies whether the latest run used `conformal` or
`conformal_predicted_gap_bucket`.

`backtest-report` includes champion metrics, compares available selection modes, and reports the
best champion mode by checkpoint and overall. `diagnostics-report` also accepts champion predictions
as a prediction source.

Recommended champion evaluation sequence:

```bash
python -m f1_prediction.cli build-season-dataset --seasons 2023 2024 --preset conventional
python -m f1_prediction.cli evaluate-baselines
python -m f1_prediction.cli ablation-backtest --strategy walk_forward --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-boosted-models --strategy walk_forward --feature-policy checkpoint_best --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode static --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode nested --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested --uncertainty conformal --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested_guarded --uncertainty conformal --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested_guarded --uncertainty conformal_predicted_gap_bucket --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-report
```

The champion policy is an orchestration layer over existing leakage-safe predictions. It does not
introduce telemetry, paid data, or a new model family.

### Temporal weighting

Milestone 21 adds opt-in season-aware temporal weighting for tabular, ablation, and boosted
walk-forward backtests. This is intended for future live/current-season qualifying prediction, where
recent same-season evidence should usually matter more than older car-season history. Formula 1 team
pace changes materially across seasons because cars, regulations, upgrades, and driver/team
combinations change; older seasons are useful as prior evidence, but they should not dominate
current-season predictions.

Temporal weighting is not a champion-policy default and does not change existing unweighted
backtest behavior unless `--temporal-weighting` is passed. The default policy remains:

```bash
--temporal-weighting uniform
```

Available policies:

* `uniform`: every training row receives weight `1.0`, matching previous behavior.
* `season_priority`: same-season prior rows use the highest weight, previous-season rows use a
  lower prior weight, and older seasons use a small prior weight.
* `exponential_recency`: training rows receive an event-distance decay relative to the test event.
* `current_season_only_with_prior`: prior seasons are used as warm-start evidence until enough
  same-season events exist; after that threshold, training is restricted to same-season prior
  events.

All policies remain leakage-safe: fold weights are computed from training rows relative to the test
event, and future events are not introduced. The models that receive `sample_weight` are Ridge,
Random Forest, and histogram gradient boosting. Baseline/constant predictors are explicitly
recorded as unweighted where applicable.

Example commands:

```bash
python -m f1_prediction.cli backtest-tabular-models --strategy walk_forward --temporal-weighting season_priority --min-events 10 --min-train-events 5
python -m f1_prediction.cli ablation-backtest --strategy walk_forward --temporal-weighting season_priority --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-boosted-models --strategy walk_forward --feature-policy checkpoint_best --temporal-weighting season_priority --min-events 10 --min-train-events 5
```

Canonical latest-run artifacts are still written as before, and policy-specific snapshots allow
weighted and unweighted outputs to coexist:

```text
reports/metrics/walk_forward_uniform_metrics.json
reports/metrics/walk_forward_season_priority_metrics.json
reports/metrics/walk_forward_exponential_recency_metrics.json
reports/metrics/walk_forward_current_season_only_with_prior_metrics.json
reports/metrics/ablation_season_priority_metrics.json
reports/metrics/boosted_season_priority_metrics.json
```

Metrics JSON files include `temporal_weighting_policy`, `temporal_weighting_config`,
`weighted_models_supported`, `weighted_models_unsupported_if_any`, and
`training_weight_summary_by_fold`. Fold summaries record training composition, weight ranges,
weight sums, same-season/prior-season weight shares, and effective sample size.

Generate the artifact-based comparison report without rerunning training:

```bash
python -m f1_prediction.cli temporal-weighting-report
```

The report writes:

```text
reports/metrics/temporal_weighting_summary.json
reports/metrics/temporal_weighting_checkpoint_comparison.csv
reports/metrics/temporal_weighting_fold_comparison.csv
reports/metrics/temporal_weighting_training_composition.csv
reports/figures/temporal_weighting_mae_by_checkpoint.png
reports/figures/temporal_weighting_delta_vs_uniform.png
reports/figures/temporal_weighting_training_composition.png
reports/figures/temporal_weighting_effective_sample_size.png
```

Recommended temporal-weighting comparison sequence:

```bash
python -m f1_prediction.cli backtest-tabular-models --strategy walk_forward --temporal-weighting uniform --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-tabular-models --strategy walk_forward --temporal-weighting season_priority --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-tabular-models --strategy walk_forward --temporal-weighting exponential_recency --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-tabular-models --strategy walk_forward --temporal-weighting current_season_only_with_prior --min-events 10 --min-train-events 5
python -m f1_prediction.cli temporal-weighting-report
```

Temporal weighting is evaluated as an opt-in training policy. It does not replace current champion
defaults, and any promotion into champion candidate selection should be validated in a later
milestone.

### Season-aware validation

Milestone 22 validates season-aware training on a broader 2023-2025 conventional-event scope before
any temporally weighted candidate is considered for champion policy. The validation keeps champion
defaults unchanged and compares saved artifacts only; it does not retrain models or choose a new
live champion.

Current-season evidence matters in Formula 1 because car competitiveness, upgrades, regulations,
and driver/team combinations can change materially across seasons. The project distinguishes:

* historical prior evidence: older seasons used as lower-priority warm-start data;
* same-season evidence: earlier events from the current test season;
* cold-start behavior: events with fewer than five same-season prior events;
* retrospective candidate comparisons: fixed, artifact-based comparisons on aligned test rows;
* deployable champion policy changes: future work only, after broader validation.

Complete the current candidate evaluation with matching uniform and
`current_season_only_with_prior` runs:

```bash
python -m f1_prediction.cli backtest-tabular-models --strategy walk_forward --temporal-weighting uniform --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-tabular-models --strategy walk_forward --temporal-weighting current_season_only_with_prior --min-events 10 --min-train-events 5
python -m f1_prediction.cli ablation-backtest --strategy walk_forward --temporal-weighting uniform --min-events 10 --min-train-events 5
python -m f1_prediction.cli ablation-backtest --strategy walk_forward --temporal-weighting current_season_only_with_prior --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-boosted-models --strategy walk_forward --feature-policy checkpoint_best --temporal-weighting uniform --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-boosted-models --strategy walk_forward --feature-policy checkpoint_best --temporal-weighting current_season_only_with_prior --min-events 10 --min-train-events 5
```

Then generate the artifact-based season-aware validation report:

```bash
python -m f1_prediction.cli season-aware-validation-report
```

The report writes:

```text
reports/metrics/season_aware_validation_summary.json
reports/metrics/season_aware_fp3_candidate_comparison.csv
reports/metrics/season_aware_event_level_comparison.csv
reports/metrics/season_aware_season_level_comparison.csv
reports/metrics/season_aware_cold_start_comparison.csv
reports/figures/season_aware_fp3_candidate_vs_static_mae.png
reports/figures/season_aware_fp3_delta_by_event.png
reports/figures/season_aware_policy_by_test_season.png
reports/figures/season_aware_cold_start_vs_established.png
reports/figures/season_aware_training_weight_composition.png
```

The fixed FP3 challenge compares the current static champion candidate
`ablation/random_forest/base_plus_relative` trained uniformly against the same configuration trained
with `current_season_only_with_prior`, on identical fold/event/checkpoint/driver rows. It also
reports season splits, cold-start/early/established regimes, training-weight composition, effective
sample size, and a deterministic paired bootstrap over event-level FP3 MAE deltas. A weighted FP3
candidate is not automatically promoted just because it improves aggregate historical metrics.

Recommended 2023-2025 validation workflow:

```bash
python -m f1_prediction.cli build-season-dataset --seasons 2023 2024 2025 --preset conventional
python -m f1_prediction.cli dataset-report
python -m f1_prediction.cli evaluate-baselines
python -m f1_prediction.cli backtest-tabular-models --strategy walk_forward --temporal-weighting uniform --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-tabular-models --strategy walk_forward --temporal-weighting current_season_only_with_prior --min-events 10 --min-train-events 5
python -m f1_prediction.cli ablation-backtest --strategy walk_forward --temporal-weighting uniform --min-events 10 --min-train-events 5
python -m f1_prediction.cli ablation-backtest --strategy walk_forward --temporal-weighting current_season_only_with_prior --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-boosted-models --strategy walk_forward --feature-policy checkpoint_best --temporal-weighting uniform --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-boosted-models --strategy walk_forward --feature-policy checkpoint_best --temporal-weighting current_season_only_with_prior --min-events 10 --min-train-events 5
python -m f1_prediction.cli temporal-weighting-report
python -m f1_prediction.cli season-aware-validation-report
python -m f1_prediction.cli backtest-report
python -m f1_prediction.cli portfolio-report
```

### Season-aware champion candidate mode

Milestone 23 promotes the validated season-aware FP3 Random Forest candidate into an opt-in
champion selection mode:

```bash
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode season_aware_nested_guarded --uncertainty conformal_predicted_gap_bucket --min-events 10 --min-train-events 5
```

This does not replace the static champion and does not change `static`, `nested`,
`stabilized_nested`, or `stabilized_nested_guarded`. The new mode starts from
`stabilized_nested_guarded`: FP1 and FP2 retain the configured static/default behavior, while FP3
first applies the no-baseline-switch guardrail and then allows one additional weighted candidate to
compete:

```text
family: ablation
model_name: random_forest
feature_group: base_plus_relative
temporal_weighting_policy: current_season_only_with_prior
```

The weighted candidate is loaded from the temporal ablation artifact
`reports/metrics/ablation_current_season_only_with_prior_predictions.parquet`. It is never silently
substituted with the uniform RF artifact. If the weighted artifact is missing, the mode records
`season_aware_selection_reason = weighted_candidate_missing` and retains the guarded/default FP3
method.

Cold-start protection is conservative. The weighted FP3 candidate can be selected only when all of
these leakage-safe conditions hold:

* the checkpoint is `after_fp3`;
* at least 5 earlier events exist in the current test season;
* at least 5 prior candidate folds and 100 prior candidate predictions are available;
* prior-fold weighted RF MAE beats the guarded/static default by at least 0.05 seconds.

Selection uses only prior folds/events and method identity. It does not inspect the current test
event outcome. Selection records include:

```text
current_season_prior_event_count
season_aware_candidate_available
season_aware_candidate_eligible
season_aware_candidate_prior_folds
season_aware_candidate_prior_predictions
season_aware_candidate_metric_value
season_aware_default_metric_value
season_aware_improvement_margin_sec
season_aware_selected
season_aware_selection_reason
temporal_weighting_policy
temporal_weighting_config_summary
```

Mode-specific outputs are:

```text
reports/metrics/champion_season_aware_nested_guarded_metrics.json
reports/metrics/champion_season_aware_nested_guarded_predictions.parquet
reports/metrics/champion_season_aware_nested_guarded_selection.parquet
```

`champion-diagnostics` also writes optional season-aware champion diagnostics when these artifacts
exist:

```text
reports/metrics/season_aware_champion_summary.json
reports/metrics/season_aware_champion_event_comparison.csv
reports/metrics/season_aware_champion_regime_comparison.csv
reports/figures/season_aware_champion_fp3_mae_by_event.png
reports/figures/season_aware_champion_fp3_delta_vs_static.png
reports/figures/season_aware_champion_selection_by_regime.png
reports/figures/season_aware_champion_current_season_history.png
```

Recommended workflow after Milestone 23:

```bash
python -m f1_prediction.cli ablation-backtest --strategy walk_forward --temporal-weighting current_season_only_with_prior --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode static --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested_guarded --uncertainty conformal_predicted_gap_bucket --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode season_aware_nested_guarded --uncertainty conformal_predicted_gap_bucket --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-diagnostics
python -m f1_prediction.cli backtest-report
python -m f1_prediction.cli portfolio-report
```

### Season-aware candidate audit

Milestones 24-25 audit why the season-aware weighted FP3 Random Forest candidate does or does not
pass the live `season_aware_nested_guarded` gates. The audit is artifact-based: it does not retrain
models, lower thresholds, or change champion defaults.

Run:

```bash
python -m f1_prediction.cli season-aware-candidate-audit
```

The audit reads saved weighted and uniform ablation artifacts, champion static/guarded artifacts
when present, and the `champion_season_aware_nested_guarded_selection.parquet` records. The live
policy and audit share one canonical comparator: for each FP3 target fold, it uses only strictly
prior folds, aligns candidate and default rows on fold, season, event, checkpoint, and driver, drops
unmatched or invalid rows, and computes both MAEs on the exact same prior row scope. The resulting
selection metadata records the aligned fold/event scope, dropped rows, candidate MAE, default MAE,
and improvement margin used for the decision.

The audit checks candidate availability, row alignment, prior-only history, current-season
cold-start state, canonical prior-evidence deltas, live gate pass/fail status, and whether the live
selection records agree with the audited comparator.

Generated tables:

```text
reports/metrics/season_aware_candidate_eligibility_by_fold.csv
reports/metrics/season_aware_candidate_history_by_fold.csv
reports/metrics/season_aware_candidate_gate_failures.csv
reports/metrics/season_aware_candidate_alignment.csv
reports/metrics/season_aware_candidate_comparator_consistency.csv
reports/metrics/season_aware_candidate_gate_sensitivity.csv
reports/metrics/season_aware_candidate_gate_sensitivity_summary.csv
reports/metrics/season_aware_candidate_audit_summary.json
```

Generated figures:

```text
reports/figures/season_aware_comparator_live_vs_audit_mae.png
reports/figures/season_aware_comparator_improvement_delta.png
reports/figures/season_aware_comparator_consistency_by_fold.png
reports/figures/season_aware_candidate_selection_after_fix.png
reports/figures/season_aware_candidate_fp3_delta_after_fix.png
```

Sensitivity analysis is retrospective only. It tests small alternative gate grids using prior-fold
metrics for eligibility, then evaluates what would have happened on the held-out fold. A
retrospective setting that improves aggregate MAE is not a deployed policy and is not a reason by
itself to lower gates.

Recommended workflow after Milestone 25:

```bash
python -m f1_prediction.cli ablation-backtest --strategy walk_forward --temporal-weighting current_season_only_with_prior --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode static --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested_guarded --uncertainty conformal_predicted_gap_bucket --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode season_aware_nested_guarded --uncertainty conformal_predicted_gap_bucket --min-events 10 --min-train-events 5
python -m f1_prediction.cli season-aware-candidate-audit
python -m f1_prediction.cli champion-diagnostics
python -m f1_prediction.cli backtest-report
python -m f1_prediction.cli portfolio-report
```

### Season-aware policy forensics

Milestone 26 reconstructs and audits the live `season_aware_nested_guarded` FP3 policy after the
canonical comparator fix. It is an artifact-only diagnostic command and does not retrain models,
change live thresholds, or promote the weighted candidate.

Run:

```bash
python -m f1_prediction.cli season-aware-policy-forensics
```

The report reads saved static, guarded, season-aware, weighted-candidate, default-candidate, and
selection artifacts. For every FP3 fold it reconstructs the expected live prediction source,
compares reconstructed predictions with saved `season_aware_nested_guarded` predictions, and checks
row counts, prediction equality, fold MAE, and aggregate MAE consistency.

Generated tables:

```text
reports/metrics/season_aware_policy_forensics_summary.json
reports/metrics/season_aware_policy_fold_reconstruction.csv
reports/metrics/season_aware_policy_event_counterfactual.csv
reports/metrics/season_aware_policy_selected_fold_analysis.csv
reports/metrics/season_aware_policy_switch_cases.csv
reports/metrics/season_aware_policy_guardrail_event_level.csv
reports/metrics/season_aware_policy_guardrail_simulation.csv
reports/metrics/season_aware_policy_guardrail_summary.json
```

Generated figures:

```text
reports/figures/season_aware_policy_selected_vs_static_event_delta.png
reports/figures/season_aware_policy_selected_switch_balance.png
reports/figures/season_aware_policy_fold_reconstruction_status.png
reports/figures/season_aware_policy_guardrail_fp3_mae.png
reports/figures/season_aware_policy_guardrail_selection_rate.png
reports/figures/season_aware_policy_prior_improvement_vs_realized_delta.png
```

Interpretation:

* reconstruction checks whether the saved live policy can be reproduced from artifact rows;
* selected-fold analysis compares weighted-candidate selections against static and guarded
  counterfactuals;
* harmful switches use the configured champion-diagnostics tolerance;
* guardrail simulations use prior evidence only, but remain retrospective and are not deployed;
* static/guarded remains the recommended current policy unless broader validation says otherwise.

Recommended workflow after Milestone 26:

```bash
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode static --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested_guarded --uncertainty conformal_predicted_gap_bucket --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode season_aware_nested_guarded --uncertainty conformal_predicted_gap_bucket --min-events 10 --min-train-events 5
python -m f1_prediction.cli season-aware-candidate-audit
python -m f1_prediction.cli season-aware-policy-forensics
python -m f1_prediction.cli champion-diagnostics
python -m f1_prediction.cli backtest-report
python -m f1_prediction.cli portfolio-report
```

### Champion source lineage

Milestone 27 audits the source-level identity of the saved static FP3 champion rows. It is an
artifact-only reproducibility command; it does not retrain models, change champion thresholds, or
promote the season-aware weighted candidate.

Run:

```bash
python -m f1_prediction.cli champion-source-lineage
```

The report compares saved `champion_static_predictions.parquet` FP3 rows against the intended
static FP3 source contract:

```text
family = ablation
model_name = random_forest
feature_group = base_plus_relative
temporal_weighting_policy = uniform
source artifact = reports/metrics/ablation_uniform_predictions.parquet
```

Generated tables:

```text
reports/metrics/champion_source_lineage_manifest.json
reports/metrics/champion_source_lineage_artifact_summary.csv
reports/metrics/champion_source_lineage_fold_comparison.csv
reports/metrics/champion_source_lineage_row_comparison.csv
```

Generated figures:

```text
reports/figures/champion_source_lineage_prediction_delta_distribution.png
reports/figures/champion_source_lineage_mae_by_artifact.png
reports/figures/champion_source_lineage_fold_match_rate.png
reports/figures/champion_source_lineage_event_mae_delta.png
reports/figures/champion_source_lineage_contract_status.png
```

Current lineage finding after the scoped rebuild:

* static FP3 and uniform ablation rows align exactly on 774 rows;
* static FP3 and uniform ablation predictions match with tolerance match rate `1.0`;
* static source verification is `true`;
* root cause classification is `none_verified`;
* season-aware forensic counterfactual labels are valid when the rebuild/source-lineage checks pass.

Clean rebuild workflow, when an intentional artifact refresh is needed:

```bash
python -m f1_prediction.cli rebuild-season-aware-artifacts --dry-run
python -m f1_prediction.cli rebuild-season-aware-artifacts --no-include-dataset-rebuild --force
```

The rebuild command refreshes only a scoped set of known generated artifacts. By default it does
not rebuild the FastF1 dataset, so it can run from an existing combined 2023-2025 modeling dataset
without network access. Use `--include-dataset-rebuild` only when intentionally rebuilding the
underlying FastF1-derived dataset.

The deterministic artifact contract keeps temporal-weighting outputs separate:

```text
reports/metrics/ablation_uniform_predictions.parquet
reports/metrics/ablation_current_season_only_with_prior_predictions.parquet
```

Static FP3 champion loading now requires the exact uniform source snapshot. It does not fall back
to the latest-run `ablation_predictions.parquet`, which may reflect a weighted run.

Rebuild validation outputs:

```text
reports/metrics/season_aware_rebuild_summary.json
reports/metrics/season_aware_rebuild_artifact_registry.csv
reports/metrics/season_aware_rebuild_validation.csv
```

Rebuild figures:

```text
reports/figures/season_aware_rebuild_artifact_contract_status.png
reports/figures/season_aware_rebuild_static_uniform_prediction_delta.png
reports/figures/season_aware_rebuild_artifact_identity_matrix.png
reports/figures/season_aware_rebuild_fp3_mae_by_source.png
reports/figures/season_aware_rebuild_validation_status.png
```

### Prospective policy evaluation

Milestone 29 adds a prospective season-held-out evaluation for frozen champion-policy profiles. This
is a governance/reporting command: it does not retrain models, change champion defaults, retune
thresholds on the held-out season, or promote the season-aware candidate.

Run the documented held-out splits with:

```bash
python -m f1_prediction.cli prospective-policy-evaluation \
  --train-seasons 2023 \
  --test-season 2024 \
  --policy-profiles static_baseline \
  --policy-profiles guarded_baseline \
  --policy-profiles season_aware_frozen \
  --uncertainty conformal_predicted_gap_bucket \
  --min-events 10 \
  --min-train-events 5

python -m f1_prediction.cli prospective-policy-evaluation \
  --train-seasons 2023 \
  --train-seasons 2024 \
  --test-season 2025 \
  --policy-profiles static_baseline \
  --policy-profiles guarded_baseline \
  --policy-profiles season_aware_frozen \
  --uncertainty conformal_predicted_gap_bucket \
  --min-events 10 \
  --min-train-events 5
```

Frozen profiles:

* `static_baseline`: verified static champion policy.
* `guarded_baseline`: `stabilized_nested_guarded`.
* `season_aware_frozen`: current `season_aware_nested_guarded` gates and weighted FP3
  Random Forest/base_plus_relative candidate.

The frozen-policy principle is strict: policy profiles, margins, cold-start rules, candidate
identities, and uncertainty mode are fixed before scoring the held-out season. Earlier events from
the held-out season may be used only when they are chronologically available to the current event;
later held-out-season events are flagged by the leakage audit.

Generated outputs:

```text
reports/metrics/prospective_policy_summary.json
reports/metrics/prospective_policy_checkpoint_comparison.csv
reports/metrics/prospective_policy_event_comparison.csv
reports/metrics/prospective_policy_selection_log.csv
reports/metrics/prospective_policy_cold_start_comparison.csv
reports/metrics/prospective_policy_leakage_audit.csv
reports/metrics/prospective_train_2023_test_2024_predictions.parquet
reports/metrics/prospective_train_2023_2024_test_2025_predictions.parquet
```

Generated figures:

```text
reports/figures/prospective_policy_fp3_mae_by_test_season.png
reports/figures/prospective_policy_fp3_delta_vs_static.png
reports/figures/prospective_policy_fp3_delta_vs_guarded.png
reports/figures/prospective_policy_selection_rate_by_regime.png
reports/figures/prospective_policy_cold_start_performance.png
reports/figures/prospective_policy_interval_coverage_width.png
```

Current prospective result:

* leakage audit valid for both available splits;
* `train_2023 -> test_2024`: season-aware FP3 MAE `0.951216`, static FP3 MAE `0.945950`,
  guarded FP3 MAE `0.952567`;
* `train_2023_2024 -> test_2025`: season-aware FP3 MAE `0.528454`, static FP3 MAE
  `0.788273`, guarded FP3 MAE `0.549534`;
* aggregate recommendation remains `retain_static_policy`, because the 2024 split does not show
  season-aware improvement versus static and the guarded comparison confidence interval includes
  zero.

### True prospective policy replay

Milestone 30 adds a retrain-based prospective replay for the same frozen profiles. Unlike the
artifact-driven prospective evaluator above, this command rebuilds the required FP3 Random
Forest/base_plus_relative candidates from raw modeling rows for each held-out event. For event `E`,
the training history includes only the configured completed train seasons plus earlier events from
the held-out season when they are chronologically available. It never uses the current event, later
held-out events, future seasons, current/future residuals, or held-out outcomes to change thresholds.

Run the documented replay splits with:

```bash
python -m f1_prediction.cli prospective-policy-replay \
  --train-seasons 2023 \
  --test-season 2024 \
  --policy-profiles static_baseline \
  --policy-profiles guarded_baseline \
  --policy-profiles season_aware_frozen \
  --uncertainty conformal_predicted_gap_bucket \
  --min-events 10 \
  --min-train-events 5

python -m f1_prediction.cli prospective-policy-replay \
  --train-seasons 2023 \
  --train-seasons 2024 \
  --test-season 2025 \
  --policy-profiles static_baseline \
  --policy-profiles guarded_baseline \
  --policy-profiles season_aware_frozen \
  --uncertainty conformal_predicted_gap_bucket \
  --min-events 10 \
  --min-train-events 5
```

Generated replay outputs are intentionally path-distinct from ordinary walk-forward artifacts and
from the Milestone 29 artifact-driven prospective outputs:

```text
reports/metrics/prospective_replay_summary.json
reports/metrics/prospective_replay_checkpoint_comparison.csv
reports/metrics/prospective_replay_event_comparison.csv
reports/metrics/prospective_replay_selection_log.csv
reports/metrics/prospective_replay_training_manifest.csv
reports/metrics/prospective_replay_leakage_audit.csv
reports/metrics/prospective_replay_cold_start_comparison.csv
reports/metrics/prospective_replay_vs_artifact_driven.csv
reports/metrics/prospective_replay_train_2023_test_2024_predictions.parquet
reports/metrics/prospective_replay_train_2023_2024_test_2025_predictions.parquet
reports/metrics/prospective_replay_shadow_candidates.parquet
reports/metrics/prospective_replay_shadow_candidate_summary.json
reports/metrics/prospective_replay_shadow_candidate_availability.csv
reports/metrics/prospective_replay_shadow_candidate_training_manifest.csv
reports/metrics/prospective_replay_shadow_candidate_leakage_audit.csv
reports/metrics/prospective_replay_shadow_vs_live_selection.csv
reports/metrics/prospective_replay_shadow_gate_feasibility.csv
reports/metrics/prospective_replay_shadow_event_comparison.csv
```

Generated figures:

```text
reports/figures/prospective_replay_fp3_mae_by_test_season.png
reports/figures/prospective_replay_fp3_delta_vs_static.png
reports/figures/prospective_replay_fp3_delta_vs_guarded.png
reports/figures/prospective_replay_selection_rate_by_regime.png
reports/figures/prospective_replay_training_history_growth.png
reports/figures/prospective_replay_interval_coverage_width.png
reports/figures/prospective_replay_vs_artifact_driven.png
reports/figures/prospective_replay_shadow_candidate_availability.png
reports/figures/prospective_replay_shadow_history_growth.png
reports/figures/prospective_replay_shadow_gate_feasibility_timeline.png
reports/figures/prospective_replay_shadow_vs_live_eligibility.png
reports/figures/prospective_replay_shadow_counterfactual_selection.png
```

Current retrain-based replay result:

* leakage audit valid for both available splits;
* `train_2023 -> test_2024`: static, guarded, and season-aware FP3 MAE all `0.945950`;
* `train_2023_2024 -> test_2025`: static, guarded, and season-aware FP3 MAE all `0.788273`;
* `season_aware_frozen` selected the weighted FP3 candidate zero times in both true replay splits;
* replay FP3 interval coverage was `0.857143` for 2024 and `0.873950` for 2025, with mean
  interval widths around `3.95` and `4.04` seconds;
* artifact-driven and retrain-based results differ most for 2025 guarded/season-aware profiles,
  because the true replay retrains from event-specific legal history rather than consuming saved
  future-informed artifacts;
* aggregate recommendation remains `retain_static_policy`.

Milestone 32 adds diagnostic-only FP3 shadow persistence to the true replay. For every successfully
trained FP3 replay event, the replay now saves both the uniform default Random
Forest/base_plus_relative prediction rows and the season-aware weighted candidate rows, regardless
of which live policy profile selected them. These rows are written to
`reports/metrics/prospective_replay_shadow_candidates.parquet`, marked `diagnostic_only = true`, and
kept out of `prospective_replay_*_predictions.parquet`, the selection log, primary replay MAE,
coverage, and policy-selection rates.

Shadow history is prior-only: an event can use shadow rows from completed replay events with earlier
event order only. The current event, later same-season events, and future seasons are excluded from
shadow-history gate evaluation. Shadow-history eligibility is reported as a frozen-gate diagnostic
and counterfactual selection, not as a deployed replay policy or a claim that live replay selected
the weighted candidate. No policy thresholds, defaults, candidate identities, temporal-weighting
rules, or model hyperparameters changed.

Interpretation guidance:

* `static_baseline`, `guarded_baseline`, and `season_aware_frozen` remain frozen profiles.
* The replay is stronger deployment simulation than artifact-driven validation because it retrains
  event-by-event from legal history.
* Differences versus Milestone 29 are expected when retraining, fold-specific history, or artifact
  provenance changes the prediction source.
* The replay is governance evidence only. It does not promote the season-aware candidate or change
  champion defaults.

### Prospective replay eligibility audit

Milestone 31 audits why the true replay selected the season-aware weighted FP3 candidate zero times.
The audit reads saved replay artifacts only; it does not retrain models, fetch FastF1 data, lower
gates, tune thresholds, or change policy behavior.

Run the audit with:

```bash
python -m f1_prediction.cli prospective-replay-eligibility-audit
```

Generated audit outputs:

```text
reports/metrics/prospective_replay_eligibility_audit_summary.json
reports/metrics/prospective_replay_eligibility_by_event.csv
reports/metrics/prospective_replay_candidate_evidence_ledger.csv
reports/metrics/prospective_replay_gate_feasibility.csv
reports/metrics/prospective_replay_gate_failure_summary.csv
reports/metrics/prospective_replay_candidate_availability_comparison.csv
reports/metrics/prospective_replay_vs_artifact_driven_eligibility.csv
reports/metrics/prospective_replay_live_selection_consistency.csv
```

Generated figures:

```text
reports/figures/prospective_replay_eligibility_gate_pass_rate_by_event.png
reports/figures/prospective_replay_eligibility_history_growth.png
reports/figures/prospective_replay_eligibility_blocking_reasons.png
reports/figures/prospective_replay_eligibility_candidate_availability.png
reports/figures/prospective_replay_eligibility_vs_artifact_driven.png
reports/figures/prospective_replay_eligibility_feasibility_timeline.png
```

Interpretation guidance:

* Candidate model availability means the replay trained a weighted FP3 candidate for the event.
* Persisted prediction availability means diagnostic candidate rows were saved for later prior-only
  evidence, even when the live policy did not select them.
* Policy eligibility means the frozen cold-start, history, alignment, and margin gates all pass.
* Artifact-driven prospective evaluation can contain saved walk-forward candidate history that the
  stricter true replay does not retain.
* With Milestone 32 shadow persistence enabled, the audit can reconstruct legal prior candidate and
  default evidence from true replay history. The resulting shadow-history eligibility is
  `counterfactual_frozen_gate_evaluation_from_legal_shadow_history`; original live replay selection
  remains separate and unchanged.
* Current shadow-history results show that the frozen history gates become feasible in both held-out
  splits once shadow rows are retained. This is diagnostic evidence only and does not promote the
  weighted candidate.

### Season-aware governance synthesis

Milestone 33 adds an artifact-driven governance report for the weighted FP3 season-aware candidate.
It compares retrospective aligned walk-forward validation, artifact-driven prospective evaluation,
true retrain-based prospective replay, prior-only shadow-history diagnostics, candidate eligibility
audit results, and policy-forensics/source-lineage checks side by side.

Run the report with:

```bash
python -m f1_prediction.cli season-aware-governance-report
```

Generated governance outputs:

```text
reports/metrics/season_aware_governance_summary.json
reports/metrics/season_aware_governance_matrix.csv
reports/metrics/season_aware_governance_evidence_inventory.csv
reports/metrics/season_aware_governance_split_summary.csv
reports/metrics/season_aware_governance_regime_summary.csv
reports/metrics/season_aware_governance_identity_validation.csv
reports/metrics/season_aware_governance_decision_trace.csv
reports/metrics/season_aware_governance_missing_evidence.csv
reports/figures/season_aware_governance_evidence_pathway.png
reports/figures/season_aware_governance_candidate_vs_default.png
reports/figures/season_aware_governance_live_vs_shadow.png
reports/figures/season_aware_governance_split_status.png
reports/figures/season_aware_governance_decision_trace.png
```

The governance report validates the canonical FP3 comparison identity:

* candidate: `ablation` / `random_forest` / `base_plus_relative` /
  `current_season_only_with_prior`;
* default: `ablation` / `random_forest` / `base_plus_relative` / `uniform`.

It treats live replay zero selections and shadow-history counterfactual eligibility as different
facts. Shadow persistence repaired diagnostic evidence availability; it did not change live replay
policy behavior. The current conservative governance state is
`candidate_requires_more_live_prospective_evidence`. No default policy, threshold, model identity,
temporal-weighting policy, or hyperparameter changed.

### Season-aware stability analysis

Milestone 34 adds an artifact-driven stability report for the same canonical FP3 comparison. It
does not retrain models, rerun replay, fetch FastF1 data, change champion policies, or treat
shadow-history counterfactual selections as live selections.

Run the report with:

```bash
python -m f1_prediction.cli season-aware-stability-report
```

Generated stability outputs:

```text
reports/metrics/season_aware_stability_summary.json
reports/metrics/season_aware_stability_by_season.csv
reports/metrics/season_aware_stability_by_event.csv
reports/metrics/season_aware_stability_by_regime.csv
reports/metrics/season_aware_stability_event_concentration.csv
reports/metrics/season_aware_stability_leave_one_event_out.csv
reports/metrics/season_aware_stability_error_distribution.csv
reports/metrics/season_aware_stability_tail_risk.csv
reports/metrics/season_aware_stability_replay_shadow_summary.csv
reports/metrics/season_aware_stability_missing_evidence.csv
reports/figures/season_aware_stability_delta_by_event_order.png
reports/figures/season_aware_stability_delta_by_season.png
reports/figures/season_aware_stability_delta_by_regime.png
reports/figures/season_aware_stability_event_concentration.png
reports/figures/season_aware_stability_error_distribution.png
reports/figures/season_aware_stability_live_vs_shadow_status.png
```

The stability report validates the canonical weighted candidate and uniform default identities
before aggregation, then separates retrospective error deltas from live replay and shadow-history
diagnostics. The current saved-artifact run classifies the weighted candidate evidence as
`tail_risk_concern`: it improves in 2024 and 2025 and is strongest in established-season events,
but 2023 is neutral, cold-start events do not improve, top beneficial events explain a large share
of the aggregate gain, and the candidate has a material worse-by-threshold row rate. The
recommendation remains `season_aware_candidate_requires_more_evidence`; no policy threshold,
candidate identity, model identity, temporal-weighting rule, or hyperparameter changed.

### Frozen prospective monitoring protocol

Milestone 35 adds a frozen out-of-season monitoring workflow for a future unseen season such as
2026. The workflow is artifact-driven and local-data only: it does not fetch FastF1 sessions,
retrain prior replay artifacts, change champion defaults, alter frozen gates, or promote the
season-aware weighted FP3 candidate.

Initialize and report a monitoring protocol with:

```bash
python -m f1_prediction.cli prospective-monitoring-init \
  --protocol-name season_2026_v1 \
  --monitor-season 2026 \
  --train-seasons 2023 2024 2025
python -m f1_prediction.cli prospective-monitoring-report
```

When local monitored-season FP3-safe feature rows and, later, qualifying targets exist, run one
event at a time:

```bash
python -m f1_prediction.cli prospective-monitoring-preflight \
  --protocol-name season_2026_v1 \
  --season 2026 \
  --event Monza
python -m f1_prediction.cli prospective-monitoring-forecast \
  --protocol-name season_2026_v1 \
  --event Monza
python -m f1_prediction.cli prospective-monitoring-settle \
  --protocol-name season_2026_v1 \
  --event Monza
python -m f1_prediction.cli prospective-monitoring-report
```

The protocol freezes the canonical FP3 weighted candidate, uniform default, observed live-policy
reference, gate configuration, temporal-weighting configuration, dataset contract, artifact
contracts, and protocol fingerprint. Re-running the same protocol validates the fingerprint;
changing frozen scope or identity requires a distinct protocol name.

Forecasts and settlements are separate phases. Forecasts are pre-qualification snapshots and must
not read the current event target. Settlements join later supplied outcomes to immutable forecast
rows and make only settled earlier-event evidence available to future monitored events. Static or
guarded policy rows remain the observed live-policy reference, while weighted candidate rows are
diagnostic-only shadow evidence. Shadow evidence may inform frozen-gate diagnostics but is never
reported as observed live selection.

Milestone 39 adds an explicit preflight gate before every future monitored forecast. The preflight
command is artifact-driven and does not create predictions. It validates the frozen protocol,
registry-only event-order lineage, FP3-safe feature artifact, required feature/identity columns,
absence of embedded or separate Q targets before forecasting, absence of an existing forecast or
settlement for the event, and legacy-prior-evidence quarantine status. A target artifact or target
coverage ledger existing before forecast blocks the event, even when otherwise valid.

Preflight writes:

```text
reports/metrics/prospective_monitoring_preflight_summary.json
reports/metrics/prospective_monitoring_preflight_checks.csv
reports/metrics/prospective_monitoring_preflight_failures.csv
reports/metrics/prospective_monitoring_preflight_runbook.md
```

The generated runbook shows the exact safe next forecast command only when the status is
`ready_to_forecast`; otherwise it lists the corrective actions for the failed checks. The forecast
command also runs the same preflight internally and refuses to generate new prediction rows unless
the preflight status is `ready_to_forecast`. New forecast artifacts persist `preflight_run_id`,
`preflight_status`, and `preflight_summary_path`. Existing legacy forecasts and settlements are not
rewritten or reclassified by this safety check.

Generated monitoring outputs include:

```text
reports/metrics/prospective_monitoring_protocol.json
reports/metrics/prospective_monitoring_readiness.json
reports/metrics/prospective_monitoring_event_registry.csv
reports/metrics/prospective_monitoring_preflight_summary.json
reports/metrics/prospective_monitoring_preflight_checks.csv
reports/metrics/prospective_monitoring_preflight_failures.csv
reports/metrics/prospective_monitoring_preflight_runbook.md
reports/metrics/prospective_monitoring_forecasts.parquet
reports/metrics/prospective_monitoring_shadow_candidates.parquet
reports/metrics/prospective_monitoring_settlements.parquet
reports/metrics/prospective_monitoring_summary.json
reports/metrics/prospective_monitoring_integrity_summary.json
reports/metrics/prospective_monitoring_event_order_reconciliation.csv
reports/metrics/prospective_monitoring_event_order_integrity_summary.json
reports/metrics/prospective_monitoring_event_order_integrity_by_event.csv
reports/metrics/prospective_monitoring_event_order_integrity_failures.csv
reports/figures/prospective_monitoring_event_status.png
reports/figures/prospective_monitoring_live_vs_shadow_mae.png
reports/figures/prospective_monitoring_gate_timeline.png
reports/figures/prospective_monitoring_evidence_growth.png
reports/figures/prospective_monitoring_integrity_status.png
```

Milestone 38 makes the frozen monitoring registry the only valid chronology source for monitored
season event order. New forecasts fail before prediction generation if the registry row is missing,
duplicated, malformed, or non-positive. Forecast rows persist registry-lineage metadata, and prior
monitoring evidence may use only valid settled rows from strictly lower registry order with matching
protocol, monitor season, checkpoint, and settlement lineage.

Existing forecasts and settlements remain immutable historical records. If an older artifact carries
a noncanonical event order, the reconciliation report keeps it available for descriptive monitoring
metrics but excludes it from future prior-only frozen-gate evidence. This does not alter live policy
behavior, defaults, gates, thresholds, model identities, temporal weighting, or hyperparameters.

The current saved protocol `season_2026_v1` is active with local Australia and Great Britain 2026
monitoring records. Both existing forecast/settlement snapshots are retained as immutable
descriptive records, but their artifact event order is legacy noncanonical relative to the frozen
registry, so they are excluded from future prior-only gate evidence. The policy recommendation
remains `season_aware_candidate_requires_more_evidence`.

### Monitored-season data onboarding

Milestone 36 adds local-only onboarding for monitored-season events. It prepares FP3-safe
pre-qualification features separately from post-qualification settlement targets so the frozen
monitoring workflow can forecast without exposing qualifying outcomes.

Prepare and register a locally cached event before qualifying is evaluated:

```bash
python -m f1_prediction.cli monitoring-prepare-event --season 2026 --event Monza
python -m f1_prediction.cli monitoring-register-event \
  --protocol-name season_2026_v1 \
  --season 2026 \
  --event Monza
python -m f1_prediction.cli monitoring-data-readiness-report
```

Only after qualifying targets are legitimately available locally, add targets and settle through
the separate monitoring command:

```bash
python -m f1_prediction.cli monitoring-add-targets --season 2026 --event Monza
python -m f1_prediction.cli prospective-monitoring-settle \
  --protocol-name season_2026_v1 \
  --event Monza
```

The prepared feature artifact is:

```text
data/processed/monitoring/{season}/{event_slug}/monitoring_fp3_features.parquet
```

It must not contain qualifying targets or any `quali_` columns. The settlement-only target artifact
is:

```text
data/processed/monitoring/{season}/{event_slug}/monitoring_qualifying_targets.parquet
```

Milestone 37 allows a target artifact to contain only the drivers with valid, evaluable qualifying
targets. Forecast rows may legitimately exist for FP3 drivers who later have no qualifying target
row. Those forecast rows are retained for auditability, excluded from post-event scoring, and
excluded from future prior-performance evidence. The coverage ledger is:

```text
data/processed/monitoring/{season}/{event_slug}/monitoring_target_coverage.csv
```

It records every feature driver, whether a qualifying target was present, whether the target was
evaluable, the missing-target reason, and whether the row is included in settlement metrics. Missing
targets are never imputed, fabricated, silently dropped, or treated as zero error. Partial coverage
is a valid monitoring state when at least one valid target row exists and all overlapping identifiers
align exactly.

Forecasts may consume only the validated FP3-safe feature artifact after preflight returns
`ready_to_forecast`. Settlements require the separate validated target artifact and never rewrite
forecasts, model identities, gates, thresholds, temporal weighting, or default policy behavior. The
current local 2026 readiness includes Australia and Great Britain records with partial target
coverage; both existing forecast snapshots are legacy noncanonical-order artifacts and remain
excluded from future prior-only gate evidence.

Milestone 41 adds an artifact-only integrity audit for monitored-event driver populations and
dashboard display safety:

```bash
.venv/bin/python -m f1_prediction.cli monitoring-data-integrity-audit
```

The audit defines three populations that must not be treated as interchangeable:

* `feature_participant`: any driver with valid FP3-safe monitored feature rows.
* `forecast_eligible_driver`: a forecasted driver with an evaluable qualifying target for the
  monitored event.
* `settlement_evaluable_driver`: a forecasted driver with a valid settlement join and valid actual
  qualifying target values.

The command reads existing artifacts only and writes:

```text
reports/metrics/monitoring_data_integrity_audit_summary.json
reports/metrics/monitoring_data_integrity_audit_checks.csv
reports/metrics/monitoring_data_integrity_audit_failures.csv
reports/metrics/monitoring_data_integrity_event_driver_population.csv
reports/metrics/monitoring_data_integrity_event_comparison.csv
reports/metrics/monitoring_data_integrity_runbook.md
```

Practice-only, reserve, rookie, or otherwise non-evaluable FP participants remain auditable, but
they are excluded from the public primary qualifying leaderboard and settlement comparison. The
dashboard export separates `qualifying_eligible_forecast_rows`, `forecast_only_rows`, and
`settlement_evaluable_rows`; missing targets remain unavailable rather than blank or zero-valued
actual results. Australia and Great Britain legacy descriptive records remain quarantined from
valid prospective evidence and are not selected as the default Current Event when no clean eligible
monitored event exists.

Milestone 42 adds a raw qualifying source-to-target parity audit:

```bash
.venv/bin/python -m f1_prediction.cli qualifying-target-parity-audit
```

The audit reconstructs qualifying positions, best Q laps, and gaps to pole from local
`data/raw/laps/{season}/{event_slug}/q_laps.parquet` files using the same documented valid-lap
semantics as the qualifying-target layer. It then compares those reconstructed raw-Q values against
stored monitored targets, settlement actual values, and dashboard-exported actual values when the
event is the current dashboard event.

The audit does not fetch external data, call FastF1, retrain models, modify forecasts, rewrite
targets, mutate settlements, or repair legacy records. Run it after monitored qualifying targets and
settlements are available locally. Generated outputs are:

```text
reports/metrics/qualifying_target_parity_audit_summary.json
reports/metrics/qualifying_target_parity_audit_checks.csv
reports/metrics/qualifying_target_parity_audit_failures.csv
reports/metrics/qualifying_target_parity_event_summary.csv
reports/metrics/qualifying_target_parity_driver_comparison.csv
reports/metrics/qualifying_target_parity_runbook.md
```

The numeric comparison tolerance is `1e-6` seconds. Legacy status is reported separately: a legacy
event may still pass raw-Q parity, and a legacy label must not hide raw/target/settlement/dashboard
mismatches.

Milestone 43 adds a raw session identity guard before monitored target onboarding and settlement:

```bash
.venv/bin/python -m f1_prediction.cli raw-session-identity-validate \
  --season 2026 \
  --event "Great Britain" \
  --session Q
```

The guard validates that the requested `season/event/event_slug/session`, raw Q Parquet path,
matching raw metadata JSON, and metadata-resolved event identity all agree before
`monitoring-add-targets` can write target coverage or target artifacts. `prospective-monitoring-settle`
then requires both a valid target artifact and a persisted `identity_verified` raw-Q identity marker
from target onboarding. Failed, missing, stale, wrong-season, wrong-session, wrong-event, or
legacy-known-mismatch identity states block target onboarding and settlement.

Generated raw identity artifacts are:

```text
reports/metrics/raw_session_identity_validation_summary.json
reports/metrics/raw_session_identity_validation_checks.csv
reports/metrics/raw_session_identity_validation_failures.csv
reports/metrics/raw_session_identity_quarantine.csv
reports/metrics/raw_session_identity_runbook.md
```

Australia and Great Britain remain immutable legacy descriptive snapshots. Australia is raw-identity
verified locally but still `legacy_noncanonical` and quarantined from prospective evidence. Great
Britain is quarantined for both legacy lineage and a raw-source event mismatch:
`data/raw/session_metadata/2026/great-britain/q_metadata.json` identifies the Q session as
`Austrian Grand Prix`. Never overwrite or reinterpret legacy Australia or Great Britain snapshots.

Milestone 44 adds a guarded end-to-end rehearsal command for one clean, non-legacy monitored event:

```bash
.venv/bin/python -m f1_prediction.cli prospective-monitoring-rehearsal \
  --protocol-name season_2026_v1 \
  --season 2026 \
  --event "<EVENT>" \
  --event-order <ORDER>
```

The rehearsal is local-artifact-only. It does not fetch FastF1 data, fabricate missing sessions,
overwrite existing forecasts or settlements, or bypass preflight, raw-Q identity, target, settlement,
audit, or dashboard guards. It runs the existing workflow in order:

1. validate FP1/FP2/FP3 raw artifacts and metadata identity;
2. prepare FP3-safe monitoring features;
3. register the event with frozen registry chronology;
4. run preflight and require `ready_to_forecast`;
5. create the immutable forecast;
6. validate raw Q identity;
7. add settlement-only targets;
8. settle the matching forecast and targets;
9. run parity and data-integrity audits;
10. export the dashboard and require the clean event as Current Event.

Synthetic rehearsal events such as `Synthetic Clean GP` are explicitly excluded from valid
prospective evidence and future prior-evidence lineage, even when the rehearsal completes. They may
be recorded in internal dashboard history for testing, but they are never selected as the dashboard
Current Event and `valid_prospective_evidence` remains `false`.

Generated rehearsal outputs are:

```text
reports/metrics/prospective_monitoring_rehearsal_summary.json
reports/metrics/prospective_monitoring_rehearsal_stages.csv
reports/metrics/prospective_monitoring_rehearsal_checks.csv
reports/metrics/prospective_monitoring_rehearsal_failures.csv
reports/metrics/prospective_monitoring_rehearsal_driver_population.csv
reports/metrics/prospective_monitoring_rehearsal_runbook.md
```

Next real GP workflow:

Before qualifying, after FP1/FP2/FP3 artifacts are available locally:

```bash
.venv/bin/python -m f1_prediction.cli monitoring-before-qualifying \
  --season 2026 \
  --event "<EVENT>"
```

After qualifying:

```bash
.venv/bin/python -m f1_prediction.cli monitoring-after-qualifying \
  --season 2026 \
  --event "<EVENT>"
```

These two commands orchestrate the existing guarded steps, stop on the first blocking failure, never
overwrite forecast or settlement snapshots, and refresh dashboard artifacts automatically after the
required prior stages pass. The before-qualifying command resolves the canonical event name, slug,
scheduled date, and chronological event order from the FastF1 season schedule before ingestion; the
registry row count, legacy records, synthetic rehearsals, and local artifact insertion order never
determine event order.

Prospective forecasts are additionally constrained by the event's qualifying entry list. Practice
participants are not treated as qualifying entrants: FP-only rookies, reserve drivers, test drivers,
or substituted drivers are excluded unless they appear in the resolved qualifying entry list. Driver
count is event-derived rather than hard-coded to 20, and unusual valid cases such as replacement
drivers or reduced entry lists are handled from the entry-list artifact. If the entry list cannot be
resolved or does not match the forecast driver set exactly, forecast creation blocks before any
immutable forecast is written and the dashboard does not expose the stale forecast as valid.

Entry-list resolution uses this precedence: validated processed entry-list artifact, explicit
official/local entry-list artifact, authoritative race-driver roster, explicit local Q metadata when
Q has genuinely been loaded, reconciled event roster using official roster data plus completed
session evidence, latest completed official pre-qualifying session only when it is demonstrably
complete and consistent, then FastF1 Q results when qualifying data are actually available. Before
qualifying, Q data are not expected to exist. Practice sessions are diagnostic evidence, not final
eligibility: an FP-only reserve is excluded when absent from the roster, but a regular roster driver
is not treated as replaced merely because they missed FP3. If roster changes are ambiguous, or an
official entrant lacks compatible latest-checkpoint features, forecast creation blocks instead of
silently reducing the driver set or falling back to all practice participants. The diagnostic audit
can be run directly:

```bash
.venv/bin/python -m f1_prediction.cli qualifying-entry-list-audit \
  --season 2026 \
  --event "<EVENT>"
```

If a previously generated forecast used an invalid driver universe, preserve it for auditability and
regenerate safely through the canonical guarded command after the corrected entry list is available:

```bash
.venv/bin/python -m f1_prediction.cli qualifying-entry-list-audit \
  --season 2026 \
  --event "Belgian Grand Prix"

.venv/bin/python -m f1_prediction.cli monitoring-before-qualifying \
  --season 2026 \
  --event "Belgian Grand Prix"
```

Settlement parity is evaluated over forecasted drivers that also received verified targets. If a
target-only qualifying entrant appears after a pre-Q resolution miss, the system preserves the
original immutable forecast and settlement, reports partial forecast coverage, uses only evaluable
forecasted drivers in metrics, and exposes the missing actual entrant in dashboard diagnostics
instead of creating a retrospective prediction.

When workflows reuse preserved artifacts, operator summaries are populated from those immutable
forecast, target, settlement, parity, and dashboard artifacts. A preserved historical forecast with
partial qualifying coverage is reported explicitly as an immutable snapshot condition, not as a
generic integrity warning; metrics continue to use only the forecasted drivers that also have
verified targets.

Generated onboarding outputs include:

```text
reports/metrics/monitoring_onboarding_integrity_summary.json
reports/metrics/monitoring_onboarding_integrity_by_event.csv
reports/metrics/monitoring_onboarding_integrity_failures.csv
reports/metrics/monitoring_onboarding_readiness.csv
reports/metrics/monitoring_data_readiness_summary.json
reports/metrics/monitoring_data_readiness_by_event.csv
reports/metrics/monitoring_data_readiness_missing_inputs.csv
reports/figures/monitoring_data_readiness_event_status.png
reports/figures/monitoring_data_readiness_session_coverage.png
reports/figures/monitoring_data_readiness_target_isolation.png
reports/figures/monitoring_data_readiness_forecast_settlement_flow.png
```

### Portfolio reporting

Milestone 16 packages the existing walk-forward, champion-policy, and diagnostics artifacts into
compact portfolio-ready outputs. It reads saved metrics and predictions whenever possible; it does
not rerun expensive backtests.

Run the report with:

```bash
python -m f1_prediction.cli portfolio-report
```

Generated outputs:

```text
reports/metrics/portfolio_summary.json
reports/metrics/champion_summary_table.csv
reports/metrics/champion_interval_summary_table.csv
reports/metrics/champion_selection_summary_table.csv
reports/metrics/worst_event_diagnostics_table.csv
reports/metrics/worst_driver_diagnostics_table.csv
reports/figures/champion_mae_by_checkpoint.png
reports/figures/champion_vs_baseline_delta_by_checkpoint.png
reports/figures/champion_interval_coverage_by_checkpoint.png
reports/figures/champion_interval_width_by_checkpoint.png
reports/figures/champion_selection_share_by_checkpoint.png
reports/figures/worst_events_mae.png
reports/model_card.md
```

Use `portfolio_summary.json` as the top-level status file. It records which expected artifacts were
available, which were missing, the best champion mode by checkpoint, key result snippets, concise
takeaways, known limitations, and the recommended next milestone. Missing optional artifacts do not
stop the command; partial tables and figures are generated when enough inputs exist.

The champion summary table compares static, nested, stabilized nested, and guarded stabilized
policies by checkpoint when the corresponding artifacts exist.
Negative champion-versus-baseline delta MAE means the champion beat the best available baseline on
that checkpoint. The interval table summarizes availability, empirical coverage, and interval width.
The selection table shows which methods each mode selected, how often it fell back to the static
policy, and, for guarded runs, whether the FP3 no-baseline-switch guardrail changed the selected
method. The worst-event and worst-driver tables are compact views of diagnostics outputs for finding
hard cases.

The main figures mirror those tables:

* `champion_mae_by_checkpoint.png` compares absolute champion MAE across modes.
* `champion_vs_baseline_delta_by_checkpoint.png` highlights where the champion beats or trails the
  best baseline.
* `champion_interval_coverage_by_checkpoint.png` and
  `champion_interval_width_by_checkpoint.png` show uncertainty quality and breadth.
* `champion_selection_share_by_checkpoint.png` shows stabilized nested method-selection shares.
* `worst_events_mae.png` surfaces the hardest event/checkpoint combinations.

Recommended workflow after Milestone 15:

```bash
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode static --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode nested --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested --uncertainty conformal --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested_guarded --uncertainty conformal --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested_guarded --uncertainty conformal_predicted_gap_bucket --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-report
python -m f1_prediction.cli diagnostics-report
python -m f1_prediction.cli portfolio-report
```

### Champion diagnostics

Milestone 17 adds targeted diagnostics for champion-policy switching and conformal interval misses.
The command reads saved champion artifacts and does not rerun backtests:

```bash
python -m f1_prediction.cli champion-diagnostics
```

Required inputs, when available:

```text
reports/metrics/champion_static_predictions.parquet
reports/metrics/champion_nested_predictions.parquet
reports/metrics/champion_stabilized_nested_predictions.parquet
reports/metrics/champion_stabilized_nested_guarded_predictions.parquet
reports/metrics/champion_static_selection.parquet
reports/metrics/champion_nested_selection.parquet
reports/metrics/champion_stabilized_nested_selection.parquet
reports/metrics/champion_stabilized_nested_guarded_selection.parquet
```

Generated diagnostic tables:

```text
reports/metrics/champion_diagnostics_summary.json
reports/metrics/champion_harmful_switches.csv
reports/metrics/champion_switch_summary_by_checkpoint.csv
reports/metrics/champion_switch_summary_by_event.csv
reports/metrics/champion_switch_summary_by_method.csv
reports/metrics/fp3_policy_failure_analysis.csv
reports/metrics/conformal_miss_summary_by_checkpoint.csv
reports/metrics/conformal_miss_summary_by_event.csv
reports/metrics/conformal_miss_summary_by_method.csv
reports/metrics/conformal_miss_summary_by_driver.csv
reports/metrics/conformal_miss_cases.csv
reports/metrics/conformal_coverage_by_error_regime.csv
```

Generated figures:

```text
reports/figures/harmful_switch_delta_by_checkpoint.png
reports/figures/fp3_static_vs_stabilized_mae_by_event.png
reports/figures/conformal_coverage_by_checkpoint.png
reports/figures/conformal_miss_count_by_event.png
reports/figures/conformal_coverage_by_actual_gap_bucket.png
```

A harmful switch means a nested, stabilized, or guarded stabilized prediction has a larger absolute
qualifying-gap error than the static champion by more than
`champion_diagnostics.harmful_switch_tolerance_sec`
(currently 0.05 seconds). A beneficial switch beats static by more than the same tolerance; smaller
deltas are treated as neutral. Use the row-level switch table for driver/event cases, then use the
checkpoint, event, and method summaries to identify where switching helps or hurts.

Conformal miss diagnostics use rows with interval bounds. `miss_side` is `below_interval` when the
actual gap is lower than the interval and `above_interval` when the actual gap is higher than the
interval. Coverage-by-regime buckets separate pole contenders, close midfield, midfield, and high-gap
outliers so FP3 undercoverage can be checked for concentration by prediction regime.

Recommended workflow after Milestone 15:

```bash
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode static --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode nested --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested --uncertainty conformal --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested_guarded --uncertainty conformal --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested_guarded --uncertainty conformal_predicted_gap_bucket --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-report
python -m f1_prediction.cli diagnostics-report
python -m f1_prediction.cli portfolio-report
python -m f1_prediction.cli champion-diagnostics
```

### Policy simulation

Milestone 18 adds artifact-based simulations for FP3 champion guardrails and regime-aware conformal
interval recalibration. Milestone 19 promotes the best deployable FP3 guardrail into the opt-in
`stabilized_nested_guarded` live champion mode; the simulation command remains analytical and also
summarizes guarded-mode artifacts when they are available.

Run the simulation with:

```bash
python -m f1_prediction.cli policy-simulation
```

Required inputs, when available:

```text
reports/metrics/champion_static_predictions.parquet
reports/metrics/champion_stabilized_nested_predictions.parquet
reports/metrics/champion_static_selection.parquet
reports/metrics/champion_stabilized_nested_selection.parquet
reports/metrics/champion_stabilized_nested_guarded_predictions.parquet
reports/metrics/champion_stabilized_nested_guarded_selection.parquet
reports/metrics/fp3_policy_failure_analysis.csv
reports/metrics/champion_harmful_switches.csv
```

Generated simulation tables:

```text
reports/metrics/policy_simulation_summary.json
reports/metrics/fp3_guardrail_simulation_table.csv
reports/metrics/fp3_guardrail_event_level_table.csv
reports/metrics/regime_conformal_simulation_table.csv
reports/metrics/regime_conformal_event_level_table.csv
```

Generated figures:

```text
reports/figures/fp3_guardrail_policy_mae.png
reports/figures/fp3_guardrail_delta_vs_static.png
reports/figures/fp3_guardrail_event_mae.png
reports/figures/regime_conformal_fp3_coverage_width.png
reports/figures/regime_conformal_coverage_by_bucket.png
```

The FP3 guardrail simulations compare current static, current stabilized nested, and candidate
rules that substitute static FP3 predictions only under specific conditions. Deployable simulations
use only saved predictions and selection metadata. The live `stabilized_nested_guarded` mode
implements the deployable `fp3_no_baseline_switch` rule; the
`fp3_harmful_event_guardrail_oracle` policy
uses realized event-level MAE and is labeled oracle/evaluation-only; it is an upper bound, not a
live policy candidate.

The regime-aware conformal simulation recomputes interval quantiles using only prior folds. It tests
global, checkpoint-level, checkpoint+method, predicted-gap bucket, and actual-gap bucket strategies.
Actual-gap bucket calibration is oracle/evaluation-only because actual qualifying gap is unavailable
at prediction time. Predicted-gap bucket calibration is the deployable regime approximation.

Recommended workflow after Milestone 15:

```bash
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode static --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode nested --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested --uncertainty conformal --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested_guarded --uncertainty conformal --min-events 10 --min-train-events 5
python -m f1_prediction.cli champion-backtest --strategy walk_forward --selection-mode stabilized_nested_guarded --uncertainty conformal_predicted_gap_bucket --min-events 10 --min-train-events 5
python -m f1_prediction.cli backtest-report
python -m f1_prediction.cli diagnostics-report
python -m f1_prediction.cli portfolio-report
python -m f1_prediction.cli champion-diagnostics
python -m f1_prediction.cli policy-simulation
```

⸻

Expected CLI Commands

The final project should expose commands similar to these:

python -m f1_prediction.cli load-session --season 2024 --event Monza --session FP2
python -m f1_prediction.cli build-dataset --seasons 2023 2024 2025
python -m f1_prediction.cli train --config configs/model.yaml
python -m f1_prediction.cli evaluate --season 2025
python -m f1_prediction.cli replay --season 2025 --event Monza --session FP2

The first milestone should implement only the basic session loading command.

⸻

Feature Engineering Plan

Practice Lap Features

For each driver and practice session:

* best lap time;
* best lap rank;
* best lap gap to session best;
* median valid lap time;
* number of valid laps;
* number of push-like laps;
* best sector 1;
* best sector 2;
* best sector 3;
* theoretical best lap;
* actual best vs theoretical best gap;
* compound used for best lap;
* tyre age of best lap;
* session improvement trend.

Relative Driver Features

* gap to teammate;
* gap to session best;
* rank within session;
* sector ranks;
* team average practice pace;
* driver delta from team average;
* teammate-relative sector deltas.

Historical Features

Computed only from past data available before the race weekend:

* rolling average qualifying gap;
* rolling average qualifying position;
* rolling Q3 rate;
* driver form over previous N races;
* team form over previous N races;
* same-circuit historical performance;
* teammate-adjusted historical pace.

Weather Features

* air temperature;
* track temperature;
* humidity;
* pressure;
* wind speed;
* rainfall;
* session averages;
* session variability;
* weather changes between FP sessions.

Telemetry-Derived Features

To be added after the first modeling dataset is working:

* maximum speed;
* mean speed;
* full-throttle ratio;
* braking ratio;
* DRS usage ratio;
* acceleration proxy;
* corner minimum speed proxy;
* gear usage statistics;
* RPM summary statistics;
* speed profile summaries by distance bins.

⸻

Push-Lap Detection

The project should identify practice laps that are most informative for qualifying.

Initial rule-based criteria:

* lap is not an in-lap;
* lap is not an out-lap;
* lap time is valid;
* sector times are valid;
* lap is close to the driver’s best session lap;
* lap is close to the session best lap;
* compound is suitable for performance comparison;
* lap does not contain obvious abnormal slowdowns.

Later versions may use machine learning or clustering to classify lap types:

push lap
cooldown lap
race simulation lap
traffic-affected lap
out lap
in lap

⸻

Baseline Models

Before training advanced models, the project must implement baselines.

Minimum baselines:

1. Best Practice Lap Baseline

Use the best valid practice lap to predict qualifying order.

2. Best FP2 Lap Baseline

Use the best valid FP2 lap only.

3. Theoretical Best Lap Baseline

Use the sum of best available practice sectors.

4. Previous Qualifying Baseline

Use previous event qualifying order or gap.

5. Rolling Form Baseline

Use rolling average qualifying gap from previous events.

ML models must be compared against these baselines.

⸻

Machine Learning Models

Initial models:

* Ridge regression;
* Random Forest;
* Gradient Boosting;
* XGBoost regression;
* LightGBM regression;
* XGBoost/LightGBM classifier for Q3 probability.

Future models:

* learning-to-rank model;
* quantile regression;
* conformal prediction;
* calibrated classifiers;
* Bayesian-style prediction updates;
* ensemble model.

Deep learning should not be used until strong tabular baselines are implemented and evaluated.

⸻

Evaluation

The system should be evaluated through historical backtesting.

Recommended strategies:

Season Holdout

Example:

Train: 2023–2024
Test: 2025

Leave-One-Event-Out

Train on all race weekends except one and test on the held-out weekend.

Walk-Forward Validation

At each race weekend, train only on previous race weekends.

This is the most realistic evaluation strategy.

⸻

Metrics

Regression Metrics

* MAE of qualifying gap;
* RMSE of qualifying gap;
* median absolute error;
* error by team;
* error by circuit type.

Ranking Metrics

* Spearman rank correlation;
* Kendall tau;
* mean absolute position error;
* top-3 accuracy;
* top-5 accuracy;
* top-10 accuracy.

Classification Metrics

* Q3 accuracy;
* Q3 ROC-AUC;
* Q3 log loss;
* Brier score;
* calibration curve.

⸻

Data Leakage Policy

The project must avoid data leakage.

Rules:

* Do not use qualifying data as input features.
* Do not use race results to predict qualifying.
* Do not use future races to compute historical features.
* Do not use FP3 data for after_fp2 predictions.
* Do not compute season-level aggregates using races that happen after the predicted event.
* Fit scalers, imputers, encoders, and models only on the training split.
* Keep target-building logic separate from feature-building logic.

⸻

Historical Replay Mode

Because paid real-time API access is not assumed, real-time functionality should initially be simulated using historical replay.

Replay mode should:

1. load a historical practice session;
2. sort events/laps chronologically;
3. expose only data available up to a given timestamp/lap;
4. recompute or update features;
5. produce updated predictions;
6. show how prediction confidence changes over time.

Example future command:

python -m f1_prediction.cli replay --season 2025 --event Monza --session FP2

⸻

Dashboard Roadmap

A dashboard can be added after the ML pipeline is working.

Possible pages:

1. Qualifying forecast table.
2. Driver-level prediction explanation.
3. Teammate comparison.
4. Practice session evolution.
5. Sector and telemetry comparison.
6. Historical replay visualization.
7. Model evaluation report.

Recommended public dashboard architecture:

FastAPI plus React/Next.js, consuming stable read-only dashboard artifacts rather than raw modeling
internals.

Dashboard artifact export:

```bash
.venv/bin/python -m f1_prediction.cli dashboard-export
```

Read-only dashboard API:

```bash
.venv/bin/python -m f1_prediction.cli dashboard-api
```

The command writes normalized JSON files under:

```text
reports/dashboard/
```

This export layer is read-only. It adapts existing final monitoring and reporting artifacts for a
future UI and does not run FastF1 ingestion, model training, forecast generation, target ingestion,
or settlement.

The API serves only validated JSON files from `reports/dashboard/`. Run `dashboard-export` before
starting it. By default it binds to `http://127.0.0.1:8000`, exposes FastAPI OpenAPI docs locally
only while the server is running, and does not trigger ingestion, training, forecasts, preflight,
target ingestion, settlement, or other ML workflow operations.

Optional API configuration:

```text
APEX_PULSE_DASHBOARD_DIR=reports/dashboard
APEX_PULSE_DASHBOARD_API_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
APEX_PULSE_DASHBOARD_STALE_AFTER_MINUTES=180
```

### Production Runtime Foundation (Milestone 49A)

The Python backend now has a production container and a narrow optional persistent-runtime layout.
This is deployment preparation only: container startup does not run FastF1, export dashboard
artifacts, create or settle forecasts, train models, start a scheduler, or mutate the monitoring
registry. The only canonical artifact-producing workflows remain:

```bash
.venv/bin/python -m f1_prediction.cli monitoring-before-qualifying \
  --season 2026 \
  --event "<EVENT>"

.venv/bin/python -m f1_prediction.cli monitoring-after-qualifying \
  --season 2026 \
  --event "<EVENT>"
```

Local development is unchanged. With no `PORT` or `APEX_PULSE_RUNTIME_ROOT`, this command still
binds to `127.0.0.1:8000` and uses the repository's existing project-relative paths:

```bash
.venv/bin/python -m f1_prediction.cli dashboard-api
```

When `PORT` is present, the API binds to `0.0.0.0` and uses that port. Explicit `--host` and
`--port` options remain available. Wildcard CORS is rejected; production must list the exact public
frontend origin or origins.

Production environment contract:

| Variable | Required in production | Contract |
| --- | --- | --- |
| `PORT` | Yes on Railway | API port supplied by the platform; its presence selects `0.0.0.0`. |
| `APEX_PULSE_RUNTIME_ROOT` | Yes for persistence | Absolute persistent-volume mount path, recommended `/runtime`. |
| `APEX_PULSE_DASHBOARD_DIR` | Recommended | Dashboard JSON directory, recommended `/runtime/reports/dashboard`. |
| `APEX_PULSE_DASHBOARD_API_CORS_ORIGINS` | Yes for a public frontend | Comma-separated explicit origins; `*` is not allowed. |
| `APEX_PULSE_DASHBOARD_STALE_AFTER_MINUTES` | Optional | Positive freshness threshold; malformed values retain the 180-minute default. |

The container keeps application code and static configuration under `/app` and initializes these
idempotent links when `APEX_PULSE_RUNTIME_ROOT=/runtime`:

```text
/app/data    -> /runtime/data
/app/reports -> /runtime/reports
/app/models  -> /runtime/models
```

Only the three mutable trees are redirected. `configs/`, `src/`, `pyproject.toml`, and other
application files remain in the image, so a Railway volume must be mounted at `/runtime`, never over
`/app`. Startup creates missing directories and links, preserves existing persistent files, and
fails instead of replacing a conflicting link, file, or non-empty application tree. Milestone 49A
does not migrate this repository's current local artifacts.

The deployment is intentionally single-writer. The API is read-only today, but future explicitly
invoked automation will mutate monitored-event artifacts in the same runtime; do not horizontally
scale writers or attach concurrent writer processes without a later coordination design.

Read-only readiness diagnostic:

```bash
.venv/bin/python -m f1_prediction.cli production-runtime-check
```

It reports the resolved runtime, dashboard, FastF1 cache, protocol, and historical dataset paths.
After Milestone 49B it distinguishes `not_seeded`, `seeded_api_ready`,
`seeded_monitoring_ready`, and `corrupt_or_conflicting`; API readiness and future-monitoring
readiness are separate. It does not initialize a protocol, fetch data, export a dashboard,
forecast, settle, or create runtime directories.

Local container verification:

```bash
docker build -t apex-pulse-api .
docker run --rm \
  -e PORT=8000 \
  -e APEX_PULSE_RUNTIME_ROOT=/runtime \
  -e APEX_PULSE_DASHBOARD_DIR=/runtime/reports/dashboard \
  -p 8000:8000 \
  apex-pulse-api
```

The process healthcheck is `GET /api/v1/health`. It returns HTTP 200 when the API is alive even if
the runtime is fresh and dashboard artifacts are unavailable. Artifact routes keep their existing
validated 503/404/500 behavior. `railway.toml` records the Dockerfile strategy and this healthcheck;
it contains no account identifiers, domains, tokens, secrets, cron jobs, or start-command override.

### Production State Bundle and Railway Bootstrap (Milestone 49B)

Milestone 49B adds an explicit state-transfer boundary; it does not reconstruct any monitoring
output. The current dependency audit classifies generated files as `authoritative_required`,
`operational_required`, `dashboard_only`, `reconstructable_optional`, `development_only`, or
`unrelated_generated`. The exported closure contains:

- the frozen protocol, registry, forecast, settlement, shadow/evidence, selection, training,
  integrity, and chronology artifacts;
- every registered event's monitoring manifest, FP3 feature snapshot, existing qualifying target,
  and coverage artifact;
- the protocol's exact combined modeling dataset, which the future forecast workflow fits from;
- registered-event raw lap/session-metadata provenance and existing qualifying-entry-list evidence;
- all seven existing validated dashboard JSON files and their available source-fingerprint closure.

The prospective workflow fits from the dataset and does not load the two development
`models/*.joblib` files, so those binaries are excluded. Clean/session-feature intermediates,
figures, unrelated reports, and other development outputs are also excluded. Of the local FastF1
cache, only the 11 tiny `session_info.ff1pkl` files referenced by the copied dashboard are included
(6,374 bytes); the remaining 338,455,594 bytes are reconstructable and remain local.

Export without touching source artifacts:

```bash
.venv/bin/python -m f1_prediction.cli production-state-export
.venv/bin/python -m f1_prediction.cli production-dashboard-fingerprint
```

The ignored `dist/production-state/` output contains the compressed tarball, JSON manifest, CSV
full-workspace inventory, and JSON summary. Every bundled file has a size, SHA-256, portable runtime
destination, classification, immutable flag, and dashboard/forecast/settlement requirement flags.
The final local export contains 140 files / 3,063,533 uncompressed bytes and compresses to
1,103,741 bytes. The archive contains no secrets and no absolute Mac path contract.

Bootstrap a fresh runtime explicitly:

```bash
.venv/bin/python -m f1_prediction.cli production-state-import \
  --bundle dist/production-state/apex-pulse-production-state.tar.gz \
  --runtime-root /runtime
```

Import validates the complete archive in staging, rejects traversal, symlinks, undeclared members,
bad manifests, and checksum failures, then preflights every destination. Identical files are reused;
any differing destination aborts the entire import and is never overwritten. There is deliberately
no force option. The bootstrap receipt is written last at
`reports/metrics/production_bootstrap_receipt.json`. A successful full seed reports
`seeded_monitoring_ready`; a dashboard-only future bundle could report `seeded_api_ready` without
claiming monitoring readiness.

The local source and temporary imported runtime both report dashboard aggregate SHA-256
`8384bfb46eba1f47ecf5dce3ef6f17344c20bc99990b7d6bbb4cdcc08d4ca799`, current event
`Hungarian Grand Prix`, lifecycle `settled_partial_coverage`, 21 forecast rows, and 21 comparison
rows. This proves parity without running `dashboard-export`.

Railway now runs one replica/single writer with one volume mounted at `/runtime` and the unchanged
49A environment contract. Milestone 49B uploaded only the explicit bundle/manifest, imported it
through the guarded bootstrap command, verified the public HTTPS API and exact local/cloud dashboard
fingerprint, then proved the hashes survived a real redeploy. The backend is live at
`https://apex-pulse-production.up.railway.app`. Never use `railway up --no-gitignore` for state and
never reseed an existing differing runtime. Optional paid volume backups remain an operator choice.

`README.md` retains its pre-existing Git `skip-worktree` bit and `PROJECT_HANDOFF.md` remains
ignored, so these documentation updates are local and are not guaranteed to appear in GitHub-backed
deployments. Production behavior depends only on source/config files, not either document.

### Autonomous Weekend Orchestrator (Milestone 49C)

Milestone 49C adds a deterministic, one-shot weekend state machine. It does not add a scheduler,
background loop, cron job, new forecast path, or new settlement path. The only mutating transitions
call the same `run_monitoring_before_qualifying` and `run_monitoring_after_qualifying` Python
operations used by the canonical CLI commands.

Run a safe diagnostic tick:

```bash
.venv/bin/python -m f1_prediction.cli autopilot-tick --dry-run --json
```

For deterministic diagnosis, `--now` requires a timezone-aware ISO value; `--season` and `--event`
must still resolve to exactly one official schedule identity. A mutating tick is refused unless:

```text
APEX_PULSE_AUTOPILOT_ENABLED=true
```

The default is `false`, including in Railway during 49C. Dry run remains available while disabled,
does not acquire the writer lock, does not invoke either monitoring workflow, and does not write the
autopilot status/audit files. The production API startup still runs no automation as a side effect.

The explicit states are:

```text
NO_EVENT_AVAILABLE
WAITING_FOR_FP1 -> FP1_TIME_ELAPSED_DATA_PENDING / FP1_COMPLETE
WAITING_FOR_FP2 -> FP2_TIME_ELAPSED_DATA_PENDING / FP2_COMPLETE
WAITING_FOR_FP3 -> FP3_INITIAL_GRACE -> FP3_TIME_ELAPSED_DATA_PENDING
READY_FOR_FORECAST -> FORECAST_AVAILABLE
WAITING_FOR_QUALIFYING -> QUALIFYING_INITIAL_GRACE
QUALIFYING_TIME_ELAPSED_DATA_PENDING -> READY_FOR_SETTLEMENT
SETTLED / SETTLED_PARTIAL_COVERAGE
UNSUPPORTED_WEEKEND_FORMAT / AUTOPILOT_DISABLED / LOCK_CONTENDED / BLOCKED / TRANSIENT_ERROR
```

FastF1's official schedule selects the current/next event and supplies timezone-aware session
windows. Calendar time says only when data may be expected: after the configured grace, a separate
minimum-data probe loads laps without telemetry, weather, or race-control messages. Qualifying also
requires a usable results table. Unavailable, incomplete, and network failures are retryable;
unsupported formats, protocol/integrity failures, immutable conflicts, and missed retrospective
forecast windows block for operator review. The guarded workflows retain the final decision.

Only conventional FP1/FP2/FP3/Q weekends are supported. Sprint and other non-standard formats are
reported as `UNSUPPORTED_WEEKEND_FORMAT`; Sprint/Sprint Qualifying is never reinterpreted as FP2 or
FP3. Default initial grace is 5 minutes for early practice probes and 15 minutes for FP3 and Q,
followed by a 10-minute recommended retry interval. A tick never sleeps.

Mutating ticks use an advisory OS filesystem lock at
`reports/metrics/autopilot.lock`. Lock metadata records PID, hostname, UTC acquisition/release time,
and run ID, while the OS lock—not a stale file-existence test—decides ownership. Results are written
to append-only `autopilot_runs.jsonl` and the atomically replaced `autopilot_status.json`. These are
operational metadata only and never become forecast/settlement source of truth.

The read-only API exposes the latest validated snapshot at:

```text
GET /api/v1/autopilot-status
```

Before the first recorded mutating tick it returns HTTP 200 with `status=not_initialized`. There is
no trigger/POST route, and dashboard schema `1.0` remains unchanged. Status and runtime diagnostics
report FastF1 cache bytes, known runtime bytes, configured volume capacity, and a warning state.
`APEX_PULSE_RUNTIME_CACHE_WARNING_MB` defaults to 400; optional
`APEX_PULSE_RUNTIME_VOLUME_CAPACITY_MB=500` documents the current Railway volume. No cache file is
deleted automatically.

Artifact-copy rehearsal is also read-only:

```bash
.venv/bin/python -m f1_prediction.cli autopilot-rehearsal
```

It uses temporary copies of the authoritative tables to simulate pre-FP3, forecast-ready,
settlement-ready, and already-settled paths. The current rehearsal uses Belgian Grand Prix for the
forecast/settlement decision states and Hungarian Grand Prix for the terminal partial-coverage
state; it invokes no canonical workflow and verifies source hashes afterward.

Production contract additions:

| Variable | 49C production value | Purpose |
| --- | --- | --- |
| `APEX_PULSE_AUTOPILOT_ENABLED` | `false` | Explicit mutation safety gate; remains disabled in 49C. |
| `APEX_PULSE_RUNTIME_CACHE_WARNING_MB` | `400` | Read-only FastF1 cache warning threshold. |
| `APEX_PULSE_RUNTIME_VOLUME_CAPACITY_MB` | `500` | Optional diagnostic capacity for the current Railway volume. |

The existing Railway contract remains `PORT=<assigned>`, `APEX_PULSE_RUNTIME_ROOT=/runtime`,
`APEX_PULSE_DASHBOARD_DIR=/runtime/reports/dashboard`, restrictive explicit CORS origins, one
replica, one writer, and one volume mounted at `/runtime` rather than `/app`.

The 49C code was deployed to the existing service without reseeding `/runtime`. HTTPS health
returned 200 and `/api/v1/autopilot-status` returned `not_initialized`. The required production
dry run ran with `autopilot_enabled=false`, selected `Dutch Grand Prix` (round 12), detected
`sprint_qualifying`, and returned `UNSUPPORTED_WEEKEND_FORMAT` with `action_taken=none`; no session
probe or canonical workflow ran. The schedule cache grew naturally from 6,374 to 43,238 bytes, while
runtime status stayed `seeded_monitoring_ready` and every protocol, registry, forecast, settlement,
modeling-dataset, current-event, and aggregate-dashboard hash remained byte-identical. The
aggregate dashboard fingerprint is still
`8384bfb46eba1f47ecf5dce3ef6f17344c20bc99990b7d6bbb4cdcc08d4ca799`.
Local verification completed with 29 orchestrator-focused tests, 615 total Python tests, 74
frontend tests, Ruff check/format, `git diff --check`, frontend lint/typecheck, and the eight-route
Next.js production build all passing.

### Production Scheduler and Automatic Dashboard Refresh (Milestone 49D)

Milestone 49D runs the existing 49C one-shot orchestrator from an optional background scheduler in
the same Railway API process. It adds no second service, cron job, queue, database, alternate
forecast path, or alternate settlement path. FastAPI lifespan starts and stops the scheduler; each
blocking tick runs on a worker thread, local ticks cannot overlap, exceptions are recorded without
terminating the API or loop, and the existing `fcntl.flock` remains the final cross-process writer
guard.

Scheduling and mutation are separate controls:

| Variable | Production value | Meaning |
| --- | --- | --- |
| `APEX_PULSE_AUTOPILOT_SCHEDULER_ENABLED` | `true` | Automatically invoke one-shot ticks. Defaults to `false`. |
| `APEX_PULSE_AUTOPILOT_ENABLED` | `true` | Permit a tick to invoke canonical mutating workflows. Defaults to `false`. |
| `APEX_PULSE_AUTOPILOT_INTERVAL_SECONDS` | `300` | Cadence; values below the safe 60-second minimum are rejected. |
| `APEX_PULSE_AUTOPILOT_INITIAL_DELAY_SECONDS` | `10` | Restart delay before the first tick; zero is allowed. |

The scheduler writes `reports/metrics/autopilot_scheduler_status.json`; the orchestrator continues
to own `autopilot_status.json`, `autopilot_runs.jsonl`, and `autopilot.lock`. The read-only
`GET /api/v1/autopilot-status` response merges additive scheduler facts such as enabled/running,
start time, iteration count, last tick run/state, origin, next tick, and consecutive scheduler
failures. These files remain operational metadata, never forecast or settlement source of truth.
`/api/v1/health` remains API process health and stays healthy for unsupported weekends, pending
FastF1 data, retryable ticks, or intentionally disabled automation.

The 49B runtime verifier now distinguishes three layers: bootstrap receipt/manifest provenance,
byte-stable static seed invariants (including protocol, frozen modeling dataset, and seeded
per-event evidence), and validated current live operational state. Canonical registry,
forecast/settlement ledgers, and dashboard JSON may evolve through valid future appends without
being compared forever to the original whole-file seed hashes. Existing forecast snapshot hashes,
settlement integrity flags, chronology, and static artifacts are still validated; conflicts are not
weakened.

The public data routes refresh with a dependency-free client component every 60 seconds using
Next.js `router.refresh()`. Polling pauses while the document is hidden, refreshes on visibility or
focus return, cleans up timers/listeners, and never performs a full browser reload. Existing
server-side API requests retain `cache: "no-store"`, while theme preference and the independent
one-second countdown remain client state. The Vercel production project is built directly from the
verified local `web/` tree with the Next.js preset and:

```text
NEXT_PUBLIC_APEX_PULSE_API_BASE_URL=https://apex-pulse-production.up.railway.app
```

Production URLs:

- Frontend: `https://apex-pulse-ten.vercel.app`
- Backend: `https://apex-pulse-production.up.railway.app`
- Railway CORS: exactly `https://apex-pulse-ten.vercel.app` (wildcards and unrelated origins fail)

Railway rollout followed the required three stages: both flags disabled; scheduler enabled with
mutation disabled (automatic `AUTOPILOT_DISABLED`); then both enabled. The armed scheduled tick and
the first tick after a real redeploy both selected Dutch Grand Prix, round 12, detected
`sprint_qualifying`, returned `UNSUPPORTED_WEEKEND_FORMAT`, and took `action_taken=none`. The
scheduler resumed automatically after redeploy. Runtime remained `seeded_monitoring_ready`, cache
diagnostics remained healthy on the 500 MB volume, and the dashboard stayed Hungarian Grand Prix
with settled partial coverage (21/22).

All protected hashes stayed byte-identical through Stages A/B/C, Vercel deployment, and Railway
restart. The dashboard aggregate remains
`8384bfb46eba1f47ecf5dce3ef6f17344c20bc99990b7d6bbb4cdcc08d4ca799`; protocol, registry,
forecast, settlement, frozen-dataset, and `current_event.json` hashes remain the values documented
in the 49C handoff. No monitoring command was run manually to manufacture scheduler evidence.

Known limitations: sprint/non-standard forecast support remains absent; schedule/public-data
availability can still be delayed; FastF1 cache pruning is diagnostic-only and not destructive;
the free 500 MB volume may need a future retention policy; and GitHub `main` still does not contain
the dirty local 49A–49D implementation. Vercel was therefore deployed from `web/` with the CLI.
`README.md` remains skip-worktree and `PROJECT_HANDOFF.md` remains ignored, so both documentation
updates may be local-only.

### Public Next-Event UX and Freshness Semantics (Milestone 49D.1)

Milestone 49D.1 separates two deliberately different public concepts. The operational event is the
current/next weekend selected by the persisted autopilot scheduler; the latest prediction/result is
the immutable dashboard event. `GET /api/v1/autopilot-status` now adds an optional validated
`operational_event` object with season, round, canonical name/slug, FastF1 event format, support
status, UTC schedule source, and ordered session names/start/end/status values. The API reads the
persisted scheduler snapshot only: an API GET never calls FastF1 and never mutates forecast,
settlement, registry, or dashboard artifacts. Status payloads written before this additive field
remain valid.

The homepage fetches this status through the existing centralized API client with `cache:
"no-store"`. If present, the operational weekend appears first and drives the one-second local
countdown; the latest prediction/result appears separately below it. The existing 60-second
`router.refresh()` supplies updated server data without a second network poller. If operational
status is unavailable or not initialized, the operational card is omitted and the latest immutable
dashboard event remains usable.

Conventional sessions remain FP1/FP2/FP3/Q. Sprint formats retain the FastF1 names and ordering
(`Practice 1`, `Sprint Qualifying`/`Sprint Shootout`, `Sprint`, `Qualifying`) for public schedule
display, but they are never reinterpreted as missing practice sessions. The public copy explains
calmly that Sprint predictions are not supported yet; the internal
`UNSUPPORTED_WEEKEND_FORMAT` enum is not primary UI text. This is display-only and does not add
Sprint prediction support.

Freshness is lifecycle-aware. Active, old forecast artifacts can still show `Data may be stale`.
Terminal `settled` and `settled_partial_coverage` artifacts show neutral age copy such as `Settled ·
Updated 4 days ago`; age alone does not imply that an immutable completed result is operationally
stale. Dashboard schema `1.0`, history, Hungary's 21/22 partial coverage, and the absence of a
retrospective PER prediction are unchanged.

Production backend: `https://apex-pulse-production.up.railway.app`. Public frontend:
`https://apex-pulse-ten.vercel.app`. The Vercel project is Git-connected to
`matteosgobba/apex-pulse`, branch `main`, Root Directory `web`; because this milestone's working
tree changes are not yet committed/pushed, its public frontend update requires the explicit Git
handoff documented in `PROJECT_HANDOFF.md`. `README.md` remains skip-worktree and
`PROJECT_HANDOFF.md` remains ignored, so these documentation changes may remain local.

Local verification completed with 629 Python tests and 86 frontend tests passing, plus focused
orchestrator/scheduler/API and operational-UX/freshness/countdown tests, Ruff check/format,
`git diff --check`, frontend lint/typecheck, and the eight-route Next.js production build. Railway
deployment `696faf11-d690-4b80-bb69-1e15193aabe9` stayed healthy and
`seeded_monitoring_ready`; its first scheduled tick remained enabled/running at 300 seconds,
selected Dutch Grand Prix round 12, persisted FP1 at `2026-08-21T10:30:00+00:00` as the next
session, returned `UNSUPPORTED_WEEKEND_FORMAT`, and took no action. All seven protected hashes,
including dashboard aggregate
`8384bfb46eba1f47ecf5dce3ef6f17344c20bc99990b7d6bbb4cdcc08d4ca799`, remained unchanged.

Browser verification of the locally built 49D.1 frontend against the production API showed Dutch
first, Practice 1 counting down once per second, calm Sprint-unavailable copy, Hungary second with
neutral `Settled · Updated 4 days ago`, preserved 21/22 coverage, theme persistence, and no normal
console errors/warnings. The current Vercel URL still serves the previous 49D UI until the explicit
Git commit/push occurs; public 49D.1 acceptance is therefore pending rather than claimed.

The exact next milestone remains **Milestone 49E — First Live Conventional Weekend Validation and
Operations Hardening**. It will observe the first naturally supported FP1/FP2/FP3/Q weekend
end-to-end, validate real retry and guarded workflow transitions, prove old event rows remain
immutable after legitimate appends, and harden alerts/runbooks/cache retention from actual
production evidence. It must not add Sprint support merely to force a forecast.

Dashboard frontend:

```bash
cd web
npm install
cp .env.example .env.local
npm run dev
```

Optional frontend configuration:

```text
NEXT_PUBLIC_APEX_PULSE_API_BASE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_APEX_PULSE_DASHBOARD_STALE_AFTER_MINUTES=180
NEXT_PUBLIC_APEX_PULSE_LINKEDIN_URL=<optional verified profile URL>
```

The frontend is a public-facing, read-only Next.js application. `/` is the coherent current/latest
event experience: Apex Pulse branding, event context, an artifact-backed session countdown and
weekend timeline, the predicted qualifying ranking, official comparison, concise metrics,
methodology, contact, and secondary artifact details. `/history` presents expandable past
predictions; `/methodology` explains the public workflow and limitations. `/monitoring-history`
remains a backward-compatible history alias, while `/forecast`, `/practice`, and `/settlement`
redirect to the relevant homepage sections.

The public frontend defaults to a premium dark theme and also provides a persistent light theme.
The accessible sun/moon control is available in the shared desktop and mobile header. A valid
selection is stored under `apex-pulse-theme`; invalid or missing values use dark. The root layout
applies the theme before hydration, and shared semantic CSS variables keep backgrounds, surfaces,
text, status colors, focus states, rankings, and comparison rows coherent across both palettes.
Reduced-motion preferences disable the short color transitions.

The interface does not run FastF1, generate forecasts, settle forecasts, train models, or mutate
artifacts. Update frequency depends on regenerated dashboard artifacts and the separately running
read-only API, not direct live telemetry. Valid prospective records, legacy descriptive artifacts,
and historical backtests stay visually and analytically separate.

The additive optional `event_schedule` export comes only from existing local FastF1 session-info
cache files. Missing timestamps produce an explicit unavailable state. The browser countdown selects
the first verified future session in FP1/FP2/FP3/Q order, updates client-side in the visitor's local
timezone, clamps at zero, and shows a completed state after qualifying.

Team identity is centralized in `web/lib/team-identity.ts`. Mixed-format local team logos are served
from `web/public/teams/` through one reusable mark component, with a deterministic monogram fallback
for unknown teams. The header selects the supplied black/white monochrome Apex Pulse logo from
`web/public/brand/` using the existing root theme.

Partial qualifying coverage is never hidden: the comparison shows only valid comparable rows, the
evaluated denominator, and any unforecasted official entrants. It never creates a retrospective
prediction. Contact configuration is centralized in `web/lib/site-config.ts`; GitHub, email, and
Matteo Sgobba's LinkedIn profile are rendered together on public contact surfaces.

Freshness labels are derived from each dashboard artifact envelope's `generated_at_utc` timestamp.
The default stale threshold is 180 minutes. Stale data is still valid if the API has validated the
artifact; the frontend shows a restrained warning rather than treating old artifacts as failed.

Dashboard Deployment:

Path 1 — Portfolio demonstration:

* Host the Next.js frontend on Vercel or an equivalent frontend platform.
* Host the FastAPI dashboard API separately on Render, Railway, Fly.io, or an equivalent Python
  service.
* In Milestone 49A, configure the persistent runtime contract above; state migration and the first
  Railway bootstrap remain deferred to Milestone 49B.
* Point `APEX_PULSE_DASHBOARD_DIR` at the validated dashboard bundle on persistent storage.
* Point `NEXT_PUBLIC_APEX_PULSE_API_BASE_URL` at the deployed API URL on the frontend service.
* Persistent backend state may contain raw data, modeling artifacts, and models needed by future
  guarded operations, but those files must not be exposed through public dashboard routes.

Path 2 — Local demonstration:

Terminal 1:

```bash
.venv/bin/python -m f1_prediction.cli dashboard-export
.venv/bin/python -m f1_prediction.cli dashboard-api
```

Terminal 2:

```bash
cd web
cp .env.example .env.local
npm install
npm run dev
```

GitHub Pages alone is not sufficient for the current dashboard architecture because the app relies
on a FastAPI service to validate and serve the dashboard JSON. A future static-only publication step
could be designed separately, but it is not implemented here.

Safe artifact publication options include uploading a validated dashboard-artifact bundle during
deployment, using a controlled private artifact store later, or adding a CI artifact/package step in
a future milestone. Avoid committing raw or generated data indiscriminately.

⸻

First Milestone

The first milestone is to build the project foundation.

Tasks:

1. Create the repository structure.
2. Create Python package under src/f1_prediction.
3. Add pyproject.toml.
4. Add .gitignore.
5. Add .env.example.
6. Add config files in configs/.
7. Implement FastF1 cache initialization.
8. Implement a function to load one session.
9. Implement a CLI command to load one session.
10. Save basic lap data as Parquet.
11. Print a short summary of the loaded session.
12. Add minimal tests for config/path utilities.
13. Update this README with actual working commands.

Target command:

python -m f1_prediction.cli load-session --season 2024 --event Monza --session FP2

Expected output:

* FastF1 cache initialized;
* session loaded;
* lap data saved locally;
* basic summary printed.

⸻

Example Future Workflow

# Load a single session
python -m f1_prediction.cli load-session --season 2024 --event Monza --session FP2
# Build full dataset
python -m f1_prediction.cli build-dataset --seasons 2023 2024 2025
# Train model
python -m f1_prediction.cli train --config configs/model.yaml
# Evaluate model
python -m f1_prediction.cli evaluate --season 2025
# Launch dashboard
# Export dashboard-facing JSON artifacts for a future UI
.venv/bin/python -m f1_prediction.cli dashboard-export
# Serve those artifacts through the read-only local API
.venv/bin/python -m f1_prediction.cli dashboard-api

⸻

Project Philosophy

This project is not about making beautiful plots only.

The core value is:

1. convert noisy free practice data into meaningful features;
2. detect representative performance signals;
3. predict qualifying outcomes;
4. evaluate predictions rigorously;
5. explain model decisions;
6. simulate progressive real-time prediction updates.

The final result should demonstrate strong skills in:

* data engineering;
* machine learning;
* feature engineering;
* time-aware evaluation;
* sports analytics;
* model explainability;
* software engineering;
* applied product thinking.

### First Live Conventional Weekend Validation (Milestone 49E, Phase A)

Milestone 49E deliberately has two phases. Phase A hardens production and captures a pre-live
integrity baseline. Phase B can finish only when the scheduler naturally processes a real supported
FP1/FP2/FP3/Q weekend. The repository status after this work is
`PHASE_A_COMPLETE_AWAITING_LIVE_WEEKEND`; no live forecast or settlement was created for validation.

The append-only monitoring ledgers can grow legitimately, so whole-file hashes are no longer the
sole live proof. `live_event_integrity.py` records semantic fingerprints for every registered
event's registry identity, forecast and settlement blocks, shadow/audit/training evidence,
per-event feature/target/manifest files, qualifying entry-list evidence, and immutable historical
dashboard record. Protocol bytes, the frozen modeling dataset, and bootstrap provenance remain
static invariants. The comparator outcomes are:

- `UNCHANGED` and `VALID_APPEND`: successful;
- `MISSING_PREEXISTING_EVENT`, `PREEXISTING_EVENT_MUTATED`, `STATIC_INVARIANT_CHANGED`,
  `INVALID_CHRONOLOGY`, and `OTHER_BLOCKING_INTEGRITY_FAILURE`: operator-blocking.

Read-only and checkpoint commands:

```bash
.venv/bin/python -m f1_prediction.cli live-integrity-baseline
.venv/bin/python -m f1_prediction.cli live-integrity-baseline \
  --output reports/metrics/live_validation/pre_weekend_baseline.json
.venv/bin/python -m f1_prediction.cli live-integrity-compare \
  --baseline reports/metrics/live_validation/pre_weekend_baseline.json
.venv/bin/python -m f1_prediction.cli live-validation-checkpoint \
  --stage post_forecast_validation
.venv/bin/python -m f1_prediction.cli live-validation-rehearsal
```

Checkpoint files contain fingerprints and operational metadata, not cache or Parquet copies, and
existing checkpoint paths are never overwritten. A post-forecast checkpoint embeds the current
event-scoped manifest and can itself be used as a later comparator baseline to prove that settlement
did not rewrite the forecast.

Every persisted tick now identifies `trigger_source=scheduler|manual|rehearsal`. The scheduler calls
the existing one-shot orchestrator directly and records a separate observer snapshot at
`reports/metrics/live_validation_status.json`. That observer does not decide workflow eligibility.
It reports session observations, authoritative forecast/settlement counts and coverage, causality
timestamps, baseline comparison, storage/cache high-watermarks, and operator attention. The API
exposes it read-only:

```text
GET /api/v1/live-validation-status
```

An absent snapshot returns HTTP 200 with `not_initialized`; waiting or an unsupported weekend never
makes `/api/v1/health` unhealthy. Operator attention categories are `NONE`,
`RETRYING_DATA_AVAILABILITY`, `UNSUPPORTED_FORMAT`, `BLOCKING_INTEGRITY_FAILURE`,
`ENTRY_LIST_BLOCKED`, `FORECAST_WORKFLOW_BLOCKED`, `SETTLEMENT_WORKFLOW_BLOCKED`,
`CACHE_CAPACITY_WARNING`, and `SCHEDULER_NOT_RUNNING`. Observability failure is logged after a tick
but cannot turn a successfully committed canonical workflow into a scheduler failure/retry.

The 14-step rehearsal covers the unsupported weekend, automatic advance, FP1/FP2 progression, FP3
schedule-vs-readiness separation, forecast commit/reuse, qualifying schedule-vs-readiness
separation, settlement commit/reuse, and next-event selection. It uses temporary state and fake
readiness providers only. Cache handling remains observation-only; no retention or deletion was
introduced. The break-glass and transient-network procedures are in
`docs/live_validation_runbook.md`.

Phase B must wait for the canonical schedule to select a supported event. It will capture the real
FP3/forecast/Q/settlement timeline, entry-list coverage, scheduler run IDs, frontend transitions,
valid append proof, restart persistence, and cache growth. The implementation does not hardcode
Italy or Monza and makes no model, protocol, dashboard-schema, or public frontend change.

Phase A is deployed on the existing Railway service as deployment
`bb59cdb4-505a-4d8a-93da-c9da37d43c14`. Health and runtime readiness pass; the scheduler remains
enabled/running with mutation permission unchanged. Its first persisted Phase A tick was
scheduler-originated and retained Dutch Grand Prix round 12 as an explicit unsupported Sprint
no-op. The production pre-live baseline contains seven anchored events; file SHA-256 is
`10131d7f1012dad3558c35df4a0697e4591aad7935c0bdb5d5c353b21eb40602`, its semantic baseline
fingerprint is `f25cb6127a35ab1865d6b4b3bf309a698f6fd9aa6a28c06a9922eaa62ec342aa`, and immediate comparison
returned `UNCHANGED`. The volume remains 38 MB / 500 MB. All supplied immutable hashes and the
dashboard aggregate remain byte-identical after deploy and baseline capture.
