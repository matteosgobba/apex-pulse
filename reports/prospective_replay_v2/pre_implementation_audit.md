# FP3 policy audit — recorded before implementation

Audit date: 2026-09-16. No production code, model configuration, historical prediction
artifact, or selection output has been changed. Existing README edits and untracked
interview documents belong to the user and are outside this task.

## A. Training policy

The production monitoring path explicitly emits the uniform RF as `observed_live_policy`;
the weighted RF is a diagnostic shadow. Static FP3 identity is
`ablation/random_forest/base_plus_relative/uniform`. The opt-in alternative is
`ablation/random_forest/base_plus_relative/current_season_only_with_prior`.
Both use median imputation fitted on training data, RandomForestRegressor with 200
trees, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=1 and remaining sklearn
defaults. No feature scaling is applied to RF.

Uniform uses every eligible prior FP3 target-bearing row with weight 1. The alternative
uses weights 1.0 for current season, 0.35 for previous season, 0.10 for older seasons
until five represented prior current-season events exist; then it **discards all older
seasons** and uses only current-season rows, weight 1.0. This is a discontinuous training
window change, not simply gentle recency weighting. half_life_events=10 belongs to the
unused exponential policy. Neither policy imposes an additional push-lap row filter on
the modeling dataset. Practice features already embody configured lap filters.

`base_plus_relative` selects numeric FP1/FP2/FP3 practice columns except historical and
quality-pattern groups, preserving dataset column order; all-missing training columns
are excluded per fit. Rolling historical features are recomputed inside the legal fold
but are not in this candidate feature group. Relative practice features are previously
computed within season/event/session and team, without qualifying targets. Exact fitted
column lists will be saved for every V2 fit.

## B. Evaluation

Canonical dataset: `data/processed/modeling/combined/modeling_dataset.parquet`, 2,634
checkpoint rows, 44 events (14 in 2023, 12 in 2024, 18 in 2025). FP3 has 878 rows;
target-bearing walk-forward evaluation after five initial events has 774 rows/39 folds.
Only 2024 and 2025 have sufficient history before their first represented event for a
full-season replay. These are incomplete calendar samples, especially 2024.

Recalculation directly from prediction Parquets already reproduces retrospective
uniform MAE 0.9199012573872805 and weighted MAE 0.7290896015412823. The original
replay static MAEs are 0.9459500048586943 (238 rows/12 events, 2024) and
0.7882732422310238 (357 rows/18 events, 2025).

The reported 0.951216 and 0.528454 are **selected policy** MAEs from artifact-driven
season slicing, not always-weighted candidate MAEs or models fitted once before the
test season. The underlying walk-forward histories include earlier current-season
events. The artifact-driven guarded universe is broader than the two-RF replay, whose
guarded profile is an alias for static FP3. These experiments must not be conflated.

## C. Candidate eligibility

Frozen gates: FP3 only; exact candidate identity/prediction available; five prior
current-season events; five prior candidate folds; 100 prior candidate predictions;
at least five folds and 100 **aligned** prior candidate/default rows with finite MAEs.
Alignment keys: fold_id, season, event_slug, checkpoint, driver. Canonical comparator
is the matching fixed uniform RF. Selection requires prior uniform MAE minus prior
weighted MAE >= 0.05 seconds. All are prior-only. Fold identifiers passed to the
canonical selector must be chronological integers (the original replay's hashed IDs
are identifiers only, not chronological counters).

## D. Production-policy selection

The opt-in champion mode overlays the season-aware gates on guarded selection;
the deployed monitoring path continues to record uniform as live even when shadow
gates pass. Baseline switching at FP3 is prohibited. The 0.05 rule is a fixed advantage
over uniform, **not stateful hysteresis** with a separate retention/exit threshold.
No uncertainty-coverage or calibration gate governs candidate promotion.

Original replay intervals use a hardcoded empirical 90th percentile of >=20 prior
absolute residuals. Despite the `conformal_predicted_gap_bucket` label, this helper does
not implement bucket fallback or finite-sample conformal correction. V2 will retain
that numerical rule and identify it honestly; it cannot be evidence of conditional
coverage. Configured champion uncertainty has additional bucket/fallback machinery.

## E. Diagnostic/counterfactual evaluation and root cause

`run_true_replay` retrains both candidates for every replay event, with strictly prior
events and no future seasons. `train_event_sources` excludes the current target from
historical rolling construction. `fit_source_candidate` generates weighted predictions
even when ineligible. `apply_profiles_for_event` appends **only selected profile rows**
to next-event history. `season_aware_decision` searches that history for weighted rows.
Starting from none, the candidate cannot pass its history gate; thus none are ever
selected. Three uniform profile copies can duplicate comparator rows, but cannot create
weighted evidence. Zero candidate folds/predictions/aligned rows are structural; zero
selections do not measure weighted performance. They do not prove it would pass every
other gate either.

Milestone 32 independently stores both source predictions and their residuals. The
audit restricts these to earlier chronological event_order within the split, so later
diagnostic gate computations are causal conditional on the frozen design. But the
implementation computes errors immediately from already-loaded target-bearing rows,
writes combined shadow files after replay, and replaces split rows on rerun. It has no
immutable pre-outcome snapshot or enforced settlement transaction. `fit_and_predict`
also drops current rows lacking target values. Consequently the strong claim that the
old shadow code enforces every requested lifecycle invariant is false. No current
target value enters the RF features or prior gate calculation, but target **availability**
affects its reported population. The canonical dataset itself was built on qualifying
rosters, limiting any historical reconstruction of the original pre-Q roster.

The shadow results are called counterfactual because those predictions were not the
recorded selected outputs; incorporating them cannot rewrite the original replay.
There is no fundamental causal obstacle to a separately versioned simulation consuming
settled prior shadows: forecasting does not intervene on the qualifying outcome. This
is a causal information-flow replay, not randomized causal-effect identification.

## Frozen V2 plan

Reuse the production targetless RF fitting helper and canonical champion comparator/
selector, unchanged. Isolate targetless forecasting from settlement; freeze both source
predictions and the selected forecast before revealing current outcomes. Save append-only
per-event JSON snapshots, hashes, manifests and separate settlement records in this
new directory. Predict every canonical FP3 row first, then score the common valid-target
population. Record the canonical-roster limitation. Preserve the original warmup scope:
freshly reconstruct earlier-season OOF shadows after five training events, starting
each independent split with an empty ledger, and enter each evaluation season with
**zero current/future-season evidence**. No old shadow artifact is an input. Also report
a separate zero-all-candidate-evidence-at-season-start sensitivity, without changing
training history. Reproduce the original replay independently into this new directory.

Only after the primary run completes, evaluate the predeclared Cartesian neighborhood:
current-season events {4,5,6}, folds {4,5,6}, predictions {80,100,120}, margin
{0.025,0.05,0.075}. Keep RF, features and the training-window threshold fixed. Report
all 81 settings, never choose a winner. An additional margin=0 diagnostic can measure
churn relative to the frozen 0.05 margin. Use paired whole-event bootstrap (10,000
replicates, fixed seed 2026022), event and row estimands, and leave-one-event-out
concentration checks. These are diagnostic, not post-selection significance claims.

## Contamination already established

Milestones 9–15 developed features/models/stabilization using historical data; Milestone
21 compared four weighting policies on 2023–2024; Milestone 22 extended comparison to
2025; Milestone 23 explicitly chose the best Milestone 22 candidate for the opt-in mode.
Git commit ec8fecf (2026-06-24) introduces weighting and season-aware gates together;
4b9e0ec (2026-06-23) introduced stabilization and the 0.05 margin. Both evaluation
seasons were already observed. No independent preregistration or search log proves
which individual constants were optimized. Do not assert that each was tuned; do assert
that neither season is untouched candidate validation. No untouched full season is
present in the canonical dataset. Uniform production remains unchanged during this task.
