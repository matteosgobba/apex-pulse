# FP3 conditional policy V1 — audit and implementation

Decision freeze: **2026-09-20 09:08:16 UTC**. Audit classification: **B**.
Implementation and local validation are complete. **No remote deployment was
performed by this task.** The inspected Railway deployment used Uniform; this
document does not claim that a new image is already serving live predictions.

## Audit decision and evidence

The proposed rule is supported in principle. Two implementation inconsistencies
required correction before it could safely represent the audited candidates:

1. `monitoring_dataset_for_forecast` combined the frozen 2023–2025 dataset with
   the current targetless feature artifact. Registry reconciliation identified
   settled 2026 events, but their features/outcomes were not materialized into
   training. Consequently a deployment-only switch could count evidence that
   the model never trained on. The previous test fixture concealed this because
   its base dataset already contained monitored-season rows.
2. Commit `f08c266` (2026-09-06) added participation/team-evidence columns and
   changed team joins for drivers absent from a practice session. The canonical
   historical `base_plus_relative` candidate used **126 columns**. Fitting on
   newly built current-season rows could silently expand this to **135**.

The new path materializes legal, settled prior events and freezes the historical
126-column contract. Its adapter retains the pre-`f08c266` direct-driver session
join. Other roster/feature consumers retain the newer implementation. No feature
was selected using new performance results. The adapter reproduced all 126
columns across all **44 canonical events**, and 60 refits reproduced every aligned
2024/2025 candidate prediction **exactly** (maximum difference 0.0).

This resolves B without inventing a different predictor. No historical comparison
flaw or current/future-target leakage was identified. The later feature change
was real, but the historical advantage itself was not invalidated: an audit
using the newer target features and the old training matrix changed aggregate
MAEs only slightly (2024 U/S 0.946106/0.716415; 2025 0.787997/0.527646). Those
numbers are diagnostic; V1 deliberately preserves the original input contract.

## End-to-end production audit

The actual path is API lifespan → `build_production_scheduler` →
`run_autopilot_tick` → weekend orchestrator → `run_monitoring_before_qualifying`
→ `create_prospective_monitoring_forecast` → dashboard export. Its CLI equivalent
is `monitoring-before-qualifying --protocol-name season_2026_v1 --season 2026
--event "<canonical EventName>"`.

Before this change, `monitoring_prediction_rows` always exposed `source['static']`
as `observed_live_policy`. Both candidates were already fitted and persisted in
shadow. The old season-aware gates only annotated diagnostic eligibility; they
did not change that public output. Replay V2 is a separate historical diagnostic
and is not called by production. The new policy does not consult its gates or
its artifacts. The older frozen protocol remains unchanged; per-forecast policy
metadata explicitly records the new deployment-rule overlay.

The initial read-only Railway inspection found deployment
`31d7841c-1718-4198-845a-efa635579ac7`, protocol `season_2026_v1`, frozen dataset
2023–2025, and matching local/cloud forecast-loader/fitter/output functions.
Its live outputs, including the registered Spanish GP, were Uniform. V2 was
absent from the deployed package. No cloud state was mutated during this task.

## Candidate and provenance audit

Both candidates use RandomForestRegressor, 200 trees, max depth 8, minimum leaf
size 2, random state 42, median imputation and `n_jobs=1`. The numeric feature
whitelist and predictive source/function hashes are in
`configs/fp3_production_policy_v1.json`. Training rows require an evaluable finite
FP3 qualifying gap. Target rows use pre-qualifying practice evidence and the
existing qualifying-entry-list contract. All supported public roster rows are
forecast; missing outcomes are excluded only at settlement.

Uniform uses all eligible legal history with unit weights. Season-aware uses
`current_season_only_with_prior`: before five current-season training events,
weights are current season **1.0**, previous season **0.35**, older seasons
**0.10**. At five it uses **only eligible current-season history**. This was a
training behavior, not the previous live deployment behavior. V1 also uses that
existing boundary for deployment. No threshold search was performed.

RF settings are traceable to `5197ef4` (2026-06-14); feature/model development
and checkpoint comparisons include `19ae09d`, `402fbfb`, and `7cbeecc` (June 15).
The temporal experiment's June 23 artifact already contains the weighting rule
and five-event training transition, preceding the later regime analysis. The
2025 fixed-candidate evaluation summary is dated June 24 07:44:36 UTC. Commit
`ec8fecf`, later that day, packages several milestones together; it is not a
complete immutable pre-evaluation preregistration. Temporal weighting, RF
configuration and ablation training were unchanged afterward through the
audited HEAD; the later roster contract change is explicitly handled above.

Thus **2024 contributed to predictive development/validation**. **2025 supplies
meaningful historical sequential out-of-sample evidence**: each target used only
earlier events, including earlier 2025 events. There is no affirmative repository
evidence that 2025 results drove predictive parameter/feature tuning before its
fixed-candidate result was reported. Missing preregistration is a provenance
limitation, not proof of contamination. The earlier blanket statement that both
seasons necessarily contaminated the predictive candidate is corrected here.
Later deployment governance, including this conditional decision, did use
historical results. Neither season is a genuinely future prospective validation
of today's policy decision.

## Reproduced fixed-candidate results

These are the always-candidate walk-forward predictions, **not** the older
held-out deployment selector's metric. Each comparison has exactly matching
season/event/checkpoint/driver keys and outcomes.

| Season | Events | Rows | Uniform MAE (s) | Season-aware MAE (s) |
|---|---:|---:|---:|---:|
| 2024 | 12 | 238 | 0.945950005 | 0.716471373 |
| 2025 | 18 | 357 | 0.788273242 | 0.527566499 |
| Pooled | 30 | 595 | 0.851343947 | 0.603128449 |

Under the already-defined boundary, pooled cold-start U/S MAEs are
0.485946/0.491982 (10 events, 197 rows), and established-season MAEs are
1.032207/0.658143 (20 events, 398 rows). Established event wins are 15/5 for
Season-aware/Uniform. This supports the user's engineering decision, not a claim
that five is optimal. Large weekend contributions and limited independent events
remain limitations; 2024 is particularly concentrated. No new optimization or
reassessment of the old selector was used to implement V1.

## Runtime semantics and integrity

`FP3_CONDITIONAL_POLICY_V1` maps prior eligible count 0–4 to
`uniform_default` and count ≥5 to `season_aware_weighted_candidate`.
Candidate versions are `FP3_UNIFORM_RF_RELATIVE_126_V1` and
`FP3_SEASON_AWARE_RF_RELATIVE_126_V1`. Versioning names the existing audited
predictive specifications; it does not assert that historical forecasts carried
these newly introduced labels.

The count is the number of distinct same-season events actually materialized
with eligible FP3 outcomes. Registry chronology must place them strictly before
the target. Existing reconciliation must accept their forecast/settlement lineage
(synthetic, legacy-invalid and unsettled records do not qualify). Settlement
timestamp must precede the forecast. Outcomes for a driver must agree across
roles. A valid separate feature artifact is required. For new V1 events, the
frozen target-feature snapshot is reused; earlier legacy events use verified
registered features reconstructed with the audited adapter. Missing or conflicting
shared history blocks forecasting instead of inflating the count.

The base dataset contributes only configured earlier seasons. Any monitored-season
rows in that base are discarded. The current event has no targets; the fitter
also clears current targets defensively and verifies all training keys precede
the target. Both candidates are fitted from scratch on identical target rows.
Current residuals are not a predictive input and are not a selection input.

The scheduler's pre-Q time guard is supplemented by the same FastF1 calendar in
the manual forecast path. Only conventional weekends with a known qualifying
deadline are supported. A new forecast must be created and persisted before that
deadline and after the recorded policy freeze. Published local Q data also blocks
creation. A missed window cannot be backfilled into the new prospective store.

Both predictions are persisted before settlement, with policy/candidate versions,
contract fingerprint, timestamps, training-event keys, actual training counts,
input fingerprints, runtime-library versions, intended/actual candidate, regime
and fallback reason. The sole canonical public role stays `observed_live_policy`;
the two comparison roles remain diagnostic. The frontend/API contract is unchanged.

Existing frozen predictions are reused, including after settlement and after a
rollback setting change. They are never reinterpreted under a newer policy.
Candidate/config conflicts, mutated snapshots or missing V1 provenance fail
loudly. Numeric hash normalization handles nullable columns when new rows are
appended alongside legacy records. A filesystem lock serializes forecast and
settlement writers. Keep the existing single-writer/single-replica deployment.

Per-event frozen files are under:

```text
<metrics>/fp3_conditional_v1/<protocol>/<event_slug>/snapshot.json
<metrics>/fp3_conditional_v1/<protocol>/<event_slug>/forecasts.parquet
<metrics>/fp3_conditional_v1/<protocol>/<event_slug>/target_features.parquet
```

The canonical ledgers remain `<metrics>/prospective_monitoring_forecasts.parquet`,
`prospective_monitoring_shadow_candidates.parquet`, and
`prospective_monitoring_settlements.parquet`. Local metrics are `reports/metrics`;
the inspected cloud uses `/runtime/reports/metrics`. Settlement preserves the
original forecasts, aligns both candidates with the same actual outcomes, retains
non-evaluable/unavailable rows explicitly, and is idempotent for V1 events.

## Fallback, rollback and future versions

If weighted fitting raises a data/model availability error, Uniform is used if
its valid fit exists. The original intended candidate, actual Uniform candidate
and exception reason are persisted on every role. An unavailable weighted
prediction is recorded as missing, not invented or copied from Uniform. Such
events appear explicitly in reporting. Invalid shared inputs or a failed Uniform
fit block the forecast; the fallback cannot bypass chronology/config validation.

Set `APEX_FP3_POLICY_MODE=uniform_only` for future forecasts to activate
`FP3_UNIFORM_ROLLBACK_V1`. Both shadows still run. Set it back to `conditional`
to restore V1. Frozen forecasts and all evidence survive either change. A remote
environment-variable change follows the existing single-writer deployment runbook;
do not delete/reseed ledgers. A previous-image rollback is also possible, but the
documented mode preserves the improved history and shadow recording.

Predictive feature/preprocessing/RF/weighting/eligibility/data-contract changes
require a new candidate version and audited fingerprint. Deployment mapping,
threshold or material fallback changes require a new policy version. Source and
function hashes make accidental predictive changes fail loudly. Runtime library
versions are recorded; dependency upgrades also require parity validation. No
adaptive selector, automatic retuning or threshold optimization is implemented.

## Read-only reporting and reproduction

```bash
# Current/latest registered event, or a specifically registered canonical name.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m f1_prediction.cli fp3-policy-report
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m f1_prediction.cli fp3-policy-report --event "Hungarian Grand Prix"

# Exact historical candidate/input parity; JSON goes to stdout only.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/verify_fp3_candidate_parity.py

# Synthetic operational lifecycle, safety checks, historical parity and existing suite.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider --disable-warnings
```

The report separates overall and cold/established metrics; provides aligned rows,
event wins, chronological event metrics, cumulative results, intended/actual
selection and fallback records; and reports production metrics on all available
rows separately from complete U/S pairs. Its default event is the latest registered
non-synthetic event, not a fabricated calendar forecast. Use `--event` to inspect
a registered upcoming event. It performs no training, settlement or artifact writes.

Only correctly versioned, post-freeze, pre-Q forecasts settled later enter these
metrics. Historical 2024/2025 rows, Replay V2 and old monitoring rows do not. The
current new prospective total is **zero**. The first qualifying event is the first
supported real weekend for which the deployed V1 creates a new immutable forecast
after the freeze and before Q. **No event is predeclared as observed evidence and
no cloud activation timestamp is claimed.** Earlier valid outcomes can be training
history without being relabeled new-policy prospective evaluation.

Validation evidence is in `reports/fp3_conditional_policy_v1/`: exact refit
comparisons and protected-artifact hashes. All **563 preexisting protected files
remained unchanged**. New config/report files are additional. Replay V2 source
and artifacts were preserved, and its reproducibility tests still pass.

Local validation: **685 tests passed**, including 22 new focused cases; lint and
diff checks passed. Tests cover boundary counts 0/4/5/10, current/future targets
and residual perturbations, actual separate-artifact training history, settlement
availability time, both shadows, public-role isolation, immutable reuse before
and after settlement, config/snapshot conflicts, fallback, rollback, diagnostic
selector isolation, exact canonical metrics and the legacy nullable schema.
Existing tests were not weakened.

Operational limitation: there is no cross-file transaction across the immutable
snapshot and legacy aggregate ledgers. A crash during the append sequence is
detected and blocks regeneration; investigate and recover the frozen snapshot
under the existing runbook rather than refitting or deleting evidence.

## Completion checklist (requested questions 1–30)

| # | Answer |
|---:|---|
| 1 | B: missing current-season training materialization and historical feature-contract drift were corrected. |
| 2 | Uniform/static was the public FP3 predictor; both alternatives already existed as shadows. |
| 3 | Replay V2 did not affect live production and still does not. |
| 4 | Yes: 2024 0.945950005/0.716471373; 2025 0.788273242/0.527566499, on aligned rows. |
| 5 | No historical leakage/comparison flaw found. Runtime inconsistencies were fixed, with exact candidate parity. |
| 6 | Yes: five was already in temporal training before the later regime analysis. |
| 7 | Implemented mapping: eligible current-season count <5 Uniform; ≥5 Season-aware. Cloud deployment was not executed. |
| 8 | `FP3_CONDITIONAL_POLICY_V1`. |
| 9 | Audited Uniform RF, historical `base_plus_relative` 126-column contract. |
| 10 | Same RF/features, `current_season_only_with_prior`, current-season-only at the boundary. |
| 11 | Distinct earlier same-season events with accepted settlement lineage/time and materialized eligible FP3 training rows. |
| 12 | Target event is excluded by registry order and an independent fitter assertion. |
| 13 | Future events/outcomes and current targets/residuals cannot enter the current fit/selection; perturbation tests pass. |
| 14 | Both candidates are fitted/persisted before Q when feasible; missing weighted output is explicitly recorded. |
| 15 | Public role is Uniform in cold-start, Season-aware when established, or recorded Uniform fallback/rollback. |
| 16 | Canonical forecast/shadow ledgers plus `fp3_conditional_v1/<protocol>/<event>/` immutable snapshots. |
| 17 | `prospective_monitoring_settlements.parquet`, separate from frozen forecasts. |
| 18 | 2026-09-20 09:08:16 UTC. The first successful forecast also records an immutable policy-freeze file. |
| 19 | First supported real event newly forecast after activation/freeze and before its Q deadline; none collected yet. |
| 20 | `python -m f1_prediction.cli fp3-policy-report [--event "<registered EventName>"]`. |
| 21 | Weighted fit failure uses valid Uniform; invalid shared inputs or failed Uniform block output. |
| 22 | `intended_candidate`, `selected_candidate`, `fallback_reason`; missing weighted residual is excluded, fallback event remains visible. |
| 23 | `APEX_FP3_POLICY_MODE=uniform_only`; preserves shadows/evidence and uses `FP3_UNIFORM_ROLLBACK_V1` for new forecasts. |
| 24 | Candidate/config/source conflicts fail; explicit new candidate version and parity review are required. |
| 25 | A changed threshold/mapping/material fallback requires a new explicit policy version. |
| 26 | 22 focused cases covering the boundary, legal history, lifecycle, public isolation, fallback, rollback, version conflicts and canonical parity. |
| 27 | 685 tests passed; no existing tests weakened. |
| 28 | No: all 563 existing protected files are byte-identical. New files are additional. |
| 29 | Replay V2 source/artifacts unchanged; deterministic replay tests pass. No full replay rewrote its historical output. |
| 30 | No automatic retuning, adaptive selection or threshold search. |
