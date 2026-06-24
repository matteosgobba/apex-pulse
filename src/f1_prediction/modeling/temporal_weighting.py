"""Leakage-safe temporal sample weighting for event-based backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import pandas as pd

from f1_prediction.config import TemporalWeightingConfig


class TemporalWeightingPolicy(str, Enum):
    """Supported temporal weighting policies."""

    uniform = "uniform"
    season_priority = "season_priority"
    exponential_recency = "exponential_recency"
    current_season_only_with_prior = "current_season_only_with_prior"


@dataclass(frozen=True)
class TemporalWeightingResult:
    """Training data, sample weights, and diagnostics for one fold."""

    train: pd.DataFrame
    sample_weights: pd.Series
    summary: dict[str, object]


def temporal_weighting_config_payload(config: TemporalWeightingConfig) -> dict[str, object]:
    """Return a JSON-serializable config snapshot."""
    return asdict(config)


def prepare_temporal_training_data(
    train: pd.DataFrame,
    *,
    test_event: str,
    event_order: list[str],
    config: TemporalWeightingConfig,
    policy: TemporalWeightingPolicy | str = TemporalWeightingPolicy.uniform,
) -> TemporalWeightingResult:
    """Filter and weight training rows relative to one test event."""
    policy = TemporalWeightingPolicy(policy)
    frame = train.copy()
    if frame.empty:
        weights = pd.Series(dtype=float, index=frame.index)
        return TemporalWeightingResult(
            frame,
            weights,
            build_training_weight_summary(
                frame,
                weights,
                test_event=test_event,
                policy=policy,
                filtered_future_rows=0,
            ),
        )

    if policy is TemporalWeightingPolicy.uniform:
        weights = pd.Series(1.0, index=frame.index, dtype=float)
        return TemporalWeightingResult(
            frame,
            weights,
            build_training_weight_summary(
                frame,
                weights,
                test_event=test_event,
                policy=policy,
                filtered_future_rows=0,
            ),
        )

    order_index = {event: index for index, event in enumerate(event_order)}
    test_index = order_index.get(test_event)
    if test_index is None:
        raise ValueError(f"Test event is missing from chronological event order: {test_event}")
    row_events = _event_key_series(frame)
    prior_mask = row_events.map(order_index).lt(test_index)
    filtered_future_rows = int((~prior_mask).sum())
    frame = frame[prior_mask].copy()
    row_events = row_events.loc[frame.index]

    if policy is TemporalWeightingPolicy.current_season_only_with_prior:
        same_season_events = _same_season_prior_events(frame, test_event)
        if len(same_season_events) >= config.min_current_season_events:
            same_season_mask = frame["season"].astype(int).eq(_event_season(test_event))
            frame = frame[same_season_mask].copy()
            row_events = row_events.loc[frame.index]

    weights = _weights_for_policy(
        frame,
        row_events=row_events,
        test_event=test_event,
        event_order=event_order,
        config=config,
        policy=policy,
    )
    return TemporalWeightingResult(
        frame,
        weights,
        build_training_weight_summary(
            frame,
            weights,
            test_event=test_event,
            policy=policy,
            filtered_future_rows=filtered_future_rows,
        ),
    )


def build_training_weight_summary(
    train: pd.DataFrame,
    weights: pd.Series,
    *,
    test_event: str,
    policy: TemporalWeightingPolicy | str,
    filtered_future_rows: int = 0,
) -> dict[str, object]:
    """Summarize training composition and sample-weight concentration."""
    policy = TemporalWeightingPolicy(policy)
    test_season = _event_season(test_event)
    aligned = weights.reindex(train.index).astype(float)
    same_season = (
        train["season"].astype(int).eq(test_season) if "season" in train else aligned.eq(0)
    )
    previous_season = (
        train["season"].astype(int).eq(test_season - 1) if "season" in train else aligned.eq(0)
    )
    older_season = (
        train["season"].astype(int).lt(test_season - 1) if "season" in train else aligned.eq(0)
    )
    total_weight = float(aligned.sum()) if len(aligned) else 0.0
    same_weight = float(aligned.loc[same_season].sum()) if len(aligned) else 0.0
    previous_weight = float(aligned.loc[previous_season].sum()) if len(aligned) else 0.0
    older_weight = float(aligned.loc[older_season].sum()) if len(aligned) else 0.0
    event_keys = _event_key_series(train)
    return {
        "fold_id": None,
        "test_event": test_event,
        "test_season": test_season,
        "temporal_weighting_policy": policy.value,
        "training_rows": int(len(train)),
        "training_events": int(event_keys.nunique()) if not train.empty else 0,
        "same_season_training_rows": int(same_season.sum()) if len(train) else 0,
        "prior_season_training_rows": int(previous_season.sum()) if len(train) else 0,
        "older_season_training_rows": int(older_season.sum()) if len(train) else 0,
        "same_season_training_events": (
            int(event_keys.loc[same_season].nunique()) if len(train) else 0
        ),
        "prior_season_training_events": (
            int(event_keys.loc[previous_season].nunique()) if len(train) else 0
        ),
        "older_season_training_events": (
            int(event_keys.loc[older_season].nunique()) if len(train) else 0
        ),
        "weight_min": _number_or_none(aligned.min()) if len(aligned) else None,
        "weight_max": _number_or_none(aligned.max()) if len(aligned) else None,
        "weight_mean": _number_or_none(aligned.mean()) if len(aligned) else None,
        "weight_sum": total_weight,
        "effective_sample_size": effective_sample_size(aligned),
        "same_season_weight_sum": same_weight,
        "prior_season_weight_sum": previous_weight,
        "older_season_weight_sum": older_weight,
        "same_season_weight_share": _safe_share(same_weight, total_weight),
        "prior_season_weight_share": _safe_share(previous_weight, total_weight),
        "older_season_weight_share": _safe_share(older_weight, total_weight),
        "filtered_future_rows": filtered_future_rows,
    }


def effective_sample_size(weights: pd.Series) -> float | None:
    """Return standard effective sample size for non-negative sample weights."""
    numeric = pd.to_numeric(weights, errors="coerce").dropna().astype(float)
    if numeric.empty:
        return None
    denominator = float((numeric**2).sum())
    if denominator <= 0:
        return None
    return float(numeric.sum() ** 2 / denominator)


def temporal_artifact_stem(base: str, policy: TemporalWeightingPolicy | str) -> str:
    """Return the policy-specific artifact stem for a canonical base name."""
    policy = TemporalWeightingPolicy(policy)
    return f"{base}_{policy.value}"


def supported_weighted_models(*, boosted: bool = False) -> tuple[str, ...]:
    """Models that receive sample weights in the current implementation."""
    if boosted:
        return ("hist_gradient_boosting",)
    return ("ridge", "random_forest")


def unsupported_weighted_models(
    *,
    boosted: bool = False,
    include_constants: bool = True,
) -> tuple[str, ...]:
    """Model names intentionally left unweighted."""
    if boosted or not include_constants:
        return ()
    return ("mean_target", "median_target")


def _weights_for_policy(
    frame: pd.DataFrame,
    *,
    row_events: pd.Series,
    test_event: str,
    event_order: list[str],
    config: TemporalWeightingConfig,
    policy: TemporalWeightingPolicy,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float, index=frame.index)
    if policy is TemporalWeightingPolicy.season_priority:
        return _season_priority_weights(frame, test_event, config)
    if policy is TemporalWeightingPolicy.current_season_only_with_prior:
        return _season_priority_weights(frame, test_event, config)
    if policy is TemporalWeightingPolicy.exponential_recency:
        order_index = {event: index for index, event in enumerate(event_order)}
        test_index = order_index[test_event]
        distances = row_events.map(lambda event: test_index - order_index[str(event)]).astype(float)
        return 0.5 ** (distances / config.half_life_events)
    return pd.Series(1.0, index=frame.index, dtype=float)


def _season_priority_weights(
    frame: pd.DataFrame,
    test_event: str,
    config: TemporalWeightingConfig,
) -> pd.Series:
    test_season = _event_season(test_event)
    seasons = frame["season"].astype(int)
    weights = pd.Series(config.older_season_weight, index=frame.index, dtype=float)
    weights.loc[seasons.eq(test_season - 1)] = config.previous_season_weight
    weights.loc[seasons.eq(test_season)] = config.current_season_weight
    return weights


def _same_season_prior_events(train: pd.DataFrame, test_event: str) -> set[str]:
    test_season = _event_season(test_event)
    if train.empty:
        return set()
    rows = train[train["season"].astype(int).eq(test_season)]
    return set(_event_key_series(rows).astype(str).tolist())


def _event_key_series(dataset: pd.DataFrame) -> pd.Series:
    if dataset.empty:
        return pd.Series(dtype=str, index=dataset.index)
    return dataset["season"].astype(str) + "/" + dataset["event_slug"].astype(str)


def _event_season(event_key: str) -> int:
    return int(str(event_key).split("/", maxsplit=1)[0])


def _safe_share(value: float, total: float) -> float | None:
    if total <= 0:
        return None
    return float(value / total)


def _number_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
