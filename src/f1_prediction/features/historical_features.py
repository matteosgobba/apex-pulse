"""Leakage-safe historical qualifying form features."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class HistoricalFeatureSettings:
    """Rolling window and minimum-history settings."""

    rolling_windows: tuple[int, ...] = (3, 5)
    min_periods: int = 1


HISTORICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "driver_prev_events_count",
    "driver_rolling3_quali_gap_mean",
    "driver_rolling3_quali_gap_median",
    "driver_rolling3_quali_position_mean",
    "driver_rolling3_q3_rate",
    "driver_rolling5_quali_gap_mean",
    "driver_rolling5_quali_position_mean",
    "driver_rolling5_q3_rate",
    "driver_expanding_quali_gap_mean",
    "driver_expanding_quali_position_mean",
    "driver_expanding_q3_rate",
    "team_prev_events_count",
    "team_rolling3_quali_gap_mean",
    "team_rolling3_quali_position_mean",
    "team_rolling3_q3_rate",
    "team_rolling5_quali_gap_mean",
    "team_rolling5_quali_position_mean",
    "team_rolling5_q3_rate",
    "team_expanding_quali_gap_mean",
    "team_expanding_quali_position_mean",
    "team_expanding_q3_rate",
    "driver_rolling3_gap_to_teammate_quali_mean",
    "driver_rolling5_gap_to_teammate_quali_mean",
    "driver_expanding_gap_to_teammate_quali_mean",
)

_TARGET_COLUMNS = ("quali_gap_to_pole_sec", "quali_position", "reached_q3")
_EVENT_COLUMNS = ("season", "event_slug")


def add_historical_features(
    dataset: pd.DataFrame,
    settings: HistoricalFeatureSettings | None = None,
    *,
    excluded_target_events: Collection[str] = (),
) -> pd.DataFrame:
    """Add prior-event driver and team form without using current/future targets.

    ``excluded_target_events`` is used by event-holdout evaluation so a held-out
    event cannot become a history source for later training rows.
    """
    required = {*_EVENT_COLUMNS, "driver", "team", *_TARGET_COLUMNS}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Historical feature input is missing columns: {', '.join(missing)}")

    config = settings or HistoricalFeatureSettings()
    if config.min_periods < 1 or not config.rolling_windows:
        raise ValueError("Historical rolling windows and min_periods must be positive")
    unsupported = set(config.rolling_windows) - {3, 5}
    if unsupported:
        raise ValueError("Historical features currently support rolling windows 3 and 5")

    frame = dataset.drop(columns=list(HISTORICAL_FEATURE_COLUMNS), errors="ignore").copy()
    frame["_event_key"] = _event_key_series(frame)
    order = _event_order_table(frame)
    driver_events = _driver_event_outcomes(frame, order)
    excluded = set(excluded_target_events)
    driver_events["_history_source"] = ~driver_events["_event_key"].isin(excluded)
    driver_events["_teammate_gap"] = _teammate_gap(driver_events)

    driver_features = _rolling_entity_features(
        driver_events,
        entity_columns=["driver"],
        value_columns={
            "quali_gap_to_pole_sec": "quali_gap",
            "quali_position": "quali_position",
            "reached_q3": "q3",
            "_teammate_gap": "gap_to_teammate_quali",
        },
        prefix="driver",
        settings=config,
    )
    driver_feature_columns = [
        column for column in HISTORICAL_FEATURE_COLUMNS if column.startswith("driver_")
    ]
    frame = frame.merge(
        driver_features.loc[:, [*_EVENT_COLUMNS, "driver", *driver_feature_columns]],
        on=[*_EVENT_COLUMNS, "driver"],
        how="left",
        validate="many_to_one",
    )

    team_events = _team_event_outcomes(driver_events, order)
    team_features = _rolling_entity_features(
        team_events,
        entity_columns=["team"],
        value_columns={
            "quali_gap_to_pole_sec": "quali_gap",
            "quali_position": "quali_position",
            "reached_q3": "q3",
        },
        prefix="team",
        settings=config,
    )
    team_feature_columns = [
        column for column in HISTORICAL_FEATURE_COLUMNS if column.startswith("team_")
    ]
    frame = frame.merge(
        team_features.loc[:, [*_EVENT_COLUMNS, "team", *team_feature_columns]],
        on=[*_EVENT_COLUMNS, "team"],
        how="left",
        validate="many_to_one",
    )
    frame[list(HISTORICAL_FEATURE_COLUMNS)] = (
        frame[list(HISTORICAL_FEATURE_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .astype("float64")
    )
    return frame.drop(columns="_event_key")


def _driver_event_outcomes(frame: pd.DataFrame, order: pd.DataFrame) -> pd.DataFrame:
    columns = [*_EVENT_COLUMNS, "_event_key", "driver", "team", *_TARGET_COLUMNS]
    outcomes = frame.loc[:, columns].drop_duplicates([*_EVENT_COLUMNS, "driver"])
    return outcomes.merge(order, on=[*_EVENT_COLUMNS, "_event_key"], validate="many_to_one")


def _team_event_outcomes(driver_events: pd.DataFrame, order: pd.DataFrame) -> pd.DataFrame:
    valid = driver_events[driver_events["team"].notna()].copy()
    values = list(_TARGET_COLUMNS)
    source_values = valid[values].where(valid["_history_source"], pd.NA)
    valid[values] = source_values
    teams = (
        valid.groupby([*_EVENT_COLUMNS, "_event_key", "team"], sort=False, dropna=False)[values]
        .mean()
        .reset_index()
    )
    teams["_history_source"] = ~teams["_event_key"].isin(
        driver_events.loc[~driver_events["_history_source"], "_event_key"]
    )
    return teams.merge(order, on=[*_EVENT_COLUMNS, "_event_key"], validate="many_to_one")


def _rolling_entity_features(
    events: pd.DataFrame,
    *,
    entity_columns: Sequence[str],
    value_columns: dict[str, str],
    prefix: str,
    settings: HistoricalFeatureSettings,
) -> pd.DataFrame:
    ordered = events.sort_values([*entity_columns, "_event_index"], kind="stable").copy()
    feature_rows: list[dict[str, object]] = []
    group_key: str | list[str] = (
        entity_columns[0] if len(entity_columns) == 1 else list(entity_columns)
    )
    for _, group in ordered.groupby(group_key, sort=False, dropna=False):
        history: list[dict[str, object]] = []
        for row in group.to_dict("records"):
            output = {column: row[column] for column in [*_EVENT_COLUMNS, *entity_columns]}
            output[f"{prefix}_prev_events_count"] = len(history)
            for window in settings.rolling_windows:
                window_rows = history[-window:]
                for source_column, label in value_columns.items():
                    values = _numeric_history(window_rows, source_column)
                    if len(values) >= settings.min_periods:
                        output[f"{prefix}_rolling{window}_{label}_mean"] = sum(values) / len(values)
                        if prefix == "driver" and label == "quali_gap" and window == 3:
                            output[f"{prefix}_rolling{window}_{label}_median"] = float(
                                pd.Series(values).median()
                            )
                    else:
                        output[f"{prefix}_rolling{window}_{label}_mean"] = pd.NA
                        if prefix == "driver" and label == "quali_gap" and window == 3:
                            output[f"{prefix}_rolling{window}_{label}_median"] = pd.NA
            for source_column, label in value_columns.items():
                values = _numeric_history(history, source_column)
                output[f"{prefix}_expanding_{label}_mean"] = (
                    sum(values) / len(values) if len(values) >= settings.min_periods else pd.NA
                )
            feature_rows.append(output)
            if row["_history_source"]:
                history.append(row)
    features = pd.DataFrame(feature_rows)
    return features.rename(
        columns={
            "driver_rolling3_q3_mean": "driver_rolling3_q3_rate",
            "driver_rolling5_q3_mean": "driver_rolling5_q3_rate",
            "driver_expanding_q3_mean": "driver_expanding_q3_rate",
            "team_rolling3_q3_mean": "team_rolling3_q3_rate",
            "team_rolling5_q3_mean": "team_rolling5_q3_rate",
            "team_expanding_q3_mean": "team_expanding_q3_rate",
        }
    )


def _teammate_gap(driver_events: pd.DataFrame) -> pd.Series:
    gaps = pd.Series(pd.NA, index=driver_events.index, dtype="Float64")
    group_columns = [*_EVENT_COLUMNS, "team"]
    for _, group in driver_events.groupby(group_columns, sort=False, dropna=False):
        if len(group) < 2:
            continue
        values = pd.to_numeric(group["quali_gap_to_pole_sec"], errors="coerce")
        for index in group.index:
            teammate_values = values.drop(index).dropna()
            driver_value = values.loc[index]
            if pd.notna(driver_value) and not teammate_values.empty:
                gaps.loc[index] = float(driver_value) - float(teammate_values.min())
    return gaps


def _event_order_table(frame: pd.DataFrame) -> pd.DataFrame:
    events = frame.loc[:, [*_EVENT_COLUMNS, "_event_key"]].copy()
    events["_appearance"] = range(len(events))
    if "event_order" in frame:
        events["_round_order"] = pd.to_numeric(frame["event_order"], errors="coerce")
    else:
        events["_round_order"] = events.groupby("season", sort=False)["event_slug"].transform(
            lambda values: pd.factorize(values, sort=False)[0] + 1
        )
    events = events.drop_duplicates(list(_EVENT_COLUMNS))
    events = events.sort_values(
        ["season", "_round_order", "_appearance"], kind="stable", na_position="last"
    ).reset_index(drop=True)
    events["_event_index"] = range(len(events))
    return events.loc[:, [*_EVENT_COLUMNS, "_event_key", "_event_index"]]


def _event_key_series(frame: pd.DataFrame) -> pd.Series:
    return frame["season"].astype(str) + "/" + frame["event_slug"].astype(str)


def _numeric_history(rows: list[dict[str, object]], column: str) -> list[float]:
    values = pd.to_numeric(pd.Series([row.get(column) for row in rows]), errors="coerce")
    return values.dropna().astype(float).tolist()
