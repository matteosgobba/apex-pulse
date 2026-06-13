"""Rule-based push-lap detection."""

from __future__ import annotations

import pandas as pd

from f1_prediction.config import PushLapConfig


def add_push_lap_flags(laps: pd.DataFrame, config: PushLapConfig) -> pd.DataFrame:
    """Mark valid laps close to driver-best and session-best pace."""
    flagged = laps.copy()
    valid_times = flagged["lap_time_sec"].where(flagged["is_valid_lap"])
    driver_best = valid_times.groupby(flagged["driver"]).transform("min")
    session_best = valid_times.min()
    session_best_available = pd.notna(session_best)
    allowed_compound = flagged["compound"].notna() & flagged["compound"].isin(
        config.allowed_compounds
    )

    flagged["is_push_lap"] = (
        flagged["is_valid_lap"]
        & allowed_compound
        & driver_best.notna()
        & session_best_available
        & flagged["lap_time_sec"].le(driver_best * config.driver_best_pct_threshold)
        & flagged["lap_time_sec"].le(session_best * config.session_best_pct_threshold)
    )
    return flagged
