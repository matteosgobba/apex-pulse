"""Stable driver and team identifiers across FastF1 seasons."""

from __future__ import annotations

import pandas as pd

from f1_prediction.utils.paths import slugify

TEAM_ALIASES: dict[str, tuple[str, ...]] = {
    "red_bull": ("Red Bull Racing", "Oracle Red Bull Racing"),
    "ferrari": ("Ferrari", "Scuderia Ferrari"),
    "mercedes": ("Mercedes", "Mercedes-AMG Petronas Formula One Team"),
    "mclaren": ("McLaren", "McLaren Formula 1 Team"),
    "aston_martin": ("Aston Martin", "Aston Martin Aramco"),
    "alpine": ("Alpine", "Alpine F1 Team", "BWT Alpine F1 Team"),
    "williams": ("Williams", "Williams Racing"),
    "sauber": (
        "Alfa Romeo",
        "Alfa Romeo Racing",
        "Kick Sauber",
        "Stake F1 Team Kick Sauber",
    ),
    "haas": ("Haas", "Haas F1 Team", "MoneyGram Haas F1 Team"),
    "rb": ("RB", "AlphaTauri", "Scuderia AlphaTauri", "Visa Cash App RB"),
}

_TEAM_ALIAS_LOOKUP = {
    slugify(alias): team_key for team_key, aliases in TEAM_ALIASES.items() for alias in aliases
}


def normalize_team_key(team_name: object) -> str | None:
    """Map a FastF1 team label to a stable key with a slug fallback."""
    value = _clean_value(team_name)
    if value is None:
        return None
    slug = slugify(value)
    return _TEAM_ALIAS_LOOKUP.get(slug, slug.replace("-", "_"))


def normalize_driver_key(driver_code: object, driver_name: object = None) -> str | None:
    """Prefer a stable three-letter code, then fall back to a name slug."""
    code = _clean_value(driver_code)
    if code is not None:
        normalized_code = "".join(character for character in code.upper() if character.isalnum())
        if normalized_code:
            return normalized_code.lower()
    name = _clean_value(driver_name)
    if name is None:
        return None
    return slugify(name).replace("-", "_")


def add_identity_columns(
    frame: pd.DataFrame,
    *,
    driver_column: str = "driver",
    team_column: str = "team",
) -> pd.DataFrame:
    """Add normalized identity columns while preserving all source columns."""
    result = frame.copy()
    driver_values = _string_column(result, driver_column)
    team_values = _string_column(result, team_column)
    existing_names = _string_column(result, "driver_name")
    driver_names = existing_names.fillna(driver_values)

    result["driver_code"] = driver_values.str.upper()
    result["driver_name"] = driver_names
    result["driver_key"] = pd.Series(
        [
            normalize_driver_key(code, name)
            for code, name in zip(driver_values, driver_names, strict=True)
        ],
        index=result.index,
        dtype="string",
    )
    result["team_name"] = team_values
    result["team_key"] = pd.Series(
        [normalize_team_key(team) for team in team_values],
        index=result.index,
        dtype="string",
    )
    return result


def _string_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    values = frame[column].astype("string").str.strip()
    return values.mask(values.eq(""))


def _clean_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip()
    return cleaned or None
