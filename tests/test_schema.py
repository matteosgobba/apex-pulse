import pandas as pd
import pytest

from f1_prediction.data.schema import REQUIRED_LAP_COLUMNS, validate_lap_schema


def test_validate_lap_schema_accepts_required_columns() -> None:
    laps = pd.DataFrame(columns=REQUIRED_LAP_COLUMNS)

    validate_lap_schema(laps)


def test_validate_lap_schema_reports_missing_columns() -> None:
    laps = pd.DataFrame(columns=["Driver", "LapTime"])

    with pytest.raises(ValueError, match="LapNumber"):
        validate_lap_schema(laps)
