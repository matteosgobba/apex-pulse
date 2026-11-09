from datetime import datetime, timezone

import pytest

from f1_prediction.data.fia_entry_list import (
    DOCUMENT_PREFIX,
    FIAEntryListPending,
    _event_documents,
    parse_fia_entry_list,
    refresh_fia_entry_list,
)

EVENT = "Example Grand Prix"
WEEKEND = "2026_example_grand_prix_-_"


def _document(name, time):
    return (
        f'<a href="{DOCUMENT_PREFIX}{name}.pdf">'
        f'Published on <span class="date-display-single">07.11.26 {time}</span> CET</a>'
    )


def test_entry_document_selection_respects_event_and_cutoff():
    html = "".join(
        [
            _document(WEEKEND + "revised_entry_list", "17:30"),
            _document(WEEKEND + "entry_list", "14:00"),
            _document(WEEKEND + "decision_-_driver_change_example_team", "15:00"),
            _document("2025_example_grand_prix_-_entry_list", "15:40"),
            _document("2026_other_grand_prix_-_entry_list", "15:40"),
        ]
    )
    cutoff = datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)
    documents = _event_documents(html, 2026, EVENT, cutoff)
    assert [kind for _, _, kind in documents] == ["entry_list", "change"]
    assert documents[-1][0].hour == 14


def test_wait_for_revised_entry_list_when_official_change_is_newer(monkeypatch):
    html = _document(WEEKEND + "entry_list", "14:00") + _document(
        WEEKEND + "decision_-_driver_change_example_team", "15:00"
    )
    monkeypatch.setattr("f1_prediction.data.fia_entry_list._download", lambda url: html.encode())
    cutoff = datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)
    with pytest.raises(FIAEntryListPending, match="revised FIA entry list"):
        refresh_fia_entry_list(None, season=2026, event=EVENT, cutoff=cutoff)


def test_parse_entrants_excludes_practice_only_section_and_validates_teams():
    text = """
        2026 EXAMPLE GRAND PRIX
        No. TLA Driver          Nat     Team            Constructor
        1    AAA Alice A        ITA     Example One     One Engine
        2    BBB Bob B          GBR     Example One     One Engine
        3    CCC Charlie C      BRA     Example Two     Two Engine
        4    NEW Dana D         FRA     Example Two     Two Engine
        In addition to the list of cars and drivers eligible to take part in the event
        91   DEV Evan E         USA     Example Two     Two Engine
    """
    rows = parse_fia_entry_list(text, season=2026, event=EVENT)
    assert [row["driver"] for row in rows] == ["AAA", "BBB", "CCC", "NEW"]
    assert rows[-1]["team"] == "Example Two"
    with pytest.raises(ValueError, match="complete, unique two-car teams"):
        parse_fia_entry_list(text.replace("NEW Dana D", "CCC Dana D"), season=2026, event=EVENT)
