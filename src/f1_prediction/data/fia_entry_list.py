"""Load the official, event-scoped FIA entry list available before qualifying."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime
from html import unescape
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from f1_prediction.config import DataConfig
from f1_prediction.utils.paths import slugify

LOGGER = logging.getLogger(__name__)
FIA_URL = "https://www.fia.com"
DOCUMENT_PREFIX = "/system/files/decision-document/"
PUBLICATION = re.compile(
    r"Published on\s*<span\b[^>]*>(\d{2}\.\d{2}\.\d{2} \d{2}:\d{2})</span>", re.I
)
TABLE_ROW = re.compile(r"^\s*(\d{1,3})\s+([A-Z]{3})\s+(.+?)\s+([A-Z]{3})\s+(.+?)\s{2,}(\S.*?)\s*$")


class FIAEntryListPending(ValueError):
    """The FIA posted a change but a corresponding usable entry list is pending."""

    retryable = True


def fia_roster_path(config: DataConfig, season: int, event: str) -> Path:
    return config.session_metadata_output_dir / str(season) / slugify(event) / "fia_roster.json"


def refresh_fia_entry_list(
    config: DataConfig, *, season: int, event: str, cutoff: datetime
) -> None:
    """Cache the latest published FIA roster, falling back to practice on outages."""
    try:
        index = _download(f"{FIA_URL}/documents/event/{quote(event, safe='')}").decode()
    except (OSError, ValueError) as exc:
        LOGGER.warning("FIA entry-list index unavailable for %s %s: %s", season, event, exc)
        return

    documents = _event_documents(index, season, event, cutoff)
    entries = [doc for doc in documents if doc[2] == "entry_list"]
    if not entries or any(doc[0] > entries[-1][0] for doc in documents if doc[2] == "change"):
        if documents:
            raise FIAEntryListPending(f"Waiting for a revised FIA entry list for {season} {event}")
        return
    published_at, path, _ = entries[-1]
    url = FIA_URL + path
    try:
        reader = PdfReader(BytesIO(_download(url)))
        text = "\n".join(page.extract_text(extraction_mode="layout") for page in reader.pages)
        roster = parse_fia_entry_list(text, season=season, event=event)
    except (OSError, PdfReadError, ValueError) as exc:
        if len(entries) > 1 or "revised" in path or any(doc[2] == "change" for doc in documents):
            raise FIAEntryListPending(
                f"Latest FIA entry list could not be read for {event}"
            ) from exc
        LOGGER.warning("FIA entry list unavailable for %s %s: %s", season, event, exc)
        return

    output = fia_roster_path(config, season, event)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "season": season,
                "event_slug": slugify(event),
                "document_url": url,
                "published_at_utc": published_at.isoformat(),
                "drivers": roster,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def _event_documents(
    index: str, season: int, event: str, cutoff: datetime
) -> list[tuple[datetime, str, str]]:
    documents = []
    for path, content in re.findall(r'<a\s+href="([^"]+)"[^>]*>(.*?)</a>', index, re.S):
        path = unescape(path)
        if not path.startswith(DOCUMENT_PREFIX) or not path.endswith(".pdf"):
            continue
        name, _, title = path.removeprefix(DOCUMENT_PREFIX).partition("_-_")
        if slugify(name.replace("_", " ")) != f"{season}-{slugify(event)}":
            continue
        if "entry_list" in title:
            kind = "entry_list"
        elif "driver_change" in title or "change_of_driver" in title:
            kind = "change"
        else:
            continue
        match = PUBLICATION.search(content)
        if match is None:
            raise FIAEntryListPending(f"FIA publication time missing for {event} {kind}")
        published = datetime.strptime(match[1], "%d.%m.%y %H:%M").replace(
            tzinfo=ZoneInfo("Europe/Paris")
        )
        if published <= cutoff:
            documents.append((published.astimezone(cutoff.tzinfo), path, kind))
    return sorted(set(documents))


def parse_fia_entry_list(text: str, *, season: int, event: str) -> list[dict[str, str]]:
    """Read competition entrants from the table above the practice-only section."""
    lines = text.splitlines()
    if not any(
        slugify(line) == f"{season}-{slugify(event)}"
        for line in lines
        if line.lstrip().startswith(str(season))
    ):
        raise ValueError("FIA entry-list event identity does not match")
    rows = []
    for line in lines:
        if "in addition to the list" in line.lower() or "the stewards" == line.strip().lower():
            # The header also names the stewards but does not occupy a line on its own.
            break
        match = TABLE_ROW.fullmatch(line)
        if match:
            number, driver, full_name, _, team, constructor = match.groups()
            rows.append(
                {
                    "driver": driver,
                    "driver_number": number,
                    "full_name": full_name.strip(),
                    "team": team.strip(),
                    "constructor": constructor.strip(),
                }
            )
    team_counts = Counter(row["team"] for row in rows)
    if not (
        len(team_counts) >= 2
        and set(team_counts.values()) == {2}
        and len({row["driver"] for row in rows}) == len(rows)
        and len({row["driver_number"] for row in rows}) == len(rows)
    ):
        raise ValueError("FIA entry list does not contain complete, unique two-car teams")
    return rows


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "f1-prediction/0.1 (public entry list)"})
    with urlopen(request, timeout=8) as response:
        if not response.geturl().startswith(f"{FIA_URL}/"):
            raise ValueError("FIA source redirected outside the official website")
        content = response.read(4_000_001)
    if len(content) > 4_000_000:
        raise ValueError("FIA source exceeds the expected document size")
    return content
