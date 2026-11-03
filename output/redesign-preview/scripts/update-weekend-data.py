"""Cache free calendar and circuit geometry data for the isolated preview."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
CALENDAR_URL = "https://raw.githubusercontent.com/theOehrly/f1schedule/master/schedule_{season}.json"
GEOMETRY_URL = "https://raw.githubusercontent.com/bacinger/f1-circuits/master/f1-circuits.geojson"
LICENSE_URL = "https://raw.githubusercontent.com/bacinger/f1-circuits/master/LICENSE.md"
COUNTRY_CODES = {
    "Australia": "AU", "Bahrain": "BH", "China": "CN", "Japan": "JP",
    "United States": "US", "Canada": "CA", "Monaco": "MC", "Spain": "ES",
    "Austria": "AT", "United Kingdom": "GB", "Belgium": "BE", "Hungary": "HU",
    "Netherlands": "NL", "Italy": "IT", "Azerbaijan": "AZ", "Singapore": "SG",
    "Mexico": "MX", "Brazil": "BR", "Qatar": "QA", "United Arab Emirates": "AE",
}
LOCATION_CIRCUITS = {
    "Melbourne": "au-1953", "Sakhir": "bh-2002", "Shanghai": "cn-2004",
    "Suzuka": "jp-1962", "Miami Gardens": "us-2022", "Miami": "us-2022",
    "Montréal": "ca-1978", "Montreal": "ca-1978", "Monte Carlo": "mc-1929",
    "Barcelona": "es-1991", "Spielberg": "at-1969", "Silverstone": "gb-1948",
    "Spa-Francorchamps": "be-1925", "Budapest": "hu-1986", "Zandvoort": "nl-1948",
    "Monza": "it-1922", "Madrid": "es-2026", "Baku": "az-2016",
    "Marina Bay": "sg-2008", "Austin": "us-2012", "Mexico City": "mx-1962",
    "São Paulo": "br-1940", "Las Vegas": "us-2023", "Lusail": "qa-2004",
    "Yas Marina": "ae-2009", "Jeddah": "sa-2021",
}
SESSION_CODES = {
    "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
    "Sprint Qualifying": "SQ", "Sprint Shootout": "SQ", "Sprint": "SPRINT",
    "Qualifying": "Q", "Race": "R",
}


def fetch(url: str) -> bytes:
    with urlopen(url, timeout=30) as response:
        return response.read()


def build_events(payload: dict, season: int) -> list[dict]:
    """Convert explicit track-local dates plus their UTC offsets without guessing timezones."""
    events = []
    for index, round_number in payload["round_number"].items():
        if round_number <= 0:
            continue
        offset = payload["gmt_offset"].get(index)
        if not offset:
            raise ValueError(f"Missing UTC offset for {season}/{index}")
        sessions = []
        for sequence in range(1, 6):
            name = payload[f"session{sequence}"].get(index)
            value = payload[f"session{sequence}_date"].get(index)
            if not name or not value:
                continue
            local = datetime.fromisoformat(value)
            if local.tzinfo is None:
                local = datetime.fromisoformat(value + offset)
            sessions.append({
                "code": SESSION_CODES.get(name, name), "name": name,
                "startUtc": local.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            })
        if not any(session["code"] == "R" for session in sessions):
            raise ValueError(f"Missing race start for {season}/{index}")
        location = payload["location"][index]
        country = payload["country"][index]
        events.append({
            "id": f"{season}-{round_number}", "season": season, "round": round_number,
            "name": payload["event_name"][index], "country": country,
            "countryCode": COUNTRY_CODES.get(country), "location": location,
            "format": payload["event_format"][index],
            "circuitId": LOCATION_CIRCUITS.get(location), "sessions": sessions,
        })
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, action="append", required=True)
    parser.add_argument("--calendar-file", type=Path, help="Offline input for one season")
    parser.add_argument("--circuits-file", type=Path, help="Offline GeoJSON input")
    args = parser.parse_args()
    if args.calendar_file and len(args.season) != 1:
        parser.error("--calendar-file requires exactly one --season")
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    sources, events = [], []
    for season in args.season:
        url = CALENDAR_URL.format(season=season)
        raw = args.calendar_file.read_bytes() if args.calendar_file else fetch(url)
        events.extend(build_events(json.loads(raw), season))
        sources.append({"kind": "calendar", "url": url, "sha256": sha256(raw).hexdigest()})
    geo_raw = args.circuits_file.read_bytes() if args.circuits_file else fetch(GEOMETRY_URL)
    circuits = {}
    for feature in json.loads(geo_raw)["features"]:
        if feature["geometry"]["type"] != "LineString":
            continue
        props = feature["properties"]
        circuits[props["id"]] = {
            "name": props["Name"],
            "points": [[round(p[0], 7), round(p[1], 7)] for p in feature["geometry"]["coordinates"]],
        }
    sources.append({"kind": "geometry", "url": GEOMETRY_URL, "sha256": sha256(geo_raw).hexdigest()})
    result = {
        "schemaVersion": 1, "retrievedAtUtc": datetime.now(UTC).isoformat(),
        "sources": sources, "events": events, "circuits": circuits,
    }
    (data_dir / "weekends.json").write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    if not args.circuits_file:
        (data_dir / "CIRCUITS-LICENSE.md").write_bytes(fetch(LICENSE_URL))
    print(f"Cached {len(events)} weekends and {len(circuits)} circuit outlines.")
    missing = [event["name"] for event in events if event["circuitId"] not in circuits]
    if missing:
        print("No matching geometry (safe unavailable state): " + ", ".join(missing))


if __name__ == "__main__":
    main()
