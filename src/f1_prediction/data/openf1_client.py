"""Bounded, cached access to OpenF1's unauthenticated historical API."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from f1_prediction.data.fastf1_loader import SessionDataUnavailableError

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://api.openf1.org/v1"


class OpenF1UnavailableError(SessionDataUnavailableError):
    """Historical data is unavailable; retry without buying live access."""

    retryable = True


class OpenF1Client:
    """Cache successful responses and stay below the free 30 requests/minute limit.

    No credentials are read or transmitted. Empty responses and HTTP errors are
    never cached, so a pre-publication response cannot poison later retries.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._last_request_at: float | None = None

    def get(
        self, endpoint: str, *, allow_empty_not_found: bool = False, **filters: Any
    ) -> list[dict[str, Any]]:
        url = f"{BASE_URL}/{endpoint}?{urlencode(sorted(filters.items()))}"
        path = self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.json"
        if path.is_file() and time.time() - path.stat().st_mtime < 300:
            return _validate_rows(json.loads(path.read_text()))
        if self._last_request_at is not None:
            time.sleep(max(0, 2.1 - (time.monotonic() - self._last_request_at)))
        self._last_request_at = time.monotonic()
        try:
            request = Request(url, headers={"User-Agent": "f1-prediction/0.1 (historical data)"})
            with urlopen(request, timeout=30) as response:
                rows = _validate_rows(json.load(response))
        except HTTPError as exc:
            if exc.code == 404 and allow_empty_not_found:
                LOGGER.info("OpenF1 %s: no published rows (404)", endpoint)
                return []
            raise OpenF1UnavailableError(
                f"OpenF1 {endpoint} returned HTTP {exc.code}. "
                "Only free historical access is enabled; live-window restrictions, "
                "rate limits and unpublished data will be retried."
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OpenF1UnavailableError(f"OpenF1 {endpoint} request failed; retry later.") from exc
        if rows:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(rows) + "\n")
            temporary.replace(path)
        LOGGER.info("OpenF1 %s: %s rows", endpoint, len(rows))
        return rows


def _validate_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise OpenF1UnavailableError("OpenF1 returned an invalid response; expected record rows.")
    return value
