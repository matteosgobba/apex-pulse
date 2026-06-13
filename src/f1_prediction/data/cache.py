"""FastF1 cache initialization."""

from pathlib import Path

import fastf1

from f1_prediction.utils.paths import ensure_directory


def initialize_fastf1_cache(cache_dir: Path) -> Path:
    """Create and enable the local FastF1 cache directory."""
    resolved_cache_dir = ensure_directory(cache_dir)
    fastf1.Cache.enable_cache(str(resolved_cache_dir))
    return resolved_cache_dir
