"""Logging configuration for command-line workflows."""

import logging


def configure_logging(verbose: bool = False) -> None:
    """Configure concise application logging once."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
