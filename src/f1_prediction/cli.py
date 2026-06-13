"""Command-line interface for data ingestion workflows."""

from pathlib import Path
from typing import Annotated

import typer

from f1_prediction.config import load_data_config
from f1_prediction.data.fastf1_loader import (
    SessionDataUnavailableError,
    SessionLoadResult,
    load_fastf1_session,
)
from f1_prediction.utils.logging import configure_logging

app = typer.Typer(
    help="Load historical Formula 1 data for qualifying prediction workflows.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Run a data ingestion command."""


@app.command("load-session")
def load_session_command(
    season: Annotated[int, typer.Option(min=1950, help="Championship season.")],
    event: Annotated[str, typer.Option(help="Event name or circuit location, e.g. Monza.")],
    session_identifier: Annotated[
        str,
        typer.Option("--session", help="FastF1 session identifier, e.g. FP1, FP2, or FP3."),
    ],
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to a data YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Load one FastF1 session and save its lap data as Parquet."""
    configure_logging(verbose=verbose)
    config = load_data_config(config_path=config_path)
    try:
        result = load_fastf1_session(
            season=season,
            event=event,
            session_identifier=session_identifier,
            config=config,
        )
    except SessionDataUnavailableError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_summary(result)


def _print_summary(result: SessionLoadResult) -> None:
    typer.echo("Session loaded successfully")
    typer.echo(f"Season: {result.season}")
    typer.echo(f"Event: {result.event}")
    typer.echo(f"Session: {result.session}")
    typer.echo(f"Drivers: {result.driver_count}")
    typer.echo(f"Laps: {result.lap_count}")
    typer.echo(f"Output: {result.output_path}")


if __name__ == "__main__":
    app()
