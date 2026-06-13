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
from f1_prediction.data.ingest import DEFAULT_EVENT_SESSIONS, EventIngestionSummary
from f1_prediction.data.ingest import ingest_event as run_event_ingestion
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


@app.command("ingest-event")
def ingest_event_command(
    season: Annotated[int, typer.Option(min=1950, help="Championship season.")],
    event: Annotated[str, typer.Option(help="Event name or circuit location, e.g. Monza.")],
    sessions: Annotated[
        list[str] | None,
        typer.Option(
            "--sessions",
            help="Sessions to ingest. Defaults to FP1, FP2, FP3, and Q.",
        ),
    ] = None,
    additional_sessions: Annotated[
        list[str] | None,
        typer.Argument(hidden=True),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Reload and overwrite successful existing outputs."),
    ] = False,
    fail_fast: Annotated[
        bool,
        typer.Option("--fail-fast", help="Stop after the first failed session."),
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to a data YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Ingest practice and qualifying sessions for one race weekend."""
    configure_logging(verbose=verbose)
    config = load_data_config(config_path=config_path)
    requested_sessions = [*(sessions or []), *(additional_sessions or [])]
    summary = run_event_ingestion(
        season=season,
        event=event,
        config=config,
        sessions=requested_sessions or DEFAULT_EVENT_SESSIONS,
        force=force,
        fail_fast=fail_fast,
        progress=typer.echo,
    )
    _print_ingestion_summary(summary, fail_fast=fail_fast)
    if summary.failed_count:
        raise typer.Exit(code=1)


def _print_summary(result: SessionLoadResult) -> None:
    typer.echo("Session loaded successfully")
    typer.echo(f"Season: {result.season}")
    typer.echo(f"Event: {result.event}")
    typer.echo(f"Session: {result.session}")
    typer.echo(f"Drivers: {result.driver_count}")
    typer.echo(f"Laps: {result.lap_count}")
    typer.echo(f"Output: {result.output_path}")


def _print_ingestion_summary(summary: EventIngestionSummary, *, fail_fast: bool) -> None:
    typer.echo("")
    typer.echo(f"Event ingestion summary: {summary.season} {summary.event}")
    typer.echo(f"Successful: {summary.success_count}")
    typer.echo(f"Skipped: {summary.skipped_count}")
    typer.echo(f"Failed: {summary.failed_count}")
    if fail_fast and summary.failed_count:
        typer.echo("Stopped after the first failed session.")


if __name__ == "__main__":
    app()
