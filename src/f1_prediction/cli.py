"""Command-line interface for data ingestion workflows."""

from pathlib import Path
from typing import Annotated

import typer

from f1_prediction.config import load_data_config, load_feature_config, load_model_config
from f1_prediction.data.fastf1_loader import (
    SessionDataUnavailableError,
    SessionLoadResult,
    load_fastf1_session,
)
from f1_prediction.data.ingest import DEFAULT_EVENT_SESSIONS, EventIngestionSummary
from f1_prediction.data.ingest import ingest_event as run_event_ingestion
from f1_prediction.data.season_builder import SeasonDatasetBuildSummary, resolve_event_selection
from f1_prediction.data.season_builder import build_season_dataset as run_season_dataset_build
from f1_prediction.features.build import (
    DEFAULT_PRACTICE_SESSIONS,
    SessionFeatureBuildSummary,
)
from f1_prediction.features.build import build_session_features as run_feature_build
from f1_prediction.features.modeling_dataset import ModelingDatasetBuildSummary
from f1_prediction.features.modeling_dataset import (
    build_modeling_dataset_files as run_modeling_dataset_build,
)
from f1_prediction.modeling.ablation import AblationBacktestSummary, run_ablation_backtest
from f1_prediction.modeling.backtest_report import BacktestReportSummary
from f1_prediction.modeling.backtest_report import create_backtest_report as run_backtest_report
from f1_prediction.modeling.backtest_tabular import (
    BacktestStrategy,
    TabularBacktestSummary,
    run_tabular_backtest,
)
from f1_prediction.modeling.boosted_backtest import (
    BoostedBacktestSummary,
    run_boosted_backtest,
)
from f1_prediction.modeling.champion_diagnostics import ChampionDiagnosticsSummary
from f1_prediction.modeling.champion_diagnostics import (
    create_champion_diagnostics_report as run_champion_diagnostics_report,
)
from f1_prediction.modeling.champion_policy import (
    ChampionBacktestSummary,
    ChampionSelectionMode,
    ChampionUncertaintyMethod,
    run_champion_backtest,
)
from f1_prediction.modeling.dataset_report import DatasetQualitySummary
from f1_prediction.modeling.dataset_report import (
    create_dataset_quality_report as run_dataset_quality_report,
)
from f1_prediction.modeling.diagnostics import DiagnosticsReportSummary
from f1_prediction.modeling.diagnostics import create_diagnostics_report as run_diagnostics_report
from f1_prediction.modeling.evaluate_baselines import BaselineEvaluationSummary
from f1_prediction.modeling.evaluate_baselines import evaluate_baselines as run_baseline_evaluation
from f1_prediction.modeling.policy_simulation import PolicySimulationSummary
from f1_prediction.modeling.policy_simulation import (
    create_policy_simulation_report as run_policy_simulation_report,
)
from f1_prediction.modeling.portfolio_report import PortfolioReportSummary
from f1_prediction.modeling.portfolio_report import create_portfolio_report as run_portfolio_report
from f1_prediction.modeling.splits import DatasetSplitSummary, SplitStrategy
from f1_prediction.modeling.splits import write_dataset_split_report as run_dataset_split
from f1_prediction.modeling.train_tabular import TabularTrainingSummary
from f1_prediction.modeling.train_tabular import train_tabular_models as run_tabular_training
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


@app.command("build-session-features")
def build_session_features_command(
    season: Annotated[int, typer.Option(min=1950, help="Championship season.")],
    event: Annotated[str, typer.Option(help="Event name or circuit location, e.g. Monza.")],
    sessions: Annotated[
        list[str] | None,
        typer.Option(
            "--sessions",
            help="Practice sessions to process. Defaults to FP1, FP2, and FP3.",
        ),
    ] = None,
    additional_sessions: Annotated[
        list[str] | None,
        typer.Argument(hidden=True),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild and overwrite requested session outputs."),
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to the data YAML configuration."),
    ] = None,
    features_config_path: Annotated[
        Path | None,
        typer.Option("--features-config", help="Optional path to the features YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Build cleaned practice laps and driver/session aggregate features."""
    configure_logging(verbose=verbose)
    data_config = load_data_config(config_path=config_path)
    feature_config = load_feature_config(
        config_path=features_config_path,
        project_root=data_config.project_root,
    )
    requested_sessions = [*(sessions or []), *(additional_sessions or [])]
    try:
        summary = run_feature_build(
            season=season,
            event=event,
            data_config=data_config,
            feature_config=feature_config,
            sessions=requested_sessions or DEFAULT_PRACTICE_SESSIONS,
            force=force,
            progress=typer.echo,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_feature_build_summary(summary, data_config.project_root)


@app.command("build-modeling-dataset")
def build_modeling_dataset_command(
    season: Annotated[int, typer.Option(min=1950, help="Championship season.")],
    event: Annotated[str, typer.Option(help="Event name or circuit location, e.g. Monza.")],
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild and overwrite an existing modeling dataset."),
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to the data YAML configuration."),
    ] = None,
    features_config_path: Annotated[
        Path | None,
        typer.Option("--features-config", help="Optional features YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Build qualifying targets and checkpoint-level modeling rows."""
    configure_logging(verbose=verbose)
    config = load_data_config(config_path=config_path)
    feature_config = load_feature_config(
        config_path=features_config_path,
        project_root=config.project_root,
    )
    try:
        summary = run_modeling_dataset_build(
            season=season,
            event=event,
            config=config,
            feature_config=feature_config,
            force=force,
            progress=typer.echo,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_modeling_dataset_summary(summary, config.project_root)


@app.command("build-season-dataset")
def build_season_dataset_command(
    seasons: Annotated[
        list[int],
        typer.Option(
            "--season",
            "--seasons",
            min=1950,
            help="Season to build. Repeat or list multiple seasons.",
        ),
    ],
    events: Annotated[
        list[str] | None,
        typer.Option("--events", help="Optional event filter. Repeat for multiple events."),
    ] = None,
    additional_values: Annotated[
        list[str] | None,
        typer.Argument(hidden=True),
    ] = None,
    preset: Annotated[
        str | None,
        typer.Option("--preset", help="Optional documented event-list preset."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild existing event pipeline outputs."),
    ] = False,
    fail_fast: Annotated[
        bool,
        typer.Option("--fail-fast", help="Stop after the first failed event."),
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to the data YAML configuration."),
    ] = None,
    features_config_path: Annotated[
        Path | None,
        typer.Option("--features-config", help="Optional path to the features YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Build and combine modeling datasets across scheduled events."""
    configure_logging(verbose=verbose)
    data_config = load_data_config(config_path=config_path)
    feature_config = load_feature_config(
        config_path=features_config_path,
        project_root=data_config.project_root,
    )
    trailing_values = additional_values or []
    trailing_seasons = [int(value) for value in trailing_values if value.isdigit()]
    trailing_events = [value for value in trailing_values if not value.isdigit()]
    requested_seasons = [*seasons, *trailing_seasons]
    try:
        requested_events = resolve_event_selection(
            requested_seasons,
            [*(events or []), *trailing_events],
            preset,
        )
        summary = run_season_dataset_build(
            seasons=requested_seasons,
            data_config=data_config,
            feature_config=feature_config,
            events=requested_events,
            preset=preset,
            force=force,
            fail_fast=fail_fast,
            progress=typer.echo,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_season_dataset_summary(summary, data_config.project_root)
    if summary.n_events_failed:
        raise typer.Exit(code=1)


@app.command("evaluate-baselines")
def evaluate_baselines_command(
    dataset_path: Annotated[
        Path | None,
        typer.Option("--dataset", help="Optional combined modeling dataset path."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to the data YAML configuration."),
    ] = None,
    features_config_path: Annotated[
        Path | None,
        typer.Option("--features-config", help="Optional features YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Evaluate transparent non-ML practice pace baselines."""
    configure_logging(verbose=verbose)
    config = load_data_config(config_path=config_path)
    feature_config = load_feature_config(
        config_path=features_config_path,
        project_root=config.project_root,
    )
    try:
        summary = run_baseline_evaluation(
            config,
            dataset_path=dataset_path,
            feature_config=feature_config,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_baseline_evaluation_summary(summary, config.project_root)


@app.command("dataset-report")
def dataset_report_command(
    dataset_path: Annotated[
        Path | None,
        typer.Option("--dataset", help="Optional combined modeling dataset path."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to the data YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Report modeling dataset coverage and missingness."""
    configure_logging(verbose=verbose)
    config = load_data_config(config_path=config_path)
    try:
        summary = run_dataset_quality_report(config, dataset_path=dataset_path)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_dataset_quality_summary(summary, config.project_root)


@app.command("split-dataset")
def split_dataset_command(
    strategy: Annotated[SplitStrategy, typer.Option(help="Time-aware split strategy.")],
    test_events: Annotated[
        list[str] | None,
        typer.Option("--test-events", help="Test event. Repeat for multiple events."),
    ] = None,
    test_seasons: Annotated[
        list[int] | None,
        typer.Option("--test-seasons", help="Test season. Repeat for multiple seasons."),
    ] = None,
    min_train_events: Annotated[
        int,
        typer.Option(min=1, help="Minimum prior events for walk-forward folds."),
    ] = 3,
    dataset_path: Annotated[
        Path | None,
        typer.Option("--dataset", help="Optional combined modeling dataset path."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to the data YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Create leakage-safe event, season, or walk-forward split metadata."""
    configure_logging(verbose=verbose)
    config = load_data_config(config_path=config_path)
    try:
        summary = run_dataset_split(
            config,
            strategy=strategy,
            dataset_path=dataset_path,
            test_events=test_events,
            test_seasons=test_seasons,
            min_train_events=min_train_events,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_dataset_split_summary(summary, config.project_root)


@app.command("train-tabular-models")
def train_tabular_models_command(
    dataset_path: Annotated[
        Path | None,
        typer.Option("--dataset", help="Optional combined modeling dataset path."),
    ] = None,
    test_season: Annotated[
        int | None,
        typer.Option(min=1950, help="Season held out from all training rows."),
    ] = None,
    test_events: Annotated[
        list[str] | None,
        typer.Option("--test-events", help="Test event. Repeat for multiple events."),
    ] = None,
    min_events: Annotated[
        int | None,
        typer.Option(min=2, help="Minimum unique events required before training."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to the data YAML configuration."),
    ] = None,
    model_config_path: Annotated[
        Path | None,
        typer.Option("--model-config", help="Optional path to the model YAML configuration."),
    ] = None,
    features_config_path: Annotated[
        Path | None,
        typer.Option("--features-config", help="Optional features YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Train simple checkpoint-safe Ridge and Random Forest regressors."""
    configure_logging(verbose=verbose)
    config = load_data_config(config_path=config_path)
    model_config = load_model_config(
        config_path=model_config_path,
        project_root=config.project_root,
    )
    feature_config = load_feature_config(
        config_path=features_config_path,
        project_root=config.project_root,
    )
    try:
        summary = run_tabular_training(
            config,
            dataset_path=dataset_path,
            test_season=test_season,
            test_events=test_events,
            min_events=min_events,
            model_config=model_config,
            feature_config=feature_config,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_tabular_training_summary(summary, config.project_root)


@app.command("backtest-report")
def backtest_report_command(
    dataset_path: Annotated[
        Path | None,
        typer.Option("--dataset", help="Optional combined modeling dataset path."),
    ] = None,
    baseline_metrics_path: Annotated[
        Path | None,
        typer.Option("--baseline-metrics", help="Optional baseline metrics JSON path."),
    ] = None,
    tabular_metrics_path: Annotated[
        Path | None,
        typer.Option("--tabular-metrics", help="Optional tabular metrics JSON path."),
    ] = None,
    quality_report_path: Annotated[
        Path | None,
        typer.Option("--quality-report", help="Optional dataset quality JSON path."),
    ] = None,
    ablation_metrics_path: Annotated[
        Path | None,
        typer.Option("--ablation-metrics", help="Optional ablation metrics JSON path."),
    ] = None,
    boosted_metrics_path: Annotated[
        Path | None,
        typer.Option("--boosted-metrics", help="Optional boosted metrics JSON path."),
    ] = None,
    champion_metrics_path: Annotated[
        Path | None,
        typer.Option("--champion-metrics", help="Optional champion metrics JSON path."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to the data YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Create a compact model-versus-baseline backtesting report."""
    configure_logging(verbose=verbose)
    config = load_data_config(config_path=config_path)
    try:
        summary = run_backtest_report(
            config,
            dataset_path=dataset_path,
            baseline_metrics_path=baseline_metrics_path,
            tabular_metrics_path=tabular_metrics_path,
            quality_report_path=quality_report_path,
            ablation_metrics_path=ablation_metrics_path,
            boosted_metrics_path=boosted_metrics_path,
            champion_metrics_path=champion_metrics_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_backtest_report_summary(summary, config.project_root)


@app.command("backtest-tabular-models")
def backtest_tabular_models_command(
    strategy: Annotated[BacktestStrategy, typer.Option(help="Repeated backtesting strategy.")],
    dataset_path: Annotated[
        Path | None,
        typer.Option("--dataset", help="Optional combined modeling dataset path."),
    ] = None,
    test_events: Annotated[
        list[str] | None,
        typer.Option("--test-events", help="Repeated-holdout test event filter."),
    ] = None,
    additional_test_events: Annotated[
        list[str] | None,
        typer.Argument(hidden=True),
    ] = None,
    min_events: Annotated[
        int,
        typer.Option(min=2, help="Minimum unique events required for backtesting."),
    ] = 5,
    min_train_events: Annotated[
        int,
        typer.Option(min=1, help="Minimum prior events before walk-forward testing."),
    ] = 5,
    fail_fast: Annotated[
        bool,
        typer.Option("--fail-fast", help="Stop after the first failed fold."),
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to the data YAML configuration."),
    ] = None,
    model_config_path: Annotated[
        Path | None,
        typer.Option("--model-config", help="Optional path to the model YAML configuration."),
    ] = None,
    features_config_path: Annotated[
        Path | None,
        typer.Option("--features-config", help="Optional features YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Backtest simple tabular models across repeated event-safe folds."""
    configure_logging(verbose=verbose)
    config = load_data_config(config_path=config_path)
    model_config = load_model_config(
        config_path=model_config_path,
        project_root=config.project_root,
    )
    feature_config = load_feature_config(
        config_path=features_config_path,
        project_root=config.project_root,
    )
    requested_test_events = [*(test_events or []), *(additional_test_events or [])]
    try:
        summary = run_tabular_backtest(
            config,
            strategy=strategy,
            dataset_path=dataset_path,
            test_events=requested_test_events or None,
            min_events=min_events,
            min_train_events=min_train_events,
            fail_fast=fail_fast,
            model_config=model_config,
            feature_config=feature_config,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_tabular_backtest_summary(summary, config.project_root)


@app.command("backtest-boosted-models")
def backtest_boosted_models_command(
    strategy: Annotated[
        BacktestStrategy,
        typer.Option(help="Event-safe backtesting strategy."),
    ] = BacktestStrategy.walk_forward,
    feature_policy: Annotated[
        str,
        typer.Option(help="Feature policy: checkpoint_best or all_features."),
    ] = "checkpoint_best",
    feature_group: Annotated[
        str | None,
        typer.Option(
            "--feature-group",
            help="Use one registered feature group at every checkpoint.",
        ),
    ] = None,
    dataset_path: Annotated[
        Path | None,
        typer.Option("--dataset", help="Optional combined modeling dataset path."),
    ] = None,
    test_events: Annotated[
        list[str] | None,
        typer.Option("--test-events", help="Repeated-holdout test event filter."),
    ] = None,
    additional_test_events: Annotated[
        list[str] | None,
        typer.Argument(hidden=True),
    ] = None,
    min_events: Annotated[
        int,
        typer.Option(min=2, help="Minimum unique events required for boosted backtesting."),
    ] = 10,
    min_train_events: Annotated[
        int,
        typer.Option(min=1, help="Minimum prior events before walk-forward testing."),
    ] = 5,
    fail_fast: Annotated[
        bool,
        typer.Option("--fail-fast", help="Stop after the first failed fold."),
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional data YAML configuration."),
    ] = None,
    model_config_path: Annotated[
        Path | None,
        typer.Option("--model-config", help="Optional model YAML configuration."),
    ] = None,
    features_config_path: Annotated[
        Path | None,
        typer.Option("--features-config", help="Optional features YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Backtest gradient-boosted models with checkpoint-safe feature policies."""
    configure_logging(verbose=verbose)
    data_config = load_data_config(config_path=config_path)
    model_config = load_model_config(
        config_path=model_config_path,
        project_root=data_config.project_root,
    )
    feature_config = load_feature_config(
        config_path=features_config_path,
        project_root=data_config.project_root,
    )
    requested_test_events = [*(test_events or []), *(additional_test_events or [])]
    try:
        summary = run_boosted_backtest(
            data_config,
            strategy=strategy,
            feature_policy=feature_policy,
            explicit_feature_group=feature_group,
            dataset_path=dataset_path,
            test_events=requested_test_events or None,
            min_events=min_events,
            min_train_events=min_train_events,
            fail_fast=fail_fast,
            model_config=model_config,
            feature_config=feature_config,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_boosted_backtest_summary(summary, data_config.project_root)


@app.command("champion-backtest")
def champion_backtest_command(
    strategy: Annotated[
        BacktestStrategy,
        typer.Option(help="Event-safe backtesting strategy."),
    ] = BacktestStrategy.walk_forward,
    selection_mode: Annotated[
        ChampionSelectionMode,
        typer.Option(
            help=(
                "Champion selection mode: static, nested, stabilized_nested, "
                "or stabilized_nested_guarded."
            )
        ),
    ] = ChampionSelectionMode.nested,
    uncertainty: Annotated[
        ChampionUncertaintyMethod,
        typer.Option(
            help=(
                "Prediction interval method: residual_std, conformal, "
                "or conformal_predicted_gap_bucket."
            )
        ),
    ] = ChampionUncertaintyMethod.residual_std,
    dataset_path: Annotated[
        Path | None,
        typer.Option("--dataset", help="Optional combined modeling dataset path."),
    ] = None,
    min_events: Annotated[
        int,
        typer.Option(min=2, help="Minimum unique events required for champion backtesting."),
    ] = 10,
    min_train_events: Annotated[
        int,
        typer.Option(min=1, help="Minimum prior events before walk-forward testing."),
    ] = 5,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional data YAML configuration."),
    ] = None,
    model_config_path: Annotated[
        Path | None,
        typer.Option("--model-config", help="Optional model YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Evaluate a checkpoint-specific static or nested champion policy."""
    configure_logging(verbose=verbose)
    data_config = load_data_config(config_path=config_path)
    model_config = load_model_config(
        config_path=model_config_path,
        project_root=data_config.project_root,
    )
    try:
        summary = run_champion_backtest(
            data_config,
            strategy=strategy,
            selection_mode=selection_mode,
            dataset_path=dataset_path,
            min_events=min_events,
            min_train_events=min_train_events,
            model_config=model_config,
            uncertainty_method=uncertainty,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_champion_backtest_summary(summary, data_config.project_root)


@app.command("diagnostics-report")
def diagnostics_report_command(
    walk_forward_path: Annotated[
        Path | None,
        typer.Option("--walk-forward", help="Optional walk-forward predictions path."),
    ] = None,
    repeated_path: Annotated[
        Path | None,
        typer.Option("--repeated-holdout", help="Optional repeated-holdout predictions path."),
    ] = None,
    baseline_path: Annotated[
        Path | None,
        typer.Option("--baselines", help="Optional full-dataset baseline predictions path."),
    ] = None,
    champion_path: Annotated[
        Path | None,
        typer.Option("--champion", help="Optional champion predictions path."),
    ] = None,
    dataset_path: Annotated[
        Path | None,
        typer.Option("--dataset", help="Optional combined modeling dataset path."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to the data YAML configuration."),
    ] = None,
    features_config_path: Annotated[
        Path | None,
        typer.Option("--features-config", help="Optional features YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Create event- and driver-level prediction error diagnostics."""
    configure_logging(verbose=verbose)
    data_config = load_data_config(config_path=config_path)
    feature_config = load_feature_config(
        config_path=features_config_path,
        project_root=data_config.project_root,
    )
    if feature_config.diagnostics is None:
        raise typer.BadParameter("Diagnostics configuration is unavailable")
    try:
        summary = run_diagnostics_report(
            data_config,
            feature_config.diagnostics,
            walk_forward_path=walk_forward_path,
            repeated_path=repeated_path,
            baseline_path=baseline_path,
            champion_path=champion_path,
            dataset_path=dataset_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_diagnostics_summary(summary, data_config.project_root)


@app.command("portfolio-report")
def portfolio_report_command(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to the data YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Generate portfolio-ready summary tables, figures, and model card."""
    configure_logging(verbose=verbose)
    data_config = load_data_config(config_path=config_path)
    try:
        summary = run_portfolio_report(data_config)
    except (ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_portfolio_report_summary(summary, data_config.project_root)


@app.command("champion-diagnostics")
def champion_diagnostics_command(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to the data YAML configuration."),
    ] = None,
    model_config_path: Annotated[
        Path | None,
        typer.Option("--model-config", help="Optional path to the model YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Generate champion-policy and conformal interval diagnostic reports."""
    configure_logging(verbose=verbose)
    data_config = load_data_config(config_path=config_path)
    model_config = load_model_config(
        config_path=model_config_path,
        project_root=data_config.project_root,
    )
    try:
        summary = run_champion_diagnostics_report(
            data_config,
            model_config.champion_diagnostics,
        )
    except (ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_champion_diagnostics_summary(summary, data_config.project_root)


@app.command("policy-simulation")
def policy_simulation_command(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional path to the data YAML configuration."),
    ] = None,
    model_config_path: Annotated[
        Path | None,
        typer.Option("--model-config", help="Optional path to the model YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Generate artifact-based champion guardrail and conformal simulations."""
    configure_logging(verbose=verbose)
    data_config = load_data_config(config_path=config_path)
    model_config = load_model_config(
        config_path=model_config_path,
        project_root=data_config.project_root,
    )
    try:
        summary = run_policy_simulation_report(
            data_config,
            model_config.policy_simulation,
        )
    except (ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_policy_simulation_summary(summary, data_config.project_root)


@app.command("ablation-backtest")
def ablation_backtest_command(
    strategy: Annotated[
        BacktestStrategy,
        typer.Option(help="Event-safe backtesting strategy."),
    ] = BacktestStrategy.walk_forward,
    feature_groups: Annotated[
        list[str] | None,
        typer.Option("--feature-groups", help="Feature groups to compare."),
    ] = None,
    additional_feature_groups: Annotated[
        list[str] | None,
        typer.Argument(hidden=True),
    ] = None,
    min_events: Annotated[
        int,
        typer.Option(min=2, help="Minimum unique events required for ablation."),
    ] = 10,
    min_train_events: Annotated[
        int,
        typer.Option(min=1, help="Minimum prior events for walk-forward folds."),
    ] = 5,
    fail_fast: Annotated[
        bool,
        typer.Option("--fail-fast", help="Stop after the first failed fold."),
    ] = False,
    dataset_path: Annotated[
        Path | None,
        typer.Option("--dataset", help="Optional combined modeling dataset path."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional data YAML configuration."),
    ] = None,
    model_config_path: Annotated[
        Path | None,
        typer.Option("--model-config", help="Optional model YAML configuration."),
    ] = None,
    features_config_path: Annotated[
        Path | None,
        typer.Option("--features-config", help="Optional features YAML configuration."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Compare feature groups using identical event-safe folds."""
    configure_logging(verbose=verbose)
    data_config = load_data_config(config_path=config_path)
    model_config = load_model_config(
        config_path=model_config_path,
        project_root=data_config.project_root,
    )
    feature_config = load_feature_config(
        config_path=features_config_path,
        project_root=data_config.project_root,
    )
    requested_groups = [*(feature_groups or []), *(additional_feature_groups or [])]
    try:
        summary = run_ablation_backtest(
            data_config,
            strategy=strategy,
            dataset_path=dataset_path,
            feature_group_names=requested_groups or None,
            min_events=min_events,
            min_train_events=min_train_events,
            fail_fast=fail_fast,
            model_config=model_config,
            feature_config=feature_config,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_ablation_summary(summary, data_config.project_root)


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


def _print_feature_build_summary(summary: SessionFeatureBuildSummary, project_root: Path) -> None:
    typer.echo("")
    typer.echo("Session features built successfully")
    typer.echo(f"Season: {summary.season}")
    typer.echo(f"Event: {summary.event}")
    typer.echo(f"Sessions: {', '.join(summary.sessions)}")
    typer.echo(f"Clean laps written: {summary.clean_lap_files_written} files")
    typer.echo(f"Aggregate rows: {summary.aggregate_rows}")
    typer.echo(f"Output: {_display_path(summary.output_path, project_root)}")


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _print_modeling_dataset_summary(
    summary: ModelingDatasetBuildSummary,
    project_root: Path,
) -> None:
    typer.echo("")
    typer.echo("Modeling dataset built successfully")
    typer.echo(f"Season: {summary.season}")
    typer.echo(f"Event: {summary.event}")
    typer.echo(f"Rows: {summary.rows}")
    typer.echo(f"Drivers: {summary.drivers}")
    typer.echo(f"Checkpoints: {', '.join(summary.checkpoints)}")
    typer.echo(f"Output: {_display_path(summary.output_path, project_root)}")
    if summary.practice_only_drivers:
        typer.echo(f"Practice-only drivers: {', '.join(summary.practice_only_drivers)}")
    if summary.qualifying_only_drivers:
        typer.echo(f"Qualifying-only drivers: {', '.join(summary.qualifying_only_drivers)}")


def _print_season_dataset_summary(
    summary: SeasonDatasetBuildSummary,
    project_root: Path,
) -> None:
    typer.echo("")
    typer.echo("Combined modeling dataset build complete")
    typer.echo(f"Seasons: {', '.join(str(season) for season in summary.requested_seasons)}")
    typer.echo(f"Events requested: {summary.n_events_requested}")
    typer.echo(f"Events successful: {summary.n_events_successful}")
    typer.echo(f"Events failed: {summary.n_events_failed}")
    typer.echo(f"Rows: {summary.n_rows}")
    typer.echo(f"Teams: {summary.n_teams}")
    typer.echo(f"Output: {_display_path(summary.output_path, project_root)}")
    typer.echo(f"Report: {_display_path(summary.report_path, project_root)}")
    for failed in summary.failed_events:
        typer.echo(f"Failed {failed.season} {failed.event}: {failed.error_message}")


def _print_baseline_evaluation_summary(
    summary: BaselineEvaluationSummary,
    project_root: Path,
) -> None:
    typer.echo("Baseline evaluation complete")
    typer.echo(f"Baselines: {', '.join(summary.baselines)}")
    typer.echo(f"Checkpoints: {', '.join(summary.checkpoints)}")
    typer.echo(f"Prediction rows: {summary.prediction_rows}")
    typer.echo(f"Metrics: {_display_path(summary.metrics_path, project_root)}")
    typer.echo(f"Predictions: {_display_path(summary.predictions_path, project_root)}")


def _print_dataset_quality_summary(summary: DatasetQualitySummary, project_root: Path) -> None:
    typer.echo("Dataset quality report complete")
    typer.echo(f"Rows: {summary.n_rows}")
    typer.echo(f"Seasons: {summary.n_seasons}")
    typer.echo(f"Events: {summary.n_events}")
    typer.echo(f"Drivers: {summary.n_drivers}")
    typer.echo(f"Checkpoints: {', '.join(summary.checkpoints)}")
    typer.echo(f"Historical features: {summary.historical_feature_count}")
    typer.echo(f"Data-quality features: {summary.data_quality_feature_count}")
    typer.echo(f"Report: {_display_path(summary.report_path, project_root)}")


def _print_dataset_split_summary(summary: DatasetSplitSummary, project_root: Path) -> None:
    typer.echo("Dataset split report complete")
    typer.echo(f"Strategy: {summary.strategy}")
    typer.echo(f"Train rows: {summary.train_rows}")
    typer.echo(f"Test rows: {summary.test_rows}")
    if summary.folds:
        typer.echo(f"Walk-forward folds: {summary.folds}")
    typer.echo(f"Report: {_display_path(summary.report_path, project_root)}")


def _print_tabular_training_summary(
    summary: TabularTrainingSummary,
    project_root: Path,
) -> None:
    if summary.status == "skipped":
        typer.echo("Tabular model training skipped")
        typer.echo(f"Reason: {summary.reason}")
        typer.echo(f"Report: {_display_path(summary.metrics_path, project_root)}")
        return
    typer.echo("Tabular model training complete")
    typer.echo(f"Events: {summary.n_events}")
    typer.echo(f"Train events: {summary.train_events}")
    typer.echo(f"Test events: {summary.test_events}")
    typer.echo(f"Models: {', '.join(summary.models)}")
    typer.echo(f"Prediction rows: {summary.prediction_rows}")
    typer.echo(f"Metrics: {_display_path(summary.metrics_path, project_root)}")
    if summary.predictions_path is not None:
        typer.echo(f"Predictions: {_display_path(summary.predictions_path, project_root)}")


def _print_backtest_report_summary(
    summary: BacktestReportSummary,
    project_root: Path,
) -> None:
    typer.echo("Backtest report complete")
    typer.echo(f"Dataset rows: {summary.dataset_rows}")
    typer.echo(f"Events: {summary.n_events}")
    typer.echo(f"Training status: {summary.training_status}")
    typer.echo(
        f"Tabular models: {', '.join(summary.tabular_models) if summary.tabular_models else 'none'}"
    )
    typer.echo(f"Report: {_display_path(summary.output_path, project_root)}")


def _print_tabular_backtest_summary(
    summary: TabularBacktestSummary,
    project_root: Path,
) -> None:
    typer.echo("Tabular backtest complete")
    typer.echo(f"Status: {summary.status}")
    typer.echo(f"Strategy: {summary.strategy}")
    typer.echo(f"Events: {summary.n_events}")
    typer.echo(f"Folds successful: {summary.n_folds_successful}")
    typer.echo(f"Folds failed: {summary.n_folds_failed}")
    typer.echo(f"Prediction rows: {summary.prediction_rows}")
    typer.echo(f"Metrics: {_display_path(summary.metrics_path, project_root)}")
    typer.echo(f"Folds: {_display_path(summary.folds_path, project_root)}")
    if summary.predictions_path is not None:
        typer.echo(f"Predictions: {_display_path(summary.predictions_path, project_root)}")


def _print_boosted_backtest_summary(
    summary: BoostedBacktestSummary,
    project_root: Path,
) -> None:
    typer.echo("Boosted model backtest complete")
    typer.echo(f"Status: {summary.status}")
    typer.echo(f"Strategy: {summary.strategy}")
    typer.echo(f"Feature policy: {summary.feature_policy}")
    typer.echo(f"Events: {summary.n_events}")
    typer.echo(f"Folds successful: {summary.n_folds_successful}")
    typer.echo(f"Folds failed: {summary.n_folds_failed}")
    typer.echo(f"Prediction rows: {summary.prediction_rows}")
    typer.echo(f"Metrics: {_display_path(summary.metrics_path, project_root)}")
    typer.echo(f"Folds: {_display_path(summary.folds_path, project_root)}")
    if summary.predictions_path is not None:
        typer.echo(f"Predictions: {_display_path(summary.predictions_path, project_root)}")


def _print_champion_backtest_summary(
    summary: ChampionBacktestSummary,
    project_root: Path,
) -> None:
    typer.echo("Champion backtest complete")
    typer.echo(f"Status: {summary.status}")
    typer.echo(f"Strategy: {summary.strategy}")
    typer.echo(f"Selection mode: {summary.selection_mode}")
    typer.echo(f"Events: {summary.n_events}")
    typer.echo(f"Folds successful: {summary.n_folds_successful}")
    typer.echo(f"Folds failed: {summary.n_folds_failed}")
    typer.echo(f"Prediction rows: {summary.prediction_rows}")
    typer.echo(f"Metrics: {_display_path(summary.metrics_path, project_root)}")
    if summary.predictions_path is not None:
        typer.echo(f"Predictions: {_display_path(summary.predictions_path, project_root)}")
    if summary.selection_path is not None:
        typer.echo(f"Selection: {_display_path(summary.selection_path, project_root)}")


def _print_diagnostics_summary(
    summary: DiagnosticsReportSummary,
    project_root: Path,
) -> None:
    typer.echo("Diagnostics report complete")
    typer.echo(f"Preferred source: {summary.preferred_prediction_source}")
    typer.echo(f"Available sources: {', '.join(summary.available_prediction_sources)}")
    typer.echo(f"Events: {summary.n_events}")
    typer.echo(f"Drivers: {summary.n_drivers}")
    typer.echo(f"Report: {_display_path(summary.report_path, project_root)}")
    typer.echo(f"Event summary: {_display_path(summary.event_summary_path, project_root)}")
    typer.echo(f"Driver summary: {_display_path(summary.driver_summary_path, project_root)}")


def _print_portfolio_report_summary(
    summary: PortfolioReportSummary,
    project_root: Path,
) -> None:
    typer.echo("Portfolio report complete")
    typer.echo(f"Summary: {_display_path(summary.summary_path, project_root)}")
    typer.echo(f"Model card: {_display_path(summary.model_card_path, project_root)}")
    typer.echo(f"Tables: {len(summary.table_paths)}")
    typer.echo(f"Figures: {len(summary.figure_paths)}")
    if summary.missing_artifacts:
        typer.echo(f"Missing artifacts: {', '.join(summary.missing_artifacts)}")
    if summary.generation_issues:
        typer.echo(f"Generation issues: {len(summary.generation_issues)}")


def _print_champion_diagnostics_summary(
    summary: ChampionDiagnosticsSummary,
    project_root: Path,
) -> None:
    typer.echo("Champion diagnostics complete")
    typer.echo(f"Status: {summary.status}")
    typer.echo(f"Summary: {_display_path(summary.summary_path, project_root)}")
    typer.echo(f"Tables: {len(summary.table_paths)}")
    typer.echo(f"Figures: {len(summary.figure_paths)}")
    if summary.missing_inputs:
        typer.echo(f"Missing inputs: {', '.join(summary.missing_inputs)}")
    if summary.generation_issues:
        typer.echo(f"Generation issues: {len(summary.generation_issues)}")


def _print_policy_simulation_summary(
    summary: PolicySimulationSummary,
    project_root: Path,
) -> None:
    typer.echo("Policy simulation complete")
    typer.echo(f"Status: {summary.status}")
    typer.echo(f"Summary: {_display_path(summary.summary_path, project_root)}")
    typer.echo(f"Tables: {len(summary.table_paths)}")
    typer.echo(f"Figures: {len(summary.figure_paths)}")
    if summary.missing_inputs:
        typer.echo(f"Missing inputs: {', '.join(summary.missing_inputs)}")
    if summary.generation_issues:
        typer.echo(f"Generation issues: {len(summary.generation_issues)}")


def _print_ablation_summary(
    summary: AblationBacktestSummary,
    project_root: Path,
) -> None:
    typer.echo("Feature ablation backtest complete")
    typer.echo(f"Status: {summary.status}")
    typer.echo(f"Strategy: {summary.strategy}")
    typer.echo(f"Events: {summary.n_events}")
    typer.echo(f"Folds successful: {summary.n_folds_successful}")
    typer.echo(f"Folds failed: {summary.n_folds_failed}")
    typer.echo(f"Feature groups: {', '.join(summary.feature_groups)}")
    typer.echo(f"Metrics: {_display_path(summary.metrics_path, project_root)}")
    typer.echo(f"Groups: {_display_path(summary.feature_groups_path, project_root)}")
    if summary.predictions_path is not None:
        typer.echo(f"Predictions: {_display_path(summary.predictions_path, project_root)}")


if __name__ == "__main__":
    app()
