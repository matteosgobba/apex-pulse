from pathlib import Path

from typer.testing import CliRunner

from f1_prediction.cli import app
from f1_prediction.config import DataConfig, FeatureConfig, PushLapConfig
from f1_prediction.data.ingest import EventIngestionSummary
from f1_prediction.data.season_builder import SeasonDatasetBuildSummary
from f1_prediction.features.build import SessionFeatureBuildSummary
from f1_prediction.features.modeling_dataset import ModelingDatasetBuildSummary
from f1_prediction.modeling.champion_diagnostics import ChampionDiagnosticsSummary
from f1_prediction.modeling.champion_policy import ChampionBacktestSummary
from f1_prediction.modeling.evaluate_baselines import BaselineEvaluationSummary
from f1_prediction.modeling.policy_simulation import PolicySimulationSummary
from f1_prediction.modeling.portfolio_report import PortfolioReportSummary


def test_ingest_event_accepts_sessions_after_single_option(monkeypatch, tmp_path: Path) -> None:
    captured_sessions: list[str] = []
    config = DataConfig(
        project_root=tmp_path,
        fastf1_cache_dir=tmp_path / "cache",
        lap_output_dir=tmp_path / "laps",
        session_metadata_output_dir=tmp_path / "metadata",
        clean_lap_output_dir=tmp_path / "clean_laps",
        session_features_output_dir=tmp_path / "session_features",
        modeling_output_dir=tmp_path / "modeling",
        metrics_output_dir=tmp_path / "metrics",
    )

    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)

    def fake_ingestion(*, season, event, config, sessions, **kwargs):
        captured_sessions.extend(sessions)
        return EventIngestionSummary(season=season, event=event, results=())

    monkeypatch.setattr("f1_prediction.cli.run_event_ingestion", fake_ingestion)

    result = CliRunner().invoke(
        app,
        [
            "ingest-event",
            "--season",
            "2024",
            "--event",
            "Monza",
            "--sessions",
            "FP1",
            "FP2",
            "Q",
        ],
    )

    assert result.exit_code == 0
    assert captured_sessions == ["FP1", "FP2", "Q"]


def test_build_session_features_accepts_session_subset(monkeypatch, tmp_path: Path) -> None:
    captured_sessions: list[str] = []
    config = DataConfig(
        project_root=tmp_path,
        fastf1_cache_dir=tmp_path / "cache",
        lap_output_dir=tmp_path / "laps",
        session_metadata_output_dir=tmp_path / "metadata",
        clean_lap_output_dir=tmp_path / "clean_laps",
        session_features_output_dir=tmp_path / "session_features",
        modeling_output_dir=tmp_path / "modeling",
        metrics_output_dir=tmp_path / "metrics",
    )
    feature_config = FeatureConfig(push_lap=PushLapConfig(1.03, 1.07, ("SOFT", "MEDIUM", "HARD")))
    output_path = tmp_path / "session_features/2024/monza/practice_session_features.parquet"

    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr(
        "f1_prediction.cli.load_feature_config",
        lambda config_path=None, project_root=None: feature_config,
    )

    def fake_build(*, season, event, sessions, **kwargs):
        captured_sessions.extend(sessions)
        return SessionFeatureBuildSummary(
            season=season,
            event=event,
            sessions=tuple(sessions),
            clean_lap_paths=(),
            clean_lap_files_written=1,
            aggregate_rows=20,
            output_path=output_path,
        )

    monkeypatch.setattr("f1_prediction.cli.run_feature_build", fake_build)

    result = CliRunner().invoke(
        app,
        [
            "build-session-features",
            "--season",
            "2024",
            "--event",
            "Monza",
            "--sessions",
            "FP2",
        ],
    )

    assert result.exit_code == 0
    assert captured_sessions == ["FP2"]
    assert "Aggregate rows: 20" in result.output


def test_build_modeling_dataset_command(monkeypatch, tmp_path: Path) -> None:
    config = DataConfig(
        project_root=tmp_path,
        fastf1_cache_dir=tmp_path / "cache",
        lap_output_dir=tmp_path / "laps",
        session_metadata_output_dir=tmp_path / "metadata",
        clean_lap_output_dir=tmp_path / "clean_laps",
        session_features_output_dir=tmp_path / "session_features",
        modeling_output_dir=tmp_path / "modeling",
        metrics_output_dir=tmp_path / "metrics",
    )
    output_path = tmp_path / "modeling/2024/monza/modeling_dataset.parquet"
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr(
        "f1_prediction.cli.load_feature_config",
        lambda config_path=None, project_root=None: FeatureConfig(
            push_lap=PushLapConfig(1.03, 1.07, ("SOFT",))
        ),
    )
    monkeypatch.setattr(
        "f1_prediction.cli.run_modeling_dataset_build",
        lambda **kwargs: ModelingDatasetBuildSummary(
            season=2024,
            event="Monza",
            rows=60,
            drivers=20,
            checkpoints=("after_fp1", "after_fp2", "after_fp3"),
            qualifying_only_drivers=(),
            practice_only_drivers=("ANT",),
            output_path=output_path,
        ),
    )

    result = CliRunner().invoke(
        app,
        ["build-modeling-dataset", "--season", "2024", "--event", "Monza"],
    )

    assert result.exit_code == 0
    assert "Rows: 60" in result.output
    assert "Practice-only drivers: ANT" in result.output


def test_build_season_dataset_accepts_repeated_seasons_and_event_filter(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    feature_config = FeatureConfig(push_lap=PushLapConfig(1.03, 1.07, ("SOFT",)))
    captured: dict[str, object] = {}
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr(
        "f1_prediction.cli.load_feature_config",
        lambda config_path=None, project_root=None: feature_config,
    )

    def fake_build(**kwargs):
        captured.update(kwargs)
        return SeasonDatasetBuildSummary(
            requested_seasons=(2023, 2024),
            n_events_requested=2,
            successful_events=(),
            failed_events=(),
            n_rows=120,
            n_drivers=22,
            checkpoints=("after_fp1", "after_fp2", "after_fp3"),
            output_path=tmp_path / "modeling/combined/modeling_dataset.parquet",
            report_path=tmp_path / "metrics/dataset_build_report.json",
        )

    monkeypatch.setattr("f1_prediction.cli.run_season_dataset_build", fake_build)
    result = CliRunner().invoke(
        app,
        [
            "build-season-dataset",
            "--season",
            "2023",
            "--season",
            "2024",
            "--events",
            "Monza",
        ],
    )

    assert result.exit_code == 0
    assert captured["seasons"] == [2023, 2024]
    assert captured["events"] == ("Monza",)


def test_build_season_dataset_accepts_preset(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    feature_config = FeatureConfig(push_lap=PushLapConfig(1.03, 1.07, ("SOFT",)))
    captured: dict[str, object] = {}
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr(
        "f1_prediction.cli.load_feature_config",
        lambda config_path=None, project_root=None: feature_config,
    )

    def fake_build(**kwargs):
        captured.update(kwargs)
        return SeasonDatasetBuildSummary(
            requested_seasons=(2024,),
            n_events_requested=12,
            successful_events=(),
            failed_events=(),
            n_rows=0,
            n_drivers=0,
            checkpoints=("after_fp1", "after_fp2", "after_fp3"),
            output_path=tmp_path / "modeling/combined/modeling_dataset.parquet",
            report_path=tmp_path / "metrics/dataset_build_report.json",
        )

    monkeypatch.setattr("f1_prediction.cli.run_season_dataset_build", fake_build)
    result = CliRunner().invoke(
        app,
        ["build-season-dataset", "--season", "2024", "--preset", "conventional_2024"],
    )

    assert result.exit_code == 0
    assert "Bahrain" in captured["events"]
    assert "Abu Dhabi" in captured["events"]


def test_build_season_dataset_accepts_multi_season_alias(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    feature_config = FeatureConfig(push_lap=PushLapConfig(1.03, 1.07, ("SOFT",)))
    captured: dict[str, object] = {}
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr(
        "f1_prediction.cli.load_feature_config",
        lambda config_path=None, project_root=None: feature_config,
    )

    def fake_build(**kwargs):
        captured.update(kwargs)
        return SeasonDatasetBuildSummary(
            requested_seasons=(2023, 2024),
            n_events_requested=26,
            successful_events=(),
            failed_events=(),
            n_rows=0,
            n_drivers=0,
            checkpoints=("after_fp1", "after_fp2", "after_fp3"),
            output_path=tmp_path / "modeling/combined/modeling_dataset.parquet",
            report_path=tmp_path / "metrics/dataset_build_report.json",
            preset="conventional",
        )

    monkeypatch.setattr("f1_prediction.cli.run_season_dataset_build", fake_build)
    result = CliRunner().invoke(
        app,
        ["build-season-dataset", "--seasons", "2023", "2024", "--preset", "conventional"],
    )

    assert result.exit_code == 0
    assert captured["seasons"] == [2023, 2024]
    assert set(captured["events"]) == {2023, 2024}
    assert captured["preset"] == "conventional"


def test_build_season_dataset_accepts_multiple_events_after_one_option(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    feature_config = FeatureConfig(push_lap=PushLapConfig(1.03, 1.07, ("SOFT",)))
    captured: dict[str, object] = {}
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr(
        "f1_prediction.cli.load_feature_config",
        lambda config_path=None, project_root=None: feature_config,
    )

    def fake_build(**kwargs):
        captured.update(kwargs)
        return SeasonDatasetBuildSummary(
            requested_seasons=(2024,),
            n_events_requested=3,
            successful_events=(),
            failed_events=(),
            n_rows=0,
            n_drivers=0,
            checkpoints=("after_fp1", "after_fp2", "after_fp3"),
            output_path=tmp_path / "modeling/combined/modeling_dataset.parquet",
            report_path=tmp_path / "metrics/dataset_build_report.json",
        )

    monkeypatch.setattr("f1_prediction.cli.run_season_dataset_build", fake_build)
    result = CliRunner().invoke(
        app,
        [
            "build-season-dataset",
            "--season",
            "2024",
            "--events",
            "Bahrain",
            "Australia",
            "Japan",
        ],
    )

    assert result.exit_code == 0
    assert captured["events"] == ("Bahrain", "Australia", "Japan")


def test_evaluate_baselines_command(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr(
        "f1_prediction.cli.load_feature_config",
        lambda config_path=None, project_root=None: FeatureConfig(
            push_lap=PushLapConfig(1.03, 1.07, ("SOFT",))
        ),
    )
    monkeypatch.setattr(
        "f1_prediction.cli.run_baseline_evaluation",
        lambda config, dataset_path=None, feature_config=None: BaselineEvaluationSummary(
            dataset_path=tmp_path / "combined.parquet",
            prediction_rows=180,
            baselines=("best_push_lap",),
            checkpoints=("after_fp1", "after_fp2", "after_fp3"),
            metrics_path=tmp_path / "metrics/baseline_metrics.json",
            predictions_path=tmp_path / "metrics/baseline_predictions.parquet",
        ),
    )

    result = CliRunner().invoke(app, ["evaluate-baselines"])

    assert result.exit_code == 0
    assert "Prediction rows: 180" in result.output


def test_champion_backtest_accepts_stabilized_nested_and_conformal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr("f1_prediction.cli.load_model_config", lambda **kwargs: object())

    def fake_champion_backtest(*args, **kwargs):
        captured.update(kwargs)
        return ChampionBacktestSummary(
            status="complete",
            strategy="walk_forward",
            selection_mode=str(kwargs["selection_mode"].value),
            n_events=10,
            n_folds_total=5,
            n_folds_successful=5,
            n_folds_failed=0,
            prediction_rows=300,
            metrics_path=tmp_path / "metrics/champion_metrics.json",
            predictions_path=tmp_path / "metrics/champion_predictions.parquet",
            selection_path=tmp_path / "metrics/champion_selection.parquet",
        )

    monkeypatch.setattr("f1_prediction.cli.run_champion_backtest", fake_champion_backtest)
    result = CliRunner().invoke(
        app,
        [
            "champion-backtest",
            "--selection-mode",
            "stabilized_nested",
            "--uncertainty",
            "conformal",
        ],
    )

    assert result.exit_code == 0
    assert captured["selection_mode"].value == "stabilized_nested"
    assert captured["uncertainty_method"].value == "conformal"
    assert "Selection mode: stabilized_nested" in result.output


def test_champion_backtest_accepts_stabilized_nested_guarded(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr("f1_prediction.cli.load_model_config", lambda **kwargs: object())

    def fake_champion_backtest(*args, **kwargs):
        captured.update(kwargs)
        return ChampionBacktestSummary(
            status="complete",
            strategy="walk_forward",
            selection_mode=str(kwargs["selection_mode"].value),
            n_events=10,
            n_folds_total=5,
            n_folds_successful=5,
            n_folds_failed=0,
            prediction_rows=300,
            metrics_path=tmp_path / "metrics/champion_metrics.json",
            predictions_path=tmp_path / "metrics/champion_predictions.parquet",
            selection_path=tmp_path / "metrics/champion_selection.parquet",
        )

    monkeypatch.setattr("f1_prediction.cli.run_champion_backtest", fake_champion_backtest)
    result = CliRunner().invoke(
        app,
        [
            "champion-backtest",
            "--selection-mode",
            "stabilized_nested_guarded",
            "--uncertainty",
            "conformal",
        ],
    )

    assert result.exit_code == 0
    assert captured["selection_mode"].value == "stabilized_nested_guarded"
    assert captured["uncertainty_method"].value == "conformal"
    assert "Selection mode: stabilized_nested_guarded" in result.output


def test_champion_backtest_accepts_predicted_gap_bucket_uncertainty(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr("f1_prediction.cli.load_model_config", lambda **kwargs: object())

    def fake_champion_backtest(*args, **kwargs):
        captured.update(kwargs)
        return ChampionBacktestSummary(
            status="complete",
            strategy="walk_forward",
            selection_mode=str(kwargs["selection_mode"].value),
            n_events=10,
            n_folds_total=5,
            n_folds_successful=5,
            n_folds_failed=0,
            prediction_rows=300,
            metrics_path=tmp_path / "metrics/champion_metrics.json",
            predictions_path=tmp_path / "metrics/champion_predictions.parquet",
            selection_path=tmp_path / "metrics/champion_selection.parquet",
        )

    monkeypatch.setattr("f1_prediction.cli.run_champion_backtest", fake_champion_backtest)
    result = CliRunner().invoke(
        app,
        [
            "champion-backtest",
            "--selection-mode",
            "stabilized_nested_guarded",
            "--uncertainty",
            "conformal_predicted_gap_bucket",
        ],
    )

    assert result.exit_code == 0
    assert captured["uncertainty_method"].value == "conformal_predicted_gap_bucket"
    assert "Selection mode: stabilized_nested_guarded" in result.output


def test_portfolio_report_command_is_registered(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr(
        "f1_prediction.cli.run_portfolio_report",
        lambda data_config: PortfolioReportSummary(
            summary_path=tmp_path / "reports/metrics/portfolio_summary.json",
            model_card_path=tmp_path / "reports/model_card.md",
            table_paths=(tmp_path / "reports/metrics/champion_summary_table.csv",),
            figure_paths=(tmp_path / "reports/figures/champion_mae_by_checkpoint.png",),
            missing_artifacts=("champion_nested_metrics.json",),
            generation_issues=(),
        ),
    )

    result = CliRunner().invoke(app, ["portfolio-report"])

    assert result.exit_code == 0
    assert "Portfolio report complete" in result.output
    assert "portfolio_summary.json" in result.output


def test_champion_diagnostics_command_is_registered(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr(
        "f1_prediction.cli.load_model_config",
        lambda config_path=None, project_root=None: type(
            "ModelConfigStub",
            (),
            {"champion_diagnostics": object()},
        )(),
    )
    monkeypatch.setattr(
        "f1_prediction.cli.run_champion_diagnostics_report",
        lambda data_config, diagnostics_config: ChampionDiagnosticsSummary(
            status="partial",
            summary_path=tmp_path / "reports/metrics/champion_diagnostics_summary.json",
            table_paths=(tmp_path / "reports/metrics/champion_harmful_switches.csv",),
            figure_paths=(tmp_path / "reports/figures/harmful_switch_delta_by_checkpoint.png",),
            missing_inputs=("champion_nested_predictions.parquet",),
            generation_issues=(),
        ),
    )

    result = CliRunner().invoke(app, ["champion-diagnostics"])

    assert result.exit_code == 0
    assert "Champion diagnostics complete" in result.output
    assert "champion_diagnostics_summary.json" in result.output


def test_policy_simulation_command_is_registered(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr("f1_prediction.cli.load_data_config", lambda config_path=None: config)
    monkeypatch.setattr(
        "f1_prediction.cli.load_model_config",
        lambda config_path=None, project_root=None: type(
            "ModelConfigStub",
            (),
            {"policy_simulation": object()},
        )(),
    )
    monkeypatch.setattr(
        "f1_prediction.cli.run_policy_simulation_report",
        lambda data_config, simulation_config: PolicySimulationSummary(
            status="partial",
            summary_path=tmp_path / "reports/metrics/policy_simulation_summary.json",
            table_paths=(tmp_path / "reports/metrics/fp3_guardrail_simulation_table.csv",),
            figure_paths=(tmp_path / "reports/figures/fp3_guardrail_policy_mae.png",),
            missing_inputs=("champion_static_predictions.parquet",),
            generation_issues=(),
        ),
    )

    result = CliRunner().invoke(app, ["policy-simulation"])

    assert result.exit_code == 0
    assert "Policy simulation complete" in result.output
    assert "policy_simulation_summary.json" in result.output


def _config(project_root: Path) -> DataConfig:
    return DataConfig(
        project_root=project_root,
        fastf1_cache_dir=project_root / "cache",
        lap_output_dir=project_root / "laps",
        session_metadata_output_dir=project_root / "metadata",
        clean_lap_output_dir=project_root / "clean_laps",
        session_features_output_dir=project_root / "session_features",
        modeling_output_dir=project_root / "modeling",
        metrics_output_dir=project_root / "metrics",
    )
