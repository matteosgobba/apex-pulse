"""Read-only reproduction of the 2024/2025 fixed candidates and input contract.

Run with the project's Python environment; writes no datasets or reports.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from f1_prediction.config import load_data_config, load_feature_config, load_model_config
from f1_prediction.features.build import build_session_features_output_path
from f1_prediction.modeling.fp3_production_policy import audited_feature_rows, load_policy
from f1_prediction.modeling.prospective_monitoring import train_monitoring_event_sources
from f1_prediction.modeling.prospective_replay import event_key_series
from f1_prediction.modeling.splits import ordered_event_keys


def main():
    root = Path(__file__).resolve().parents[1]
    config = load_data_config(project_root=root)
    spec = load_policy(config)
    columns = spec["candidate_contract"]["feature_columns"]
    dataset = pd.read_parquet(config.modeling_output_dir / "combined/modeling_dataset.parquet")
    fp3 = dataset[dataset.checkpoint.eq("after_fp3")].copy()
    for (season, event), old in fp3.groupby(["season", "event"]):
        practice = pd.read_parquet(
            build_session_features_output_path(
                config.session_features_output_dir, int(season), event
            )
        )
        new = (
            audited_feature_rows(practice, columns).set_index("driver_key").reindex(old.driver_key)
        )
        np.testing.assert_allclose(
            old[columns].to_numpy(dtype=float),
            new[columns].to_numpy(dtype=float),
            equal_nan=True,
            atol=0,
            rtol=0,
        )
    order = ordered_event_keys(dataset)
    references = {}
    for name, temporal in [("static", "uniform"), ("weighted", "current_season_only_with_prior")]:
        references[name] = pd.read_parquet(
            config.metrics_output_dir / f"ablation_{temporal}_predictions.parquet"
        ).query(
            "checkpoint == 'after_fp3' and model_name == 'random_forest' "
            "and feature_group == 'base_plus_relative'"
        )
    comparisons = []
    for event in [k for k in order if k[:4] in {"2024", "2025"}]:
        source = train_monitoring_event_sources(
            dataset=fp3,
            row_keys=event_key_series(fp3),
            event_order=order,
            event_key=event,
            legal_train_events=order[: order.index(event)],
            model_config=load_model_config(),
            feature_config=load_feature_config(),
            test_season=int(event[:4]),
            frozen_feature_columns=columns,
        )
        for name in references:
            joined = source[name].merge(
                references[name],
                on=["season", "event_slug", "checkpoint", "driver"],
                suffixes=("_new", "_ref"),
                validate="one_to_one",
            )
            np.testing.assert_array_equal(
                joined.predicted_quali_gap_to_pole_sec_new,
                joined.predicted_quali_gap_to_pole_sec_ref,
            )
            comparisons.append(
                {
                    "event": event,
                    "candidate": name,
                    "rows": len(joined),
                    "max_prediction_difference": 0.0,
                }
            )
    metrics = []
    for season in (2024, 2025):
        for name, data in references.items():
            data = data[data.season.eq(season)]
            metrics.append(
                {
                    "season": season,
                    "candidate": name,
                    "rows": len(data),
                    "events": data.event_slug.nunique(),
                    "mae": float(
                        (data.predicted_quali_gap_to_pole_sec - data.quali_gap_to_pole_sec)
                        .abs()
                        .mean()
                    ),
                }
            )
    print(
        json.dumps(
            {
                "adapter_events": fp3[["season", "event_slug"]].drop_duplicates().shape[0],
                "features": len(columns),
                "metrics": metrics,
                "refits": comparisons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
