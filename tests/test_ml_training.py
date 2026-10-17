from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet

from packages.ml.training import train_dataset


def test_training_report_contains_models_and_group_split(tmp_path: Path) -> None:
    rows_by_label = {"human": [], "engine_like": []}
    for label in rows_by_label:
        for user_index in range(8):
            for index in range(10):
                rows_by_label[label].append(
                    {
                        "dataset_class": label,
                        "game_id": f"{label}-{user_index}-{index}",
                        "username": f"{label}-{user_index}",
                        "move_count": 30 + index,
                        "mean_cpl": 20 if label == "human" else 2,
                        "top1_match_rate": 0.7 if label == "human" else 0.99,
                        "top3_match_rate": 0.9,
                        "mean_move_time_ms": 1000,
                        "rated": label == "human",
                        "speed": "rapid",
                        "result": "1-0",
                    }
                )
    for label, rows in rows_by_label.items():
        parquet.write_table(
            pa.Table.from_pylist(rows), tmp_path / f"{label}_features.parquet"
        )
    report = train_dataset(tmp_path, tmp_path / "report.json", speed="rapid")
    assert {model["model_name"] for model in report["models"]} == {
        "logistic_regression",
        "xgboost",
    }
    assert sum(report["split_counts"].values()) == 160
    assert "dataset_hash" in report


def test_training_rejects_missing_behavioral_features(tmp_path: Path) -> None:
    rows = [
        {
            "dataset_class": label,
            "game_id": f"{label}-1",
            "username": f"{label}-user",
            "move_count": 40,
            "rated": label == "human",
            "speed": "rapid",
            "result": "1-0",
        }
        for label in ("human", "engine_like")
    ]
    table = pa.Table.from_pylist(rows)
    parquet.write_table(table, tmp_path / "human_features.parquet")
    parquet.write_table(table, tmp_path / "engine_features.parquet")
    import pytest

    with pytest.raises(ValueError, match="behavioral features are incomplete"):
        train_dataset(tmp_path, tmp_path / "report.json", speed="rapid")
