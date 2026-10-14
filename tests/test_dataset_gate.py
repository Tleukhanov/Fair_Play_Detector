"""Tests for the dataset collection gate without network or Parquet dependencies."""

import pytest

from packages.dataset.collector import ByteQuota, DatasetQuotaExceeded, MissingCollectionPeriod
from packages.dataset.contracts import DatasetCollectionReport, DatasetFeatureRow


def test_periods_are_required_before_network_access() -> None:
    from packages.dataset.collector import _require_period

    with pytest.raises(MissingCollectionPeriod):
        _require_period("TCEC_SEASON", None)


def test_quota_stops_at_combined_limit() -> None:
    quota = ByteQuota(limit=10)
    quota.consume(10)
    with pytest.raises(DatasetQuotaExceeded, match="DATASET_QUOTA_EXCEEDED"):
        quota.consume(1)


def test_report_requires_exactly_five_sample_rows_and_no_pgn_column() -> None:
    row = DatasetFeatureRow(dataset_class="human", game_id="g1", move_count=12)
    report = DatasetCollectionReport(
        lichess_elite_month="2025-01",
        tcec_season="2024",
        quota_bytes=5 * 1024**3,
        downloaded_bytes=100,
        parquet_files={},
        sources=[],
        class_counts={"human": 5, "engine_like": 0},
        class_percentages={"human": 100.0, "engine_like": 0.0},
        sample_rows=[row] * 5,
        columns=list(DatasetFeatureRow.model_fields),
    )
    assert len(report.sample_rows) == 5
    assert "pgn" not in report.columns
