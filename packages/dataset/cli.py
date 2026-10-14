"""Command-line entry point for the gated dataset collection."""

import argparse
from pathlib import Path

from .collector import collect_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect bounded dataset features; never train.")
    parser.add_argument("--lichess-elite-month", required=True, metavar="YYYY-MM")
    parser.add_argument("--tcec-season", required=True)
    parser.add_argument("--output-dir", default="data/processed", type=Path)
    args = parser.parse_args()
    report = collect_dataset(
        lichess_elite_month=args.lichess_elite_month,
        tcec_season=args.tcec_season,
        output_dir=args.output_dir,
    )
    print("Сбор завершён без обучения.")
    print(f"Human ({report.lichess_elite_month}): {report.class_counts['human']} партий")
    print(f"Engine-like ({report.tcec_season}): {report.class_counts['engine_like']} партий")
    print(f"Скачано: {report.downloaded_bytes} / {report.quota_bytes} bytes")
    print("Обучение не запускалось; ожидается явное подтверждение отчёта.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
