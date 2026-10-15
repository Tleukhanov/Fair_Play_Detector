"""Command-line entry point for ML Stage 5."""

from argparse import ArgumentParser
from pathlib import Path

from .training import train_dataset


def main() -> None:
    parser = ArgumentParser(description="Train fair-play detector models from gate Parquet files.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/models/ml_training_report.json"),
    )
    args = parser.parse_args()
    report = train_dataset(args.input_dir, args.report)
    print(f"trained {len(report['models'])} models on {report['row_count']} rows")


if __name__ == "__main__":
    main()
