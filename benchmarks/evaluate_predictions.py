"""Evaluate external model predictions with the Tox21Full metric protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from benchmark_utils import (
    bootstrap_macro_ci,
    collect_prediction_records_from_wide,
    label_columns,
    macro_summary,
    metrics_to_frame,
    per_assay_metrics,
    read_table,
    write_json,
    write_predictions,
)


def resolve_row_ids(
    predictions: pd.DataFrame,
    split_df: pd.DataFrame,
    test_input: Path | None,
) -> Sequence[int]:
    """Resolve test row IDs from predictions or their paired input file."""
    if "row_id" in predictions.columns:
        return predictions["row_id"].to_numpy(dtype=int)
    if test_input is not None:
        test_frame = read_table(test_input)
        if "row_id" in test_frame.columns:
            return test_frame["row_id"].to_numpy(dtype=int)
    return split_df.loc[split_df["split"] == "test", "row_id"].to_numpy(dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate wide prediction CSVs against Tox21Full labels."
    )
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument(
        "--test-input",
        type=Path,
        help="CSV used for prediction. Used to recover row_id if needed.",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260506)
    args = parser.parse_args()

    truth = read_table(args.truth)
    split_df = pd.read_csv(args.split_file)
    predictions = read_table(args.predictions)
    row_ids = resolve_row_ids(predictions, split_df, args.test_input)

    if len(row_ids) != len(predictions):
        raise ValueError(
            "Prediction row count does not match resolved test row IDs. "
            "Pass --test-input with a row_id column if the prediction file "
            "does not preserve row identifiers."
        )

    assays = label_columns(truth)
    records = collect_prediction_records_from_wide(
        truth=truth,
        predictions=predictions,
        row_ids=row_ids,
        assays=assays,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(args.out_dir / f"{args.model_name}_test_predictions.csv", records)

    metrics = per_assay_metrics(args.model_name, records, split="test")
    metric_frame = metrics_to_frame(metrics)
    metric_frame.to_csv(args.out_dir / "per_assay_metrics.csv", index=False)

    summary = macro_summary(metrics)
    summary.update(
        bootstrap_macro_ci(
            records,
            test_row_ids=split_df.loc[
                split_df["split"] == "test", "row_id"
            ].to_numpy(dtype=int),
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
    )
    pd.DataFrame([summary]).to_csv(args.out_dir / "summary.csv", index=False)
    write_json(
        args.out_dir / "summary.json",
        {
            "truth": str(args.truth),
            "split_file": str(args.split_file),
            "predictions": str(args.predictions),
            "test_input": str(args.test_input) if args.test_input else None,
            "model": args.model_name,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "summary": summary,
            "assays_with_predictions": len(records),
        },
    )


if __name__ == "__main__":
    main()
