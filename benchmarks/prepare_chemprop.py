"""Prepare scaffold-split Tox21Full CSV files for Chemprop v2."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import pandas as pd
from benchmark_utils import SPLIT_ORDER, label_columns, read_table, write_json
from make_scaffold_split import make_split, parse_fractions


def shell_join(args: list[str]) -> str:
    return shlex.join(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Tox21Full files and commands for Chemprop."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--split-file",
        type=Path,
        help="Existing scaffold_split.csv. If omitted, one is generated.",
    )
    parser.add_argument(
        "--fractions",
        default="0.8,0.1,0.1",
        type=parse_fractions,
        help="Train,valid,test fractions used when generating a split.",
    )
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ensemble-size", type=int, default=1)
    args = parser.parse_args()

    df = read_table(args.data)
    assays = label_columns(df)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.split_file:
        split_df = pd.read_csv(args.split_file)
    else:
        split_df = make_split(df, args.fractions)
        split_df.to_csv(args.out_dir / "scaffold_split.csv", index=False)

    chemprop_df = df.copy()
    chemprop_df.insert(0, "row_id", range(len(chemprop_df)))

    split_paths = {}
    for split in SPLIT_ORDER:
        row_ids = split_df.loc[split_df["split"] == split, "row_id"].to_numpy(dtype=int)
        split_frame = chemprop_df.iloc[row_ids]
        path = args.out_dir / f"chemprop_{split}.csv"
        split_frame.to_csv(path, index=False)
        split_paths[split] = path

    test_smiles = chemprop_df.iloc[
        split_df.loc[split_df["split"] == "test", "row_id"].to_numpy(dtype=int)
    ][["row_id", "smiles"]]
    test_smiles_path = args.out_dir / "chemprop_test_smiles.csv"
    test_smiles.to_csv(test_smiles_path, index=False)

    targets_path = args.out_dir / "chemprop_targets.txt"
    targets_path.write_text("\n".join(assays) + "\n")

    model_dir = args.out_dir / "chemprop_mpnn"
    prediction_path = args.out_dir / "chemprop_test_predictions.csv"

    train_cmd = [
        "chemprop",
        "train",
        "--data-path",
        str(split_paths["train"]),
        str(split_paths["valid"]),
        str(split_paths["test"]),
        "--task-type",
        "classification",
        "--smiles-columns",
        "smiles",
        "--target-columns",
        *assays,
        "--ignore-columns",
        "row_id",
        "--class-balance",
        "--metrics",
        "roc",
        "prc",
        "--tracking-metric",
        "prc",
        "--show-individual-scores",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--ensemble-size",
        str(args.ensemble_size),
        "--pytorch-seed",
        str(args.seed),
        "--output-dir",
        str(model_dir),
    ]
    predict_cmd = [
        "chemprop",
        "predict",
        "--test-path",
        str(test_smiles_path),
        "--model-paths",
        str(model_dir),
        "--output",
        str(prediction_path),
        "--smiles-columns",
        "smiles",
    ]
    eval_cmd = [
        "python",
        "benchmarks/evaluate_predictions.py",
        "--truth",
        str(args.data),
        "--split-file",
        str(args.split_file or args.out_dir / "scaffold_split.csv"),
        "--predictions",
        str(prediction_path),
        "--test-input",
        str(test_smiles_path),
        "--model-name",
        "chemprop_mpnn",
        "--out-dir",
        str(args.out_dir / "chemprop_metrics"),
        "--bootstrap",
        "1000",
        "--seed",
        str(args.seed),
    ]

    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Train the multitask directed message-passing neural network.",
        shell_join(train_cmd),
        "",
        "# Export test-set probabilities for the fixed scaffold split.",
        shell_join(predict_cmd),
        "",
        "# Convert Chemprop predictions to the shared Tox21Full metrics format.",
        shell_join(eval_cmd),
        "",
    ]
    command_path = args.out_dir / "run_chemprop.sh"
    command_path.write_text("\n".join(commands))
    command_path.chmod(0o755)

    write_json(
        args.out_dir / "chemprop_prep_summary.json",
        {
            "data": str(args.data),
            "split_file": str(args.split_file or args.out_dir / "scaffold_split.csv"),
            "target_count": len(assays),
            "targets_file": str(targets_path),
            "split_paths": {k: str(v) for k, v in split_paths.items()},
            "test_smiles_path": str(test_smiles_path),
            "commands_file": str(command_path),
            "chemprop_model_dir": str(model_dir),
            "chemprop_prediction_path": str(prediction_path),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "ensemble_size": args.ensemble_size,
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
