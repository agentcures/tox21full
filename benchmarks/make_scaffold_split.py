"""Create a deterministic Bemis-Murcko scaffold split for Tox21Full."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from benchmark_utils import SPLIT_ORDER, label_columns, read_table, write_json
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def scaffold_key(smiles: str, row_id: int) -> tuple[str, str]:
    """Return a scaffold grouping key and a status string."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return f"invalid:{row_id}", "invalid_smiles"

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if scaffold:
        return f"scaffold:{scaffold}", "murcko_scaffold"

    canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
    return f"acyclic:{canonical}", "acyclic_canonical_smiles"


def assign_scaffold_groups(
    groups: Sequence[tuple[str, list[int]]],
    n_rows: int,
    fractions: tuple[float, float, float],
) -> dict[int, str]:
    """Assign scaffold groups to train/valid/test without splitting groups."""
    train_target = int(round(n_rows * fractions[0]))
    valid_target = int(round(n_rows * fractions[1]))
    assignments: dict[int, str] = {}
    counts = dict.fromkeys(SPLIT_ORDER, 0)

    for _, row_ids in groups:
        size = len(row_ids)
        if counts["train"] + size <= train_target or counts["train"] == 0:
            split = "train"
        elif counts["valid"] + size <= valid_target or counts["valid"] == 0:
            split = "valid"
        else:
            split = "test"
        for row_id in row_ids:
            assignments[row_id] = split
        counts[split] += size

    return assignments


def make_split(df: pd.DataFrame, fractions: tuple[float, float, float]) -> pd.DataFrame:
    """Build split assignments for a dataframe with a smiles column."""
    scaffold_rows: dict[str, list[int]] = defaultdict(list)
    scaffold_status: dict[str, str] = {}
    smiles_values = df["smiles"].fillna("").astype(str).tolist()

    for row_id, smiles in enumerate(smiles_values):
        key, status = scaffold_key(smiles, row_id)
        scaffold_rows[key].append(row_id)
        scaffold_status[key] = status

    groups = sorted(scaffold_rows.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    assignments = assign_scaffold_groups(groups, len(df), fractions)

    split_rows = []
    for key, row_ids in groups:
        for row_id in row_ids:
            split_rows.append(
                {
                    "row_id": row_id,
                    "smiles": smiles_values[row_id],
                    "scaffold": key,
                    "scaffold_status": scaffold_status[key],
                    "split": assignments[row_id],
                }
            )

    split_df = pd.DataFrame(split_rows).sort_values("row_id")
    return split_df.reset_index(drop=True)


def parse_fractions(value: str) -> tuple[float, float, float]:
    parts = tuple(float(v) for v in value.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values")
    if abs(sum(parts) - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError("Split fractions must sum to 1.0")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic scaffold split for Tox21Full."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--fractions",
        default="0.8,0.1,0.1",
        type=parse_fractions,
        help="Train,valid,test fractions. Default: 0.8,0.1,0.1",
    )
    parser.add_argument(
        "--write-split-csvs",
        action="store_true",
        help="Also write train.csv, valid.csv, and test.csv subsets.",
    )
    args = parser.parse_args()

    df = read_table(args.data)
    label_columns(df)
    split_df = make_split(df, args.fractions)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.out_dir / "scaffold_split.csv"
    split_df.to_csv(split_path, index=False)

    if args.write_split_csvs:
        for split in SPLIT_ORDER:
            row_ids = split_df.loc[split_df["split"] == split, "row_id"]
            df.iloc[row_ids.to_numpy(dtype=int)].to_csv(
                args.out_dir / f"{split}.csv", index=False
            )

    counts = split_df["split"].value_counts().to_dict()
    status_counts = split_df["scaffold_status"].value_counts().to_dict()
    scaffold_counts = (
        split_df.drop_duplicates("scaffold")["scaffold_status"].value_counts().to_dict()
    )
    write_json(
        args.out_dir / "scaffold_split_summary.json",
        {
            "data": str(args.data),
            "fractions": {
                "train": args.fractions[0],
                "valid": args.fractions[1],
                "test": args.fractions[2],
            },
            "row_counts": {split: int(counts.get(split, 0)) for split in SPLIT_ORDER},
            "row_status_counts": {
                key: int(value) for key, value in sorted(status_counts.items())
            },
            "scaffold_status_counts": {
                key: int(value) for key, value in sorted(scaffold_counts.items())
            },
            "unique_scaffolds": int(split_df["scaffold"].nunique()),
            "split_file": str(split_path),
        },
    )


if __name__ == "__main__":
    main()
