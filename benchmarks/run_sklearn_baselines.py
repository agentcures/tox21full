"""Run classical Tox21Full baselines under a fixed scaffold split."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from benchmark_utils import (
    SPLIT_ORDER,
    bootstrap_macro_ci,
    label_columns,
    macro_summary,
    metrics_to_frame,
    per_assay_metrics,
    read_table,
    write_json,
    write_predictions,
)
from make_scaffold_split import make_split, parse_fractions


MODEL_CHOICES = ("ecfp_logreg", "ecfp_rf", "rdkit_hgb")


def molecules_from_smiles(smiles: Sequence[str]) -> List[Chem.Mol]:
    """Parse SMILES strings; invalid molecules are represented as None."""
    return [Chem.MolFromSmiles(str(value)) for value in smiles]


def ecfp_features(
    mols: Sequence[Chem.Mol], radius: int, n_bits: int
) -> np.ndarray:
    """Compute dense ECFP/Morgan bit vectors."""
    features = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, mol in enumerate(tqdm(mols, desc="[Features] ECFP", unit="mol")):
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=n_bits
        )
        DataStructs.ConvertToNumpyArray(fp, features[i])
    return features


def descriptor_features(mols: Sequence[Chem.Mol]) -> pd.DataFrame:
    """Compute RDKit 2D descriptor features."""
    descriptor_fns = Descriptors._descList
    rows: List[List[float]] = []
    for mol in tqdm(mols, desc="[Features] RDKit descriptors", unit="mol"):
        if mol is None:
            rows.append([np.nan] * len(descriptor_fns))
            continue
        values: List[float] = []
        for _, fn in descriptor_fns:
            try:
                value = float(fn(mol))
            except Exception:
                value = np.nan
            if not np.isfinite(value):
                value = np.nan
            values.append(value)
        rows.append(values)
    columns = [name for name, _ in descriptor_fns]
    return pd.DataFrame(rows, columns=columns)


def make_model(model_name: str, seed: int):
    """Construct a fixed-hyperparameter baseline model."""
    if model_name == "ecfp_logreg":
        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
                solver="liblinear",
            ),
        )
    if model_name == "ecfp_rf":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "rdkit_hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                early_stopping=True,
                learning_rate=0.05,
                l2_regularization=0.0,
                max_iter=300,
                random_state=seed,
            ),
        )
    raise ValueError(f"Unknown model: {model_name}")


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    """Return sample weights inversely proportional to class frequency."""
    y = y.astype(int)
    n = y.size
    counts = np.bincount(y, minlength=2).astype(float)
    weights = np.ones(n, dtype=float)
    for klass in (0, 1):
        if counts[klass] > 0:
            weights[y == klass] = n / (2.0 * counts[klass])
    return weights


def fit_model(model, model_name: str, x_train: np.ndarray, y_train: np.ndarray):
    """Fit a model, passing sample weights where the estimator supports them."""
    if model_name == "rdkit_hgb":
        model.fit(
            x_train,
            y_train,
            histgradientboostingclassifier__sample_weight=class_balanced_weights(
                y_train
            ),
        )
    else:
        model.fit(x_train, y_train)
    return model


def predict_positive_probability(model, x_test: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities for a fitted sklearn classifier."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        if proba.shape[1] == 1:
            return np.zeros(x_test.shape[0], dtype=float)
        return proba[:, 1]
    decision = model.decision_function(x_test)
    return 1.0 / (1.0 + np.exp(-decision))


def run_model(
    model_name: str,
    features: np.ndarray,
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    assays: Sequence[str],
    seed: int,
) -> Dict[str, pd.DataFrame]:
    """Train one-vs-rest assay classifiers and return test predictions."""
    split_values = split_df["split"].to_numpy()
    train_rows = split_values == "train"
    test_rows = split_values == "test"
    prediction_records: Dict[str, pd.DataFrame] = {}

    for assay in tqdm(assays, desc=f"[Train] {model_name}", unit="assay"):
        observed_train = train_rows & df[assay].notna().to_numpy()
        observed_test = test_rows & df[assay].notna().to_numpy()
        y_train = df.loc[observed_train, assay].to_numpy(dtype=int)

        if y_train.size == 0 or np.unique(y_train).size < 2:
            continue
        if observed_test.sum() == 0:
            continue

        model = make_model(model_name, seed)
        model = fit_model(model, model_name, features[observed_train], y_train)
        y_score = predict_positive_probability(model, features[observed_test])
        row_ids = np.flatnonzero(observed_test)

        prediction_records[assay] = pd.DataFrame(
            {
                "row_id": row_ids,
                "smiles": df.loc[observed_test, "smiles"].to_numpy(),
                "y_true": df.loc[observed_test, assay].to_numpy(dtype=float),
                "y_score": y_score,
            }
        )

    return prediction_records


def build_features_for_models(
    models: Iterable[str],
    smiles: Sequence[str],
    ecfp_radius: int,
    ecfp_bits: int,
) -> Mapping[str, np.ndarray]:
    """Compute only the feature matrices required by requested models."""
    models = tuple(models)
    mols = molecules_from_smiles(smiles)
    features: Dict[str, np.ndarray] = {}

    if any(model.startswith("ecfp_") for model in models):
        ecfp = ecfp_features(mols, radius=ecfp_radius, n_bits=ecfp_bits)
        for model in models:
            if model.startswith("ecfp_"):
                features[model] = ecfp

    if "rdkit_hgb" in models:
        descriptors = descriptor_features(mols)
        features["rdkit_hgb"] = descriptors.to_numpy(dtype=np.float32)

    return features


def parse_models(value: str) -> List[str]:
    models = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [model for model in models if model not in MODEL_CHOICES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown model(s): {', '.join(invalid)}"
        )
    return models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scaffold-split sklearn baselines for Tox21Full."
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
    parser.add_argument(
        "--models",
        default="ecfp_logreg,ecfp_rf,rdkit_hgb",
        type=parse_models,
        help=f"Comma-separated subset of: {', '.join(MODEL_CHOICES)}",
    )
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--ecfp-radius", type=int, default=2)
    parser.add_argument("--ecfp-bits", type=int, default=2048)
    args = parser.parse_args()

    started = time.time()
    df = read_table(args.data)
    assays = label_columns(df)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.split_file:
        split_df = pd.read_csv(args.split_file)
    else:
        split_df = make_split(df, args.fractions)
        split_df.to_csv(args.out_dir / "scaffold_split.csv", index=False)

    if len(split_df) != len(df):
        raise ValueError("Split file row count does not match dataset row count")
    if list(split_df["row_id"]) != list(range(len(df))):
        raise ValueError("Split file must contain row_id values in dataset order")

    features_by_model = build_features_for_models(
        args.models,
        df["smiles"].fillna("").astype(str).tolist(),
        ecfp_radius=args.ecfp_radius,
        ecfp_bits=args.ecfp_bits,
    )

    all_metric_frames: List[pd.DataFrame] = []
    all_summaries: List[Dict[str, object]] = []
    test_row_ids = split_df.loc[
        split_df["split"] == "test", "row_id"
    ].to_numpy(dtype=int)

    for model_name in args.models:
        predictions = run_model(
            model_name=model_name,
            features=features_by_model[model_name],
            df=df,
            split_df=split_df,
            assays=assays,
            seed=args.seed,
        )
        write_predictions(
            args.out_dir / "predictions" / f"{model_name}_test_predictions.csv",
            predictions,
        )

        metrics = per_assay_metrics(model_name, predictions, split="test")
        metric_frame = metrics_to_frame(metrics)
        metric_frame.to_csv(
            args.out_dir / f"{model_name}_per_assay_metrics.csv", index=False
        )
        all_metric_frames.append(metric_frame)

        summary = macro_summary(metrics)
        summary.update(
            bootstrap_macro_ci(
                predictions,
                test_row_ids=test_row_ids,
                n_bootstrap=args.bootstrap,
                seed=args.seed,
            )
        )
        all_summaries.append(summary)

    if all_metric_frames:
        pd.concat(all_metric_frames, ignore_index=True).to_csv(
            args.out_dir / "per_assay_metrics.csv", index=False
        )
    pd.DataFrame(all_summaries).to_csv(args.out_dir / "summary.csv", index=False)

    write_json(
        args.out_dir / "summary.json",
        {
            "data": str(args.data),
            "split_file": str(args.split_file or args.out_dir / "scaffold_split.csv"),
            "models": args.models,
            "seed": args.seed,
            "bootstrap": args.bootstrap,
            "ecfp_radius": args.ecfp_radius,
            "ecfp_bits": args.ecfp_bits,
            "elapsed_seconds": round(time.time() - started, 3),
            "split_counts": {
                split: int((split_df["split"] == split).sum())
                for split in SPLIT_ORDER
            },
            "summaries": all_summaries,
        },
    )


if __name__ == "__main__":
    main()
