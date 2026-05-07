"""Shared benchmark utilities for Tox21Full.

The helpers in this module intentionally keep model training separate from
evaluation.  Classical baselines, Chemprop, and future model families should all
write predictions that flow through the same metric code.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


SPLIT_ORDER = ("train", "valid", "test")


@dataclass
class MetricResult:
    model: str
    assay: str
    split: str
    n: int
    positives: int
    negatives: int
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    reason: str = ""


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV, gzipped CSV, or Parquet table."""
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format for {path}")


def label_columns(df: pd.DataFrame, smiles_column: str = "smiles") -> List[str]:
    """Return assay label columns in dataset order."""
    if smiles_column not in df.columns:
        raise ValueError(f"Missing required SMILES column: {smiles_column}")
    return [c for c in df.columns if c != smiles_column]


def compute_binary_metrics(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[Optional[float], Optional[float], str]:
    """Compute ROC-AUC and PR-AUC when the labels support the metric."""
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask].astype(int)
    y_score = y_score[mask].astype(float)

    if y_true.size == 0:
        return None, None, "no observed labels"

    positives = int(y_true.sum())
    negatives = int(y_true.size - positives)
    roc = None
    pr = None
    reasons = []

    if positives > 0 and negatives > 0:
        roc = float(roc_auc_score(y_true, y_score))
    else:
        reasons.append("single class in evaluation labels")

    if positives > 0:
        pr = float(average_precision_score(y_true, y_score))
    else:
        reasons.append("no positive labels for PR-AUC")

    return roc, pr, "; ".join(reasons)


def per_assay_metrics(
    model_name: str,
    prediction_records: Mapping[str, pd.DataFrame],
    split: str = "test",
) -> List[MetricResult]:
    """Evaluate a model's predictions for each assay."""
    results: List[MetricResult] = []
    for assay, pred in prediction_records.items():
        y_true = pred["y_true"].to_numpy(dtype=float)
        y_score = pred["y_score"].to_numpy(dtype=float)
        finite = np.isfinite(y_true) & np.isfinite(y_score)
        labels = y_true[finite].astype(int)
        positives = int(labels.sum())
        negatives = int(labels.size - positives)
        roc, pr, reason = compute_binary_metrics(y_true, y_score)
        results.append(
            MetricResult(
                model=model_name,
                assay=assay,
                split=split,
                n=int(labels.size),
                positives=positives,
                negatives=negatives,
                roc_auc=roc,
                pr_auc=pr,
                reason=reason,
            )
        )
    return results


def metrics_to_frame(metrics: Sequence[MetricResult]) -> pd.DataFrame:
    """Convert metric dataclasses to a DataFrame."""
    return pd.DataFrame([m.__dict__ for m in metrics])


def macro_summary(metrics: Sequence[MetricResult]) -> Dict[str, object]:
    """Summarize per-assay metrics using macro means over defined assays."""
    roc_values = [m.roc_auc for m in metrics if m.roc_auc is not None]
    pr_values = [m.pr_auc for m in metrics if m.pr_auc is not None]
    total_n = int(sum(m.n for m in metrics))
    total_pos = int(sum(m.positives for m in metrics))
    total_neg = int(sum(m.negatives for m in metrics))
    model = metrics[0].model if metrics else ""
    return {
        "model": model,
        "assays_total": len(metrics),
        "assays_with_roc_auc": len(roc_values),
        "assays_with_pr_auc": len(pr_values),
        "test_observed_labels": total_n,
        "test_positive_labels": total_pos,
        "test_negative_labels": total_neg,
        "macro_roc_auc": float(np.mean(roc_values)) if roc_values else None,
        "macro_pr_auc": float(np.mean(pr_values)) if pr_values else None,
    }


def _weighted_metric_sample(
    pred: pd.DataFrame, draw_counts: Mapping[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    positions: List[int] = []
    row_ids = pred["row_id"].to_numpy(dtype=int)
    for pos, row_id in enumerate(row_ids):
        count = draw_counts.get(int(row_id), 0)
        if count:
            positions.extend([pos] * count)
    if not positions:
        return np.array([], dtype=float), np.array([], dtype=float)
    return (
        pred["y_true"].to_numpy(dtype=float)[positions],
        pred["y_score"].to_numpy(dtype=float)[positions],
    )


def bootstrap_macro_ci(
    prediction_records: Mapping[str, pd.DataFrame],
    test_row_ids: Sequence[int],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, object]:
    """Bootstrap macro ROC-AUC and PR-AUC by resampling test compounds."""
    if n_bootstrap <= 0:
        return {
            "bootstrap_samples": 0,
            "macro_roc_auc_ci95": None,
            "macro_pr_auc_ci95": None,
        }

    rng = np.random.default_rng(seed)
    test_row_ids = np.asarray(test_row_ids, dtype=int)
    roc_samples: List[float] = []
    pr_samples: List[float] = []

    for _ in range(n_bootstrap):
        draw = rng.choice(test_row_ids, size=test_row_ids.size, replace=True)
        draw_counts = Counter(int(i) for i in draw)
        roc_values: List[float] = []
        pr_values: List[float] = []

        for pred in prediction_records.values():
            y_true, y_score = _weighted_metric_sample(pred, draw_counts)
            roc, pr, _ = compute_binary_metrics(y_true, y_score)
            if roc is not None:
                roc_values.append(roc)
            if pr is not None:
                pr_values.append(pr)

        if roc_values:
            roc_samples.append(float(np.mean(roc_values)))
        if pr_values:
            pr_samples.append(float(np.mean(pr_values)))

    def ci(values: Sequence[float]) -> Optional[List[float]]:
        if not values:
            return None
        lo, hi = np.percentile(np.asarray(values), [2.5, 97.5])
        return [float(lo), float(hi)]

    return {
        "bootstrap_samples": int(n_bootstrap),
        "macro_roc_auc_ci95": ci(roc_samples),
        "macro_pr_auc_ci95": ci(pr_samples),
    }


def write_json(path: Path, payload: object) -> None:
    """Write pretty JSON with stable key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_predictions(
    out_path: Path, prediction_records: Mapping[str, pd.DataFrame]
) -> None:
    """Write long-format predictions for all assays."""
    frames = []
    for assay, pred in prediction_records.items():
        frame = pred.copy()
        frame.insert(0, "assay", assay)
        frames.append(frame)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    else:
        pd.DataFrame(
            columns=["assay", "row_id", "smiles", "y_true", "y_score"]
        ).to_csv(out_path, index=False)


def collect_prediction_records_from_wide(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    row_ids: Sequence[int],
    assays: Sequence[str],
    smiles_column: str = "smiles",
) -> Dict[str, pd.DataFrame]:
    """Convert wide prediction output into the long format used by metrics."""
    records: Dict[str, pd.DataFrame] = {}
    row_ids = np.asarray(row_ids, dtype=int)
    truth_slice = truth.iloc[row_ids]

    aliases: Dict[str, str] = {}
    for column in predictions.columns:
        lower = column.lower()
        aliases[column] = column
        if lower.startswith("pred_"):
            aliases[column[5:]] = column
        if lower.endswith("_pred"):
            aliases[column[:-5]] = column

    for assay in assays:
        pred_col = aliases.get(assay)
        if pred_col is None:
            continue
        y_true = truth_slice[assay].to_numpy(dtype=float)
        y_score = predictions[pred_col].to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_score)
        records[assay] = pd.DataFrame(
            {
                "row_id": row_ids[mask],
                "smiles": truth_slice[smiles_column].to_numpy()[mask],
                "y_true": y_true[mask],
                "y_score": y_score[mask],
            }
        )
    return records
