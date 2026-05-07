# Tox21Full Baseline Benchmark Protocol

This directory contains the benchmark workflow.  The protocol fixes the split, models,
metrics, and output schema so baseline results can be regenerated and audited.

## Scope

The primary benchmark uses a deterministic 80/10/10 Bemis-Murcko scaffold split.
All labels for a compound stay in the same split, and missing assay labels are
masked for both training and evaluation.

Report the following models:

- `ecfp_logreg`: ECFP4/Morgan fingerprints plus class-balanced logistic
  regression.
- `ecfp_rf`: ECFP4/Morgan fingerprints plus class-balanced random forest.
- `rdkit_hgb`: RDKit 2D descriptors plus histogram gradient-boosted trees.
- `chemprop_mpnn`: Chemprop directed message-passing neural network, trained as
  a multitask classifier with blank targets preserved as missing labels.

The required metrics are per-assay ROC-AUC and PR-AUC, macro means across
defined assays, and 95% confidence intervals from compound-level bootstrap
resampling of the scaffold test split.

## Installation

Classical baselines:

```bash
python -m pip install -r benchmarks/requirements-benchmark.txt
```

Chemprop baseline:

```bash
python -m pip install -r benchmarks/requirements-chemprop.txt
```

## Classical Baselines

Generate a split and run the sklearn baselines:

```bash
python benchmarks/run_sklearn_baselines.py \
  --data tox21full.csv \
  --out-dir benchmarks/results \
  --models ecfp_logreg,ecfp_rf,rdkit_hgb \
  --bootstrap 1000 \
  --seed 20260506
```

Generate the per-assay PR-AUC distribution figure:

```bash
python benchmarks/plot_pr_auc_distribution.py \
  --metrics benchmarks/results/per_assay_metrics_all.csv \
  --out benchmarks/results/figures/per_assay_pr_auc_distribution.pdf
```

Outputs:

- `scaffold_split.csv`: row-level split assignments and scaffold keys.
- `summary.csv` / `summary.json`: macro ROC-AUC, macro PR-AUC, and 95% CIs.
- `per_assay_metrics.csv`: one row per model and assay.
- `predictions/*_test_predictions.csv`: long-format test predictions with
  `assay`, `row_id`, `smiles`, `y_true`, and `y_score`.

## Chemprop Baseline

Prepare Chemprop inputs from the same scaffold split:

```bash
python benchmarks/prepare_chemprop.py \
  --data tox21full.csv \
  --split-file benchmarks/results/scaffold_split.csv \
  --out-dir benchmarks/results/chemprop \
  --epochs 50 \
  --batch-size 64 \
  --ensemble-size 1 \
  --seed 20260506
```

Run the generated command file:

```bash
bash benchmarks/results/chemprop/run_chemprop.sh
```

The command file trains Chemprop, exports test-set probabilities, and evaluates
them with `benchmarks/evaluate_predictions.py`, producing the same
`summary.csv`, `per_assay_metrics.csv`, and long-format predictions used by the
classical baselines.

## Benchmark Table Contract

Once the runs complete, the benchmark report should include a compact macro table:

| Model | ROC-AUC | PR-AUC | ROC-AUC 95% CI | PR-AUC 95% CI |
| --- | ---: | ---: | ---: | ---: |
| ECFP + logistic regression | from `summary.csv` | from `summary.csv` | from `summary.json` | from `summary.json` |
| ECFP + random forest | from `summary.csv` | from `summary.csv` | from `summary.json` | from `summary.json` |
| RDKit descriptors + boosted trees | from `summary.csv` | from `summary.csv` | from `summary.json` | from `summary.json` |
| Chemprop D-MPNN | from Chemprop `summary.csv` | from Chemprop `summary.csv` | from Chemprop `summary.json` | from Chemprop `summary.json` |

The full benchmark output should include `per_assay_metrics.csv` and prediction files so
reviewers can inspect assays with undefined ROC-AUC or low positive counts.

## Notes

Scaffold handling follows a conservative rule: molecules with a non-empty
Bemis-Murcko scaffold are grouped by scaffold; valid acyclic molecules are
grouped by canonical acyclic SMILES so all acyclic compounds are not forced into
one giant split group; invalid SMILES, if any, are isolated by row ID and marked
in the split file.
