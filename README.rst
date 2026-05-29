Tox21 Full
~~~~~~~~~~

This is a Tox21-like dataset created from the public PubChem BioAssay Tox21
summary assay records. The common Tox21 benchmark includes 12 assays; this
generator currently includes 75 Tox21 summary assays.

The package builds a multi-task molecular activity table with one SMILES column
and one binary activity column per assay.


Downloads PubChem assay data and creates a clean CSV.GZ file ready for import
into pandas:

::

    tox21full ~/Downloads/tox21full.csv.gz


You can also create it as a parquet file (more efficient):

::

    tox21full --format parquet ~/Downloads/tox21full.parquet


Benchmark workflow
------------------

The benchmark protocol lives in ``benchmarks/``.  It creates a
deterministic scaffold split, runs ECFP/logistic regression, ECFP/random forest,
and RDKit-descriptor boosted-tree baselines, and prepares the same split for a
Chemprop MPNN baseline.

::

    python -m pip install -r benchmarks/requirements-benchmark.txt
    python benchmarks/run_sklearn_baselines.py \
        --data tox21full.csv \
        --out-dir benchmarks/results \
        --bootstrap 1000 \
        --seed 20260506

For the Chemprop baseline:

::

    python -m pip install -r benchmarks/requirements-chemprop.txt
    python benchmarks/prepare_chemprop.py \
        --data tox21full.csv \
        --split-file benchmarks/results/scaffold_split.csv \
        --out-dir benchmarks/results/chemprop
    bash benchmarks/results/chemprop/run_chemprop.sh
