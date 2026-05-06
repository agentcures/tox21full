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


You can also create it as a parquet file (more efficent):

:: 

    tox21full --format parquet ~/Downloads/tox21full.parquet
