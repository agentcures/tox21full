# Novelty Positioning for Tox21Full

This note supports the benchmark positioning argument that Tox21Full is
not merely a repackaging of existing toxicity datasets.  The benchmark report should
use the concise version of this comparison; this file can remain as supplemental
positioning and reviewer-facing documentation.

## Comparison Matrix

| Resource | What it already provides | Why it does not replace Tox21Full | How Tox21Full is positioned |
| --- | --- | --- | --- |
| MoleculeNet / canonical Tox21 | Standard molecular ML benchmark with roughly 8k compounds and 12 Tox21 qualitative targets. DeepChem's `load_tox21` exposes the same 12 named tasks and defaults to ECFP-style featurization plus balancing transforms. | It is widely used but frozen around the 2014 challenge-style task set. It does not expose the broader set of PubChem BioAssay Tox21 summary assays now available through maintained PubChem endpoints. | Tox21Full keeps the familiar SMILES-by-task multilabel format but expands the endpoint set to 75 fixed PubChem Tox21 summary assays, enabling stronger tests of multitask transfer, endpoint imbalance, and missing-label masking. |
| DeepChem ToxCast loader | A ready-to-use MolNet loader for a larger toxicology collection. DeepChem documents ToxCast as qualitative results for over 600 experiments on about 8k compounds. | ToxCast is broader and more heterogeneous than the Tox21 summary-assay setting. It is valuable, but it answers a different benchmark question: broad ToxCast bioactivity coverage rather than a reproducible expansion of Tox21-style PubChem summary assays. | Tox21Full occupies the middle ground between the small 12-task Tox21 benchmark and broad ToxCast: more Tox21 endpoints while preserving a single-source, Tox21-summary-assay construction. |
| EPA ToxCast / CompTox resources | EPA provides large-scale bioactivity, chemistry, toxicity, exposure, and dashboard/download resources. EPA notes that ToxCast data integrate assays from many sources and that recent invitrodb releases are available for download. | These are authoritative toxicology resources, but they are not a single small molecular ML benchmark table with fixed PubChem AIDs, explicit any-active aggregation, missing-label semantics, scaffold split, and baseline predictions. | Tox21Full is a model-evaluation artifact derived from public records, not a replacement for EPA's toxicology databases. It trades breadth for benchmark simplicity, auditability, and direct comparability to the canonical Tox21 modeling workflow. |
| Raw PubChem BioAssay / PUG-REST | PubChem provides programmatic access to assay records, summaries, concise assay data, and compound properties in formats including JSON and CSV. | Raw availability is not the same as a stable benchmark. Users must decide which AIDs to include, how to group multiple rows per CID, how to map substances to compounds, how to treat missing labels, how to version live data, and how to report metrics. Different choices would create incomparable datasets. | Tox21Full fixes those choices in code and metadata: AID manifest, retrieval date, CID grouping, binary label rule, SMILES mapping, artifact hashes, Croissant metadata, scaffold split, and baseline evaluation scripts. |

## Reviewer-Facing Novelty Claims

Use these claims only when the corresponding artifacts are present:

1. **Expanded Tox21-style endpoint coverage.** Tox21Full increases the canonical
   12-task Tox21 modeling surface to 75 PubChem Tox21 summary assays while
   retaining a familiar structure-first molecular classification table.
2. **Fixed and auditable PubChem construction.** The release fixes source AIDs,
   retrieval date, row filtering, CID aggregation, SMILES mapping, and missing
   label semantics rather than relying on opaque challenge-era files.
3. **Benchmark, not only dataset.** The release includes a deterministic
   scaffold split, classical baselines, a Chemprop D-MPNN path, per-assay
   metrics, and compound-bootstrap confidence intervals.
4. **ToxCast complement, not competitor.** Tox21Full provides a controlled
   Tox21-summary-assay benchmark; ToxCast/CompTox remains the broader
   toxicology resource for heterogeneous assay programs and chemical metadata.
5. **Live-data reproducibility boundary.** PubChem is live, so Tox21Full
   distinguishes the hosted versioned artifact from future regenerations and
   records hashes and provenance files for audit.

## Benchmark Report Changes Needed After Running Baselines

The final benchmark report should include:

- A main-text table of macro ROC-AUC and PR-AUC for all four model families.
- A short statement that all reported metrics mask missing labels.
- A note that confidence intervals are compound-bootstrap intervals over the
  fixed scaffold test split.
- A supplement table or CSV with per-assay metrics and assay-level positive
  counts.
- A paragraph explaining why Tox21Full is deliberately narrower than ToxCast but
  broader than canonical Tox21.

## Primary Sources Consulted

- DeepChem MoleculeNet API reference for `load_tox21` and `load_toxcast`:
  https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html
- EPA ToxCast data page:
  https://www.epa.gov/comptox-tools/exploring-toxcast-data
- EPA CompTox Chemicals Dashboard resource hub:
  https://www.epa.gov/comptox-tools/comptox-chemicals-dashboard-resource-hub
- PubChem PUG-REST specification and tutorial:
  https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
  https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest-tutorial
- Chemprop CLI training documentation:
  https://chemprop.readthedocs.io/en/latest/tutorial/cli/train.html
