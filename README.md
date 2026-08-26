# FGAT-Bench

FGAT-Bench is a sentence-level benchmark for fine-grained ATT&CK technique identification. This repository contains the dataset, an independent TRAM evaluation copy, deterministic split manifests, full class distributions, reproducibility configurations, and reference code for single-label training and auditing.

The dataset was constructed from MITRE ATT&CK Enterprise **v16.1**. The task is evaluated as **single-label, multi-class classification** over 656 flattened technique/sub-technique targets for FGAT-Bench and 50 targets for TRAM.

## Dataset summary

| Dataset | Samples | Classes | Train | Validation | Test | Target column |
|---|---:|---:|---:|---:|---:|---|
| FGAT-Bench | 24,042 | 656 | 19,233 | 2,404 | 2,405 | `sub_technology_labels` |
| TRAM | 11,130 | 50 | 8,904 | 1,113 | 1,113 | `labels` |

The split seed is `42`. Exact membership is stored in `splits/`; experiments should use these files instead of generating a new random split.

## Single-label release policy

`data/fgat_bench.csv` previously represented labels as Python-list strings. It has been normalized as follows:

- only the first item in each original label list is retained;
- `tactic_labels`, `technique_labels`, and `sub_technology_labels` now contain scalar ATT&CK IDs;
- the classification target is `sub_technology_labels`;
- technique and sub-technique IDs are separate mutually exclusive classes in one flattened label space;
- when text mentions multiple ATT&CK techniques, the dataset-annotated target is retained as the single dominant label; the sample is not duplicated and co-mentioned techniques are not added as extra labels.

This transformation reduced 4,900 multi-valued tactic cells and 351 multi-valued technique cells. The text field and row order were preserved. The resulting FGAT target space contains exactly 656 classes.

## Repository layout

```text
data/fgat_bench.csv                     FGAT-Bench, normalized scalar labels
data/tram_benchmark.csv                 independent TRAM benchmark copy
configs/proposed_model.yaml             proposed-model configuration
configs/baselines.csv                   per-baseline reproducibility table
splits/                                 exact sample-level split membership
statistics/                             full per-class distributions and dataset manifest
resources/final_feature_library.*       feature-library release-status placeholders
docs/                                   provenance and reproduction guidance
scripts/                                preparation and audit utilities
src/                                    single-label training/evaluation code
```

## Quick start

Create an environment and validate the release:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/validate_repository.py
```

Fine-tune CySecBERT on the exact FGAT split:

```bash
python src/train_fgat.py \
  --dataset fgat \
  --model-checkpoint markusbayer/CySecBERT \
  --output-dir outputs/fgat-cysecbert
```

Evaluate an exported prediction file:

```bash
python src/evaluate_predictions.py \
  --predictions outputs/fgat-cysecbert/test_predictions.csv \
  --output-dir outputs/fgat-cysecbert/metrics
```

The evaluation utility reports accuracy, weighted/macro/micro Precision, Recall and F1, and per-class statistics. For this single-label setting, micro-Precision, micro-Recall, micro-F1, and accuracy are checked for numerical equality.

## Reproducibility resources

- ATT&CK version and stage-specific information-source mapping: [`docs/ATTACK_PROVENANCE.md`](docs/ATTACK_PROVENANCE.md)
- Dataset schemas and label semantics: [`docs/DATASET.md`](docs/DATASET.md)
- End-to-end reproduction commands: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- Expert enrichment protocol: [`docs/EXPERT_ENRICHMENT_GUIDELINES.md`](docs/EXPERT_ENRICHMENT_GUIDELINES.md)
- Five-category correction audit: `scripts/audit_transitions.py`

The CSV and XLSX files named `final_feature_library` are release-status
placeholders. In accordance with the manuscript's data-availability statement,
the final feature-library entries will be uploaded after manuscript acceptance.

Model weights and the 840B Common Crawl GloVe vectors are not redistributed here because of file size and upstream licensing. The configuration files identify the required checkpoints and parameters.
