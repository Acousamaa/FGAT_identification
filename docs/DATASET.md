# Dataset specification

## FGAT-Bench

`FGAT_identification.csv` contains 24,042 sentence-level records and four columns:

| Column | Meaning |
|---|---|
| `tactic_labels` | First tactic ID from the original ordered annotation list; metadata only |
| `technique_labels` | First technique ID from the original ordered annotation list; metadata only |
| `sub_technology_labels` | Single flattened classification target |
| `text` | Preprocessed CTI/ATT&CK-derived sentence |

All three label columns are scalar strings. The target column contains 656 unique technique or sub-technique IDs. A sub-technique sample retains the sub-technique ID and does not additionally receive its parent technique label.

The conversion from the earlier list-valued CSV is deterministic: `scripts/prepare_data.py` parses each list and retains element zero. It verifies that every text value is unchanged after rewriting the CSV.

## Multiple-technique sentences

If a sentence textually mentions multiple ATT&CK techniques, the technique selected by the benchmark annotation is retained as the single dominant target. The sentence is not duplicated, and co-mentioned techniques are not added as additional labels. This preserves the benchmark's single-label formulation and avoids introducing labels absent from the provided ground truth.

## TRAM

`TRAM-data.csv` contains 11,130 records with columns `labels` and `text`. Each `labels` cell contains exactly one ATT&CK ID, represented using the source list syntax. The loader accepts both scalar and one-element-list forms.

## Exact split files

`splits/fgat_split.csv` and `splits/tram_split.csv` contain:

- `sample_id`: stable release identifier;
- `source_index`: zero-based data-row index in the source CSV;
- `split`: `train`, `validation`, or `test`;
- `label`: parsed scalar target;
- `text_sha256`: hash of the exact text value.

The split was produced with a scikit-learn-compatible shuffled split using random seed 42: 80% train, followed by a 50/50 division of the remaining 20% into validation and test. Hashes make accidental row reordering or content drift detectable.

## Class distributions

`statistics/*_class_distribution.csv` reports total, training, validation, and test counts for every class. Classes with zero validation or test instances are retained rather than silently omitted.
