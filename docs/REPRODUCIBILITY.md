# Reproducibility guide

## 1. Verify the release

```bash
pip install -r requirements.txt
python scripts/validate_repository.py
```

The validator checks row counts, scalar FGAT labels, expected class counts, exact split coverage, text hashes, class-distribution reconciliation, and the feature-library release-status placeholder.

## 2. Regenerate derived artifacts

The committed files are the source of truth. To regenerate splits and statistics without rewriting labels:

```bash
python scripts/prepare_data.py --skip-normalize
```

The scalar-label conversion is idempotent, so running without `--skip-normalize` is also safe.

## 3. Train the single-label classifier

```bash
python train_fgat.py \
  --dataset fgat \
  --model-checkpoint markusbayer/CySecBERT \
  --output-dir outputs/fgat-cysecbert
```

Reference parameters are in `configs/proposed_model.yaml`: 16 epochs, batch size 16, sequence length 256, AdamW, learning rate 2e-5, 500 warmup steps, weight decay 0.01, and seed 42. The model uses one integer target per sample, a softmax output, and cross-entropy loss.

## 4. Evaluate

```bash
python src/evaluate_predictions.py \
  --predictions outputs/fgat-cysecbert/test_predictions.csv \
  --output-dir outputs/fgat-cysecbert/metrics
```

Primary Precision, Recall, and F1 are weighted averages. Macro and micro metrics are reported separately. In single-label multi-class evaluation, micro-Precision, micro-Recall, micro-F1, and accuracy must be equal; the evaluation script raises an error if this invariant is violated.

## 5. Correction audit

Given a CSV with `sample_id,true_label,before_label,after_label`, run:

```bash
python scripts/audit_transitions.py \
  --predictions correction_predictions.csv \
  --output-dir outputs/correction_audit
```

The report uses exactly five mutually exclusive categories: correct-to-correct, correct-to-incorrect, incorrect-to-correct, incorrect-to-same-incorrect, and incorrect-to-different-incorrect. It also writes per-ground-truth-class counts and the number of changed predictions.

## 6. Feature-library release status

`resources/final_feature_library.csv` and
`resources/final_feature_library.xlsx` are placeholders during peer review. The
final feature-library entries will be uploaded after manuscript acceptance, as
stated in the manuscript. The correction utility requires that released file
and will report a clear error while the placeholder is present.

## External assets

- CySecBERT checkpoint: `markusbayer/CySecBERT`
- GloVe: 300-dimensional Common Crawl 840B vectors
- ATT&CK source: MITRE ATT&CK Enterprise v16.1

These large/upstream assets are not committed to this repository.
