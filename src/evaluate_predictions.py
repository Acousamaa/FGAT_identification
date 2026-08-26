#!/usr/bin/env python3
"""Evaluate scalar predictions with explicit averaging strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.predictions)
    required = {"true_label", "predicted_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    true_labels = frame["true_label"].astype(str)
    predicted_labels = frame["predicted_label"].astype(str)

    metrics = {"samples": int(len(frame)), "accuracy": float(accuracy_score(true_labels, predicted_labels))}
    labels = sorted(set(true_labels) | set(predicted_labels))
    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=labels,
            average=None,
            zero_division=0,
        )
    )
    per_class = pd.DataFrame(
        {
            "label": labels,
            "precision": per_class_precision,
            "recall": per_class_recall,
            "f1": per_class_f1,
            "support": per_class_support,
        }
    )
    for average in ("weighted", "macro", "micro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average=average,
            zero_division=0,
        )
        metrics[average] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    for name in ("precision", "recall", "f1"):
        if not np.isclose(metrics["micro"][name], metrics["accuracy"], atol=1e-12):
            raise AssertionError(f"Single-label invariant failed: micro-{name} != accuracy")

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    per_class.to_csv(args.output_dir / "per_class_metrics.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
