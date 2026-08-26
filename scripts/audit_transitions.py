#!/usr/bin/env python3
"""Audit the five mutually exclusive pre/post-correction transition types."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


CATEGORIES = (
    "correct_to_correct",
    "correct_to_incorrect",
    "incorrect_to_correct",
    "incorrect_to_same_incorrect",
    "incorrect_to_different_incorrect",
)


def transition(true_label: str, before_label: str, after_label: str) -> str:
    before_correct = before_label == true_label
    after_correct = after_label == true_label
    if before_correct and after_correct:
        return "correct_to_correct"
    if before_correct and not after_correct:
        return "correct_to_incorrect"
    if not before_correct and after_correct:
        return "incorrect_to_correct"
    if before_label == after_label:
        return "incorrect_to_same_incorrect"
    return "incorrect_to_different_incorrect"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.predictions, dtype=str).fillna("")
    required = {"true_label", "before_label", "after_label"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")

    overall = Counter()
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    categories = []
    for row in frame.itertuples(index=False):
        category = transition(str(row.true_label), str(row.before_label), str(row.after_label))
        categories.append(category)
        overall[category] += 1
        by_class[str(row.true_label)][category] += 1
    frame["transition_category"] = categories
    frame.to_csv(args.output_dir / "audited_predictions.csv", index=False)

    per_class_rows = []
    for label in sorted(by_class):
        row = {"true_label": label}
        row.update({category: by_class[label][category] for category in CATEGORIES})
        row["total"] = sum(row[category] for category in CATEGORIES)
        per_class_rows.append(row)
    pd.DataFrame(per_class_rows).to_csv(args.output_dir / "per_class_transitions.csv", index=False)

    changed_predictions = (
        overall["correct_to_incorrect"]
        + overall["incorrect_to_correct"]
        + overall["incorrect_to_different_incorrect"]
    )
    report = {
        "processed_samples": int(len(frame)),
        "changed_predictions": int(changed_predictions),
        "transition_counts": {category: int(overall[category]) for category in CATEGORIES},
        "five_category_total": int(sum(overall[category] for category in CATEGORIES)),
    }
    if report["five_category_total"] != report["processed_samples"]:
        raise AssertionError("Five transition categories do not cover every processed sample")
    (args.output_dir / "transition_summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
