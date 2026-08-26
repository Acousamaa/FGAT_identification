#!/usr/bin/env python3
"""Normalize labels and build deterministic reproducibility artifacts.

The published FGAT CSV originally stored labels as Python-list strings.  The
paper evaluates a single-label, multi-class task, so this script keeps only the
first item from every FGAT label column and writes scalar ATT&CK identifiers.

It also materializes the exact 80/10/10 split membership, per-class counts,
and dataset hashes. The final feature library remains a release-status
placeholder until manuscript acceptance.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ATTACK_VERSION = "16.1"
SPLIT_SEED = 42
FGAT_LABEL_COLUMNS = (
    "tactic_labels",
    "technique_labels",
    "sub_technology_labels",
)
LABEL_PATTERNS = {
    "tactic_labels": re.compile(r"^TA\d{4}$"),
    "technique_labels": re.compile(r"^T\d{4}(?:\.\d{3})?$"),
    "sub_technology_labels": re.compile(r"^T\d{4}(?:\.\d{3})?$"),
    "labels": re.compile(r"^T\d{4}(?:\.\d{3})?$"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_labels(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"Expected a label list, received: {text!r}")
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [text]


def first_label(value: object) -> str:
    labels = parse_labels(value)
    if not labels:
        raise ValueError("Encountered an empty label cell")
    return labels[0]


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames), rows


def write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def normalize_fgat(path: Path) -> dict[str, object]:
    fieldnames, rows = read_csv_rows(path)
    missing = [name for name in FGAT_LABEL_COLUMNS if name not in fieldnames]
    if missing:
        raise ValueError(f"FGAT CSV is missing columns: {missing}")

    before_text_hashes = [text_sha256(row["text"]) for row in rows]
    multi_counts = Counter()
    for row in rows:
        for column in FGAT_LABEL_COLUMNS:
            labels = parse_labels(row[column])
            if len(labels) > 1:
                multi_counts[column] += 1
            row[column] = labels[0]

    for row_index, row in enumerate(rows):
        for column in FGAT_LABEL_COLUMNS:
            label = row[column]
            if not LABEL_PATTERNS[column].fullmatch(label):
                raise ValueError(f"Invalid {column} at row {row_index}: {label!r}")

    write_csv_atomic(path, fieldnames, rows)
    _, normalized_rows = read_csv_rows(path)
    after_text_hashes = [text_sha256(row["text"]) for row in normalized_rows]
    if before_text_hashes != after_text_hashes:
        raise RuntimeError("Text values changed while normalizing labels")

    return {
        "rows": len(rows),
        "multi_label_cells_reduced": dict(multi_counts),
        "unique_tactics": len({row["tactic_labels"] for row in rows}),
        "unique_techniques": len({row["technique_labels"] for row in rows}),
        "unique_targets": len({row["sub_technology_labels"] for row in rows}),
    }


def split_membership(row_count: int) -> list[str]:
    indices = np.arange(row_count)
    first_rng = np.random.RandomState(SPLIT_SEED)
    first_permutation = first_rng.permutation(row_count)
    temporary_size = math.ceil(row_count * 0.20)
    temporary_indices = indices[first_permutation[:temporary_size]]
    train_indices = indices[first_permutation[temporary_size:]]

    second_rng = np.random.RandomState(SPLIT_SEED)
    second_permutation = second_rng.permutation(len(temporary_indices))
    test_size = math.ceil(len(temporary_indices) * 0.50)
    test_indices = temporary_indices[second_permutation[:test_size]]
    validation_indices = temporary_indices[second_permutation[test_size:]]
    membership = [""] * row_count
    for index in train_indices:
        membership[index] = "train"
    for index in validation_indices:
        membership[index] = "validation"
    for index in test_indices:
        membership[index] = "test"
    if any(not value for value in membership):
        raise RuntimeError("A sample was not assigned to a split")
    return membership


def build_split_and_distribution(
    dataset_name: str,
    csv_path: Path,
    label_column: str,
    output_root: Path,
) -> dict[str, object]:
    _, rows = read_csv_rows(csv_path)
    labels = [first_label(row[label_column]) for row in rows]
    texts = [row["text"] for row in rows]
    membership = split_membership(len(rows))
    prefix = "fgat" if dataset_name == "FGAT-Bench" else "tram"

    split_rows = []
    for index, (label, text, split) in enumerate(zip(labels, texts, membership)):
        split_rows.append(
            {
                "sample_id": f"{prefix}-{index + 1:06d}",
                "source_index": index,
                "split": split,
                "label": label,
                "text_sha256": text_sha256(text),
            }
        )
    write_csv_atomic(
        output_root / "splits" / f"{prefix}_split.csv",
        ("sample_id", "source_index", "split", "label", "text_sha256"),
        split_rows,
    )

    counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }
    for label, split in zip(labels, membership):
        counts[split][label] += 1
    distribution_rows = []
    for label in sorted(set(labels)):
        distribution_rows.append(
            {
                "label": label,
                "total_count": sum(counts[split][label] for split in counts),
                "train_count": counts["train"][label],
                "validation_count": counts["validation"][label],
                "test_count": counts["test"][label],
            }
        )
    write_csv_atomic(
        output_root / "statistics" / f"{prefix}_class_distribution.csv",
        ("label", "total_count", "train_count", "validation_count", "test_count"),
        distribution_rows,
    )

    split_counts = Counter(membership)
    return {
        "dataset": dataset_name,
        "file": csv_path.name,
        "sha256": sha256_file(csv_path),
        "rows": len(rows),
        "classes": len(set(labels)),
        "label_column": label_column,
        "split_seed": SPLIT_SEED,
        "split_method": "scikit-learn-compatible ShuffleSplit; 80/20 then 50/50 on the temporary split",
        "split_counts": {
            "train": split_counts["train"],
            "validation": split_counts["validation"],
            "test": split_counts["test"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-normalize", action="store_true")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    fgat_path = root / "data" / "fgat_bench.csv"
    tram_path = root / "data" / "tram_benchmark.csv"
    summary: dict[str, object] = {
        "attck_version": ATTACK_VERSION,
        "split_seed": SPLIT_SEED,
    }
    if not args.skip_normalize:
        summary["fgat_normalization"] = normalize_fgat(fgat_path)
    summary["datasets"] = [
        build_split_and_distribution(
            "FGAT-Bench", fgat_path, "sub_technology_labels", root
        ),
        build_split_and_distribution("TRAM", tram_path, "labels", root),
    ]
    summary["feature_library"] = {
        "release_status": "Pending upload after manuscript acceptance",
        "placeholder_files": [
            "resources/final_feature_library.csv",
            "resources/final_feature_library.xlsx",
        ],
    }

    manifest_path = root / "statistics" / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
