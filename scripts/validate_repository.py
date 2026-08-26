#!/usr/bin/env python3
"""Validate dataset schema, single-label invariants, splits, and resources."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


FGAT_COLUMNS = ["tactic_labels", "technique_labels", "sub_technology_labels", "text"]
LABEL_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")
TACTIC_RE = re.compile(r"^TA\d{4}$")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_one(value: str) -> str:
    text = value.strip()
    if text.startswith("["):
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, (list, tuple)) or len(parsed) != 1:
            raise ValueError(f"Expected exactly one label: {value!r}")
        return str(parsed[0]).strip()
    return text


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_dataset(
    root: Path,
    filename: str,
    label_column: str,
    split_filename: str,
    distribution_filename: str,
    expected_rows: int,
    expected_classes: int,
) -> dict[str, object]:
    data_rows = read_rows(root / filename)
    if len(data_rows) != expected_rows:
        raise AssertionError(f"{filename}: expected {expected_rows} rows, found {len(data_rows)}")
    labels = [parse_one(row[label_column]) for row in data_rows]
    if any(not LABEL_RE.fullmatch(label) for label in labels):
        raise AssertionError(f"{filename}: invalid ATT&CK target label")
    if len(set(labels)) != expected_classes:
        raise AssertionError(
            f"{filename}: expected {expected_classes} classes, found {len(set(labels))}"
        )

    split_rows = read_rows(root / "splits" / split_filename)
    if len(split_rows) != len(data_rows):
        raise AssertionError(f"{split_filename}: split coverage does not match dataset")
    seen_indices = set()
    split_counts = Counter()
    for split_row in split_rows:
        index = int(split_row["source_index"])
        if index in seen_indices:
            raise AssertionError(f"{split_filename}: duplicate source_index {index}")
        seen_indices.add(index)
        source = data_rows[index]
        if split_row["label"] != parse_one(source[label_column]):
            raise AssertionError(f"{split_filename}: label mismatch at index {index}")
        if split_row["text_sha256"] != text_hash(source["text"]):
            raise AssertionError(f"{split_filename}: text hash mismatch at index {index}")
        split_counts[split_row["split"]] += 1
    if seen_indices != set(range(len(data_rows))):
        raise AssertionError(f"{split_filename}: incomplete source_index coverage")

    distribution_rows = read_rows(root / "statistics" / distribution_filename)
    if len(distribution_rows) != expected_classes:
        raise AssertionError(f"{distribution_filename}: wrong number of classes")
    if sum(int(row["total_count"]) for row in distribution_rows) != len(data_rows):
        raise AssertionError(f"{distribution_filename}: total counts do not reconcile")
    for split_name in ("train", "validation", "test"):
        column = f"{split_name}_count"
        if sum(int(row[column]) for row in distribution_rows) != split_counts[split_name]:
            raise AssertionError(f"{distribution_filename}: {split_name} counts do not reconcile")

    return {
        "file": filename,
        "rows": len(data_rows),
        "classes": len(set(labels)),
        "split_counts": dict(split_counts),
    }


def validate_fgat_scalar_columns(root: Path) -> None:
    rows = read_rows(root / "FGAT_identification.csv")
    if not rows or list(rows[0]) != FGAT_COLUMNS:
        raise AssertionError("FGAT CSV schema differs from the documented four-column schema")
    for index, row in enumerate(rows):
        if row["tactic_labels"].startswith("["):
            raise AssertionError(f"FGAT tactic label remains list-valued at row {index}")
        if row["technique_labels"].startswith("["):
            raise AssertionError(f"FGAT technique label remains list-valued at row {index}")
        if row["sub_technology_labels"].startswith("["):
            raise AssertionError(f"FGAT target label remains list-valued at row {index}")
        if not TACTIC_RE.fullmatch(row["tactic_labels"]):
            raise AssertionError(f"Invalid tactic label at row {index}")


def validate_feature_library_placeholder(root: Path) -> dict[str, str]:
    rows = read_rows(root / "resources" / "final_feature_library.csv")
    if len(rows) != 1 or rows[0].get("release_status") != "Pending upload":
        raise AssertionError("Feature-library placeholder has an unexpected format or status")
    if "technique_id" in rows[0] or "final_features" in rows[0]:
        raise AssertionError("Feature-library placeholder exposes feature data columns")
    return {"release_status": rows[0]["release_status"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()

    validate_fgat_scalar_columns(root)
    report = {
        "fgat": validate_dataset(
            root,
            "FGAT_identification.csv",
            "sub_technology_labels",
            "fgat_split.csv",
            "fgat_class_distribution.csv",
            24042,
            656,
        ),
        "tram": validate_dataset(
            root,
            "TRAM-data.csv",
            "labels",
            "tram_split.csv",
            "tram_class_distribution.csv",
            11130,
            50,
        ),
        "feature_library": validate_feature_library_placeholder(root),
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
