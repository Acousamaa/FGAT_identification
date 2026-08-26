#!/usr/bin/env python3
"""Apply the frozen feature library to low-confidence scalar predictions.

This transparent release implementation uses exact phrase matches first and an
optional GloVe semantic match for unresolved phrases. It never emits multiple
labels for one sentence. The final library entries are scheduled for release
after manuscript acceptance.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


TOKEN_RE = re.compile(r"[A-Za-z0-9_./\\:-]+")


def features(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def phrase_pattern(value: str) -> re.Pattern[str]:
    normalized = " ".join(value.casefold().split())
    return re.compile(r"(?<![a-z0-9_])" + re.escape(normalized) + r"(?![a-z0-9_])")


def load_library(path: Path) -> dict[str, list[str]]:
    frame = pd.read_csv(path)
    required = {"technique_id", "final_features"}
    if missing := required - set(frame.columns):
        raise RuntimeError(
            "The final feature library is pending release after manuscript acceptance; "
            f"the current placeholder is missing columns: {sorted(missing)}"
        )
    return {
        str(row.technique_id): features(row.final_features)
        for row in frame.itertuples(index=False)
    }


def mean_vector(tokens: list[str], vectors: KeyedVectors) -> np.ndarray | None:
    available = [vectors[token] for token in tokens if token in vectors]
    if not available:
        return None
    vector = np.mean(available, axis=0)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else None


def semantic_similarity(text: str, phrase: str, vectors: KeyedVectors) -> float:
    text_tokens = [token.casefold() for token in TOKEN_RE.findall(text)]
    phrase_tokens = [token.casefold() for token in TOKEN_RE.findall(phrase)]
    phrase_vector = mean_vector(phrase_tokens, vectors)
    if phrase_vector is None or not text_tokens:
        return 0.0
    width = max(1, len(phrase_tokens))
    best = 0.0
    for start in range(0, max(1, len(text_tokens) - width + 1)):
        candidate = mean_vector(text_tokens[start : start + width], vectors)
        if candidate is not None:
            best = max(best, float(np.dot(phrase_vector, candidate)))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--feature-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glove", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--semantic-threshold", type=float, default=0.8)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.predictions)
    required = {"sample_id", "text", "predicted_label", "confidence"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    library = load_library(args.feature_library)
    vectors = None
    if args.glove:
        vectors = KeyedVectors.load_word2vec_format(args.glove, binary=False, no_header=True)

    output_rows = []
    for row in frame.itertuples(index=False):
        before_label = str(row.predicted_label)
        after_label = before_label
        matched = []
        best_score = 0.0
        if float(row.confidence) < args.confidence_threshold:
            lower_text = " ".join(str(row.text).casefold().split())
            for technique_id, entries in library.items():
                exact_hits = [entry for entry in entries if phrase_pattern(entry).search(lower_text)]
                semantic_hits = []
                if vectors is not None:
                    for entry in entries:
                        if entry not in exact_hits and semantic_similarity(row.text, entry, vectors) >= args.semantic_threshold:
                            semantic_hits.append(entry)
                score = 2.0 * len(exact_hits) + len(semantic_hits)
                if score > best_score:
                    best_score = score
                    after_label = technique_id
                    matched = [*exact_hits, *semantic_hits]
        output = row._asdict()
        output.update(
            {
                "before_label": before_label,
                "after_label": after_label,
                "correction_score": best_score,
                "matched_features": "; ".join(matched),
            }
        )
        output_rows.append(output)
    pd.DataFrame(output_rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
