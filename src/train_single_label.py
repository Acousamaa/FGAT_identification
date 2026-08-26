#!/usr/bin/env python3
"""Fine-tune a Hugging Face sequence classifier on the released exact splits."""

from __future__ import annotations

import argparse
import ast
import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_label(value: object) -> str:
    text = str(value).strip()
    if text.startswith("["):
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, (list, tuple)) or len(parsed) != 1:
            raise ValueError(f"Expected one label, received {text!r}")
        return str(parsed[0]).strip()
    return text


class ClassificationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, label_to_id: dict[str, int], max_length: int):
        self.texts = frame["text"].astype(str).tolist()
        self.labels = [label_to_id[label] for label in frame["label"].tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def metric_dict(true_ids: np.ndarray, predicted_ids: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {"accuracy": float(accuracy_score(true_ids, predicted_ids))}
    for average in ("weighted", "macro", "micro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_ids,
            predicted_ids,
            average=average,
            zero_division=0,
        )
        metrics[f"{average}_precision"] = float(precision)
        metrics[f"{average}_recall"] = float(recall)
        metrics[f"{average}_f1"] = float(f1)
    for key in ("micro_precision", "micro_recall", "micro_f1"):
        if not np.isclose(metrics[key], metrics["accuracy"], atol=1e-12):
            raise AssertionError(f"Single-label invariant failed: {key} != accuracy")
    return metrics


def load_release(root: Path, dataset: str) -> tuple[pd.DataFrame, str, Path]:
    if dataset == "fgat":
        filename = "FGAT_identification.csv"
        label_column = "sub_technology_labels"
        split_path = root / "splits" / "fgat_split.csv"
    else:
        filename = "TRAM-data.csv"
        label_column = "labels"
        split_path = root / "splits" / "tram_split.csv"
    frame = pd.read_csv(root / filename, encoding="utf-8-sig")
    frame["label"] = frame[label_column].map(parse_label)
    frame["source_index"] = np.arange(len(frame))
    split_frame = pd.read_csv(split_path)
    merged = frame.merge(
        split_frame[["sample_id", "source_index", "split", "text_sha256"]],
        on="source_index",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(frame):
        raise AssertionError("Split manifest does not cover the dataset exactly once")
    return merged, label_column, split_path


def build_training_arguments(args) -> TrainingArguments:
    values = dict(
        output_dir=str(args.output_dir),
        run_name=f"{args.dataset}-CySecBERT-single-label",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available() and not args.no_fp16,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
    )
    parameter_names = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameter_names:
        values["eval_strategy"] = "epoch"
    else:
        values["evaluation_strategy"] = "epoch"
    return TrainingArguments(**values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", choices=("fgat", "tram"), default="fgat")
    parser.add_argument("--model-checkpoint", default="markusbayer/CySecBERT")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    frame, _, _ = load_release(args.repo_root, args.dataset)
    labels = sorted(frame["label"].unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_checkpoint,
        num_labels=len(labels),
        label2id=label_to_id,
        id2label=id_to_label,
        problem_type="single_label_classification",
    )
    datasets = {
        split: ClassificationDataset(
            frame[frame["split"] == split].reset_index(drop=True),
            tokenizer,
            label_to_id,
            args.max_length,
        )
        for split in ("train", "validation", "test")
    }

    def compute_metrics(prediction) -> dict[str, float]:
        logits, true_ids = prediction
        predicted_ids = np.asarray(logits).argmax(axis=1)
        return metric_dict(np.asarray(true_ids), predicted_ids)

    trainer = Trainer(
        model=model,
        args=build_training_arguments(args),
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir / "best_model")
    tokenizer.save_pretrained(args.output_dir / "best_model")

    prediction = trainer.predict(datasets["test"])
    logits = np.asarray(prediction.predictions)
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    predicted_ids = probabilities.argmax(axis=1)
    test_frame = frame[frame["split"] == "test"].reset_index(drop=True)
    output_rows = []
    for index, row in test_frame.iterrows():
        order = np.argsort(probabilities[index])[::-1][:5]
        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "true_label": row["label"],
                "predicted_label": id_to_label[int(predicted_ids[index])],
                "confidence": float(probabilities[index, predicted_ids[index]]),
                "top5_labels": json.dumps([id_to_label[int(item)] for item in order]),
                "top5_scores": json.dumps([float(probabilities[index, item]) for item in order]),
                "text": row["text"],
            }
        )
    with (args.output_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(prediction.metrics, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
