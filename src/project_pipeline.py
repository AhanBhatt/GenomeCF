from __future__ import annotations

import json
import random
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline


ZENODO_RECORD = "https://zenodo.org/records/16605299/files/{dataset}_v0.zip?download=1"
DATASETS = {
    "human_nontata_promoters": 0,
    "human_enhancers_cohn": 0,
}
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


@dataclass
class DatasetBundle:
    name: str
    train_sequences: list[str]
    train_labels: list[int]
    test_sequences: list[str]
    test_labels: list[int]


def ensure_sequence_dataset(dataset_name: str, data_root: Path) -> Path:
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_dir = data_root / dataset_name
    if dataset_dir.exists():
        return dataset_dir

    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / f"{dataset_name}_v0.zip"
    url = ZENODO_RECORD.format(dataset=dataset_name)
    urllib.request.urlretrieve(url, zip_path)
    shutil.unpack_archive(zip_path, data_root)
    zip_path.unlink()
    return dataset_dir


def read_split(dataset_dir: Path, split: str) -> tuple[list[str], list[int]]:
    sequences: list[str] = []
    labels: list[int] = []
    for label_name, label_id in [("negative", 0), ("positive", 1)]:
        label_dir = dataset_dir / split / label_name
        files = sorted(label_dir.glob("*.txt"))
        for file_path in files:
            seq = file_path.read_text().strip().upper()
            if seq:
                sequences.append(seq)
                labels.append(label_id)
    return sequences, labels


def load_dataset(dataset_name: str, data_root: Path) -> DatasetBundle:
    dataset_dir = ensure_sequence_dataset(dataset_name, data_root)
    train_sequences, train_labels = read_split(dataset_dir, "train")
    test_sequences, test_labels = read_split(dataset_dir, "test")
    return DatasetBundle(
        name=dataset_name,
        train_sequences=train_sequences,
        train_labels=train_labels,
        test_sequences=test_sequences,
        test_labels=test_labels,
    )


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def gc_matched_shuffle(seq: str, rng: random.Random) -> str:
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def create_counterfactuals(sequences: list[str], seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    return {
        "reverse_complement": [reverse_complement(seq) for seq in sequences],
        "gc_shuffle": [gc_matched_shuffle(seq, rng) for seq in sequences],
    }


def build_kmer_logistic_regression(k: int = 6) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "vectorizer",
                CountVectorizer(
                    analyzer="char",
                    ngram_range=(k, k),
                    lowercase=False,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=400,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_binary_classifier(
    model: Pipeline,
    sequences: list[str],
    labels: list[int],
) -> dict[str, float | list[float]]:
    probabilities = model.predict_proba(sequences)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "probabilities": probabilities.tolist(),
    }


def summarize_counterfactual_behavior(
    model: Pipeline,
    sequences: list[str],
    labels: list[int],
    counterfactuals: dict[str, list[str]],
) -> dict[str, float]:
    original_probs = model.predict_proba(sequences)[:, 1]
    rc_probs = model.predict_proba(counterfactuals["reverse_complement"])[:, 1]
    shuffle_probs = model.predict_proba(counterfactuals["gc_shuffle"])[:, 1]
    positive_mask = pd.Series(labels) == 1
    negative_mask = ~positive_mask

    metrics = {
        "reverse_complement_mean_abs_delta": float((abs(original_probs - rc_probs)).mean()),
        "reverse_complement_prediction_flip_rate": float(
            ((original_probs >= 0.5) != (rc_probs >= 0.5)).mean()
        ),
        "positive_original_mean_prob": float(original_probs[positive_mask].mean()),
        "positive_gc_shuffle_mean_prob": float(shuffle_probs[positive_mask].mean()),
        "positive_gc_shuffle_prob_drop": float(
            original_probs[positive_mask].mean() - shuffle_probs[positive_mask].mean()
        ),
        "negative_original_mean_prob": float(original_probs[negative_mask].mean()),
        "negative_gc_shuffle_mean_prob": float(shuffle_probs[negative_mask].mean()),
    }
    return metrics


def save_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def bundle_to_frame(bundle: DatasetBundle) -> pd.DataFrame:
    rows = []
    for split_name, sequences, labels in [
        ("train", bundle.train_sequences, bundle.train_labels),
        ("test", bundle.test_sequences, bundle.test_labels),
    ]:
        for seq, label in zip(sequences, labels):
            rows.append(
                {
                    "dataset": bundle.name,
                    "split": split_name,
                    "label": label,
                    "sequence": seq,
                    "length": len(seq),
                    "gc_fraction": (seq.count("G") + seq.count("C")) / len(seq),
                }
            )
    return pd.DataFrame(rows)
