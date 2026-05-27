from __future__ import annotations

import copy
import json
import math
import os
import random
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer


ZENODO_RECORD = "https://zenodo.org/records/16605299/files/{dataset}_v0.zip?download=1"
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
DEVICE = torch.device("cpu")
VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
DEFAULT_BINARY_TASKS = [
    "human_nontata_promoters",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ocr_ensembl",
    "dummy_mouse_enhancers_ensembl",
    "drosophila_enhancers_stark",
]
HUMAN_TASKS = {
    "human_nontata_promoters",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ocr_ensembl",
}
HOLDOUT_CHROMS = ("chr1", "chr2")


@dataclass
class TaskFrame:
    dataset: str
    split_scheme: str
    train: pd.DataFrame
    test: pd.DataFrame


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_sequence_dataset(dataset_name: str, data_root: Path) -> Path:
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


def read_interval_metadata(dataset_name: str, repo_root: Path) -> pd.DataFrame:
    dataset_dir = repo_root / "datasets" / dataset_name
    rows: list[pd.DataFrame] = []
    for split in ["train", "test"]:
        split_dir = dataset_dir / split
        for file_path in sorted(split_dir.glob("*.csv.gz")):
            label_name = file_path.stem.replace(".csv", "")
            df = pd.read_csv(file_path, compression="gzip")
            df["orig_split"] = split
            df["label_name"] = label_name
            rows.append(df)
    frame = pd.concat(rows, ignore_index=True)
    label_names = sorted(frame["label_name"].unique())
    if len(label_names) != 2:
        raise ValueError(f"{dataset_name} is not binary: {label_names}")
    frame["label"] = (frame["label_name"] == "positive").astype(int)
    if set(label_names) != {"negative", "positive"}:
        # For non-negative/positive binary tasks, assign sorted labels.
        label_map = {label_names[0]: 0, label_names[1]: 1}
        frame["label"] = frame["label_name"].map(label_map)
    return frame


def add_sequence_paths(frame: pd.DataFrame, dataset_name: str, data_root: Path) -> pd.DataFrame:
    ensure_sequence_dataset(dataset_name, data_root)
    sequence_root = data_root / dataset_name
    enriched = frame.copy()
    enriched["sequence_path"] = enriched.apply(
        lambda row: sequence_root / row["orig_split"] / row["label_name"] / f"{row['id']}.txt",
        axis=1,
    )
    return enriched


def sample_per_label(frame: pd.DataFrame, n_per_label: int | None, seed: int) -> pd.DataFrame:
    if n_per_label is None:
        return frame.reset_index(drop=True)
    parts = []
    for label in sorted(frame["label"].unique()):
        subset = frame[frame["label"] == label]
        if len(subset) <= n_per_label:
            parts.append(subset)
        else:
            parts.append(subset.sample(n=n_per_label, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def read_sequences(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["sequence"] = enriched["sequence_path"].map(lambda path: Path(path).read_text().strip().upper())
    enriched["length"] = enriched["sequence"].str.len()
    enriched["gc_fraction"] = enriched["sequence"].map(lambda seq: (seq.count("G") + seq.count("C")) / len(seq))
    return enriched


def load_task_frame(
    dataset_name: str,
    data_root: Path,
    repo_root: Path,
    split_scheme: str,
    seed: int,
    train_per_label: int | None = None,
    test_per_label: int | None = None,
) -> TaskFrame:
    metadata = add_sequence_paths(read_interval_metadata(dataset_name, repo_root), dataset_name, data_root)
    if split_scheme == "official":
        train = metadata[metadata["orig_split"] == "train"].copy()
        test = metadata[metadata["orig_split"] == "test"].copy()
    elif split_scheme == "chromosome_holdout":
        if dataset_name not in HUMAN_TASKS:
            raise ValueError(f"Chromosome-held-out split only supported for human tasks, got {dataset_name}.")
        train = metadata[~metadata["region"].isin(HOLDOUT_CHROMS)].copy()
        test = metadata[metadata["region"].isin(HOLDOUT_CHROMS)].copy()
    else:
        raise ValueError(f"Unknown split scheme: {split_scheme}")

    train = sample_per_label(train, train_per_label, seed=seed)
    test = sample_per_label(test, test_per_label, seed=seed + 1)
    return TaskFrame(
        dataset=dataset_name,
        split_scheme=split_scheme,
        train=read_sequences(train),
        test=read_sequences(test),
    )


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def mononucleotide_shuffle(seq: str, rng: random.Random) -> str:
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def dinucleotide_shuffle(seq: str, rng: random.Random) -> str:
    if len(seq) < 3:
        return seq
    adjacency: dict[str, list[str]] = {}
    for left, right in zip(seq[:-1], seq[1:]):
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, adjacency.get(right, []))
    shuffled_adjacency = {key: values[:] for key, values in adjacency.items()}
    for values in shuffled_adjacency.values():
        rng.shuffle(values)

    stack = [seq[0]]
    path: list[str] = []
    while stack:
        node = stack[-1]
        if shuffled_adjacency.get(node):
            stack.append(shuffled_adjacency[node].pop())
        else:
            path.append(stack.pop())
    shuffled = "".join(path[::-1])
    if len(shuffled) != len(seq):
        raise RuntimeError("Dinucleotide shuffle produced an invalid sequence length.")
    return shuffled


def create_counterfactuals(sequences: list[str], seed: int) -> dict[str, list[str]]:
    rng_mono = random.Random(seed)
    rng_dinuc = random.Random(seed + 17)
    return {
        "reverse_complement": [reverse_complement(seq) for seq in sequences],
        "mono_shuffle": [mononucleotide_shuffle(seq, rng_mono) for seq in sequences],
        "dinuc_shuffle": [dinucleotide_shuffle(seq, rng_dinuc) for seq in sequences],
    }


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins[1:-1], right=True)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray], float],
    n_items: int,
    seed: int,
    n_bootstrap: int = 200,
    ci: float = 0.95,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_items, size=n_items)
        values.append(metric_fn(indices))
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def build_kmer_logistic_regression(k: int = 6, seed: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("vectorizer", CountVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False)),
            ("classifier", LogisticRegression(max_iter=400, solver="liblinear", random_state=seed)),
        ]
    )


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[str], labels: list[int]) -> None:
        self.sequences = [torch.tensor([VOCAB.get(base, 4) for base in seq], dtype=torch.long) for seq in sequences]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index]


class SmallSequenceCNN(nn.Module):
    def __init__(self, vocab_size: int = 5, embed_dim: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.features = nn.Sequential(
            nn.Conv1d(embed_dim, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        features = self.features(embedded)
        return self.classifier(features).squeeze(-1)


def collate_sequence_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True, padding_value=VOCAB["N"])
    return padded, torch.stack(list(labels))


def make_loader(sequences: list[str], labels: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        SequenceDataset(sequences, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_sequence_batch,
    )


def predict_cnn_probabilities(model: SmallSequenceCNN, sequences: list[str], batch_size: int = 256) -> np.ndarray:
    model.eval()
    loader = make_loader(sequences, [0] * len(sequences), batch_size=batch_size, shuffle=False)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for tokens, _ in loader:
            probs = torch.sigmoid(model(tokens.to(DEVICE))).cpu().numpy()
            outputs.append(probs)
    return np.concatenate(outputs)


def train_cnn_model(
    train_sequences: list[str],
    train_labels: list[int],
    seed: int,
    rc_augment: bool = False,
    epochs: int = 6,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
) -> tuple[SmallSequenceCNN, list[dict[str, float]]]:
    set_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))

    train_sequences = list(train_sequences)
    train_labels = list(train_labels)
    if rc_augment:
        train_sequences = train_sequences + [reverse_complement(seq) for seq in train_sequences]
        train_labels = train_labels + train_labels

    tr_seq, val_seq, tr_lab, val_lab = train_test_split(
        train_sequences,
        train_labels,
        test_size=0.1,
        random_state=seed,
        stratify=train_labels,
    )

    model = SmallSequenceCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    loader = make_loader(tr_seq, tr_lab, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val_auroc = -float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for tokens, labels in loader:
            tokens = tokens.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(tokens)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_probs = predict_cnn_probabilities(model, val_seq)
        val_metrics = compute_standard_metrics(np.array(val_lab), val_probs)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "val_auroc": val_metrics["auroc"],
                "val_accuracy": val_metrics["accuracy"],
                "val_ece": val_metrics["ece"],
            }
        )
        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("CNN training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    return model, history


class DNABERT2Embedder:
    def __init__(self, model_dir: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, low_cpu_mem_usage=False)
        self.model.eval()
        self.model.to(DEVICE)

    def encode(self, sequences: list[str], batch_size: int = 16) -> np.ndarray:
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                batch = sequences[start : start + batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
                outputs = self.model(**inputs)[0]
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                batches.append(pooled.cpu().numpy())
        return np.vstack(batches)


def compute_standard_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    preds = (probs >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, preds)),
        "ece": expected_calibration_error(labels, probs),
        "brier": brier_score(labels, probs),
    }


def compute_counterfactual_metrics(
    labels: np.ndarray,
    original_probs: np.ndarray,
    perturbed_probs: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    positive_mask = labels == 1
    return {
        f"{prefix}_mean_abs_delta": float(np.abs(original_probs - perturbed_probs).mean()),
        f"{prefix}_flip_rate": float(((original_probs >= 0.5) != (perturbed_probs >= 0.5)).mean()),
        f"{prefix}_positive_prob_drop": float(original_probs[positive_mask].mean() - perturbed_probs[positive_mask].mean()),
        f"{prefix}_ece": expected_calibration_error(labels, perturbed_probs),
        f"{prefix}_brier": brier_score(labels, perturbed_probs),
    }


def evaluate_prediction_set(
    labels: np.ndarray,
    original_probs: np.ndarray,
    counterfactual_probs: dict[str, np.ndarray],
    seed: int,
) -> dict[str, float]:
    metrics = compute_standard_metrics(labels, original_probs)
    metrics["auroc_ci_low"], metrics["auroc_ci_high"] = bootstrap_ci(
        lambda idx: roc_auc_score(labels[idx], original_probs[idx]),
        n_items=len(labels),
        seed=seed,
    )
    for name, probs in counterfactual_probs.items():
        metrics.update(compute_counterfactual_metrics(labels, original_probs, probs, prefix=name))
        low, high = bootstrap_ci(
            lambda idx, p=probs: float(np.abs(original_probs[idx] - p[idx]).mean()),
            n_items=len(labels),
            seed=seed + hash(name) % 1000,
        )
        metrics[f"{name}_mean_abs_delta_ci_low"] = low
        metrics[f"{name}_mean_abs_delta_ci_high"] = high
    return metrics


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
