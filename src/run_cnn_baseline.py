from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from project_pipeline import create_counterfactuals, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results" / "cnn_baseline"
FIGURES_ROOT = PROJECT_ROOT / "figures"
DATASETS = ["human_nontata_promoters", "human_enhancers_cohn"]
SEED = 42
DEVICE = torch.device("cpu")
VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[str], labels: list[int]) -> None:
        self.sequences = [self.encode(seq) for seq in sequences]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    @staticmethod
    def encode(sequence: str) -> torch.Tensor:
        return torch.tensor([VOCAB.get(base, 4) for base in sequence], dtype=torch.long)

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


@dataclass
class TrainingArtifacts:
    model: SmallSequenceCNN
    history: list[dict[str, float]]


def make_loader(sequences: list[str], labels: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    dataset = SequenceDataset(sequences, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict_probabilities(model: SmallSequenceCNN, sequences: list[str], batch_size: int = 256) -> np.ndarray:
    model.eval()
    loader = make_loader(sequences, [0] * len(sequences), batch_size=batch_size, shuffle=False)
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for tokens, _ in loader:
            logits = model(tokens.to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
            probabilities.append(probs)
    return np.concatenate(probabilities)


def evaluate_model(model: SmallSequenceCNN, sequences: list[str], labels: list[int]) -> dict[str, float]:
    probs = predict_probabilities(model, sequences)
    preds = (probs >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, preds)),
    }


def train_model(
    train_sequences: list[str],
    train_labels: list[int],
    val_sequences: list[str],
    val_labels: list[int],
    epochs: int = 6,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
) -> TrainingArtifacts:
    model = SmallSequenceCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = make_loader(train_sequences, train_labels, batch_size=batch_size, shuffle=True)
    best_state = None
    best_val_auroc = -float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for tokens, labels in train_loader:
            tokens = tokens.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(tokens)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_metrics = evaluate_model(model, train_sequences[: min(4000, len(train_sequences))], train_labels[: min(4000, len(train_labels))])
        val_metrics = evaluate_model(model, val_sequences, val_labels)
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(epoch_losses)),
            "train_auroc_subset": train_metrics["auroc"],
            "val_auroc": val_metrics["auroc"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record))

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a model state.")
    model.load_state_dict(best_state)
    return TrainingArtifacts(model=model, history=history)


def summarize_counterfactuals(
    model: SmallSequenceCNN,
    sequences: list[str],
    labels: list[int],
) -> dict[str, float]:
    counterfactuals = create_counterfactuals(sequences, seed=SEED)
    original_probs = predict_probabilities(model, sequences)
    rc_probs = predict_probabilities(model, counterfactuals["reverse_complement"])
    shuffle_probs = predict_probabilities(model, counterfactuals["gc_shuffle"])
    labels_array = np.array(labels)
    positive_mask = labels_array == 1
    negative_mask = labels_array == 0

    return {
        "reverse_complement_mean_abs_delta": float(np.abs(original_probs - rc_probs).mean()),
        "reverse_complement_prediction_flip_rate": float(((original_probs >= 0.5) != (rc_probs >= 0.5)).mean()),
        "positive_original_mean_prob": float(original_probs[positive_mask].mean()),
        "positive_gc_shuffle_mean_prob": float(shuffle_probs[positive_mask].mean()),
        "positive_gc_shuffle_prob_drop": float(original_probs[positive_mask].mean() - shuffle_probs[positive_mask].mean()),
        "negative_original_mean_prob": float(original_probs[negative_mask].mean()),
        "negative_gc_shuffle_mean_prob": float(shuffle_probs[negative_mask].mean()),
    }


def plot_training_history(history: list[dict[str, float]], dataset_name: str) -> None:
    frame = np.array([[row["epoch"], row["train_loss"], row["val_auroc"]] for row in history])
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(frame[:, 0], frame[:, 1], marker="o", color="#204b87")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="#204b87")
    ax1.tick_params(axis="y", labelcolor="#204b87")

    ax2 = ax1.twinx()
    ax2.plot(frame[:, 0], frame[:, 2], marker="s", color="#4c9f70")
    ax2.set_ylabel("Validation AUROC", color="#4c9f70")
    ax2.tick_params(axis="y", labelcolor="#4c9f70")

    fig.tight_layout()
    path = FIGURES_ROOT / f"{dataset_name}_cnn_training_curve.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    model_rows: list[dict[str, float | str]] = []
    counterfactual_rows: list[dict[str, float | str]] = []

    for dataset_name in DATASETS:
        bundle = load_dataset(dataset_name, DATA_ROOT)
        train_sequences, val_sequences, train_labels, val_labels = train_test_split(
            bundle.train_sequences,
            bundle.train_labels,
            test_size=0.1,
            random_state=SEED,
            stratify=bundle.train_labels,
        )

        artifacts = train_model(train_sequences, train_labels, val_sequences, val_labels)
        plot_training_history(artifacts.history, dataset_name)

        test_metrics = evaluate_model(artifacts.model, bundle.test_sequences, bundle.test_labels)
        cf_metrics = summarize_counterfactuals(artifacts.model, bundle.test_sequences, bundle.test_labels)

        payload = {
            "dataset": dataset_name,
            "model": "small_sequence_cnn",
            "test_auroc": test_metrics["auroc"],
            "test_accuracy": test_metrics["accuracy"],
            **cf_metrics,
        }
        model_rows.append(
            {
                "dataset": dataset_name,
                "model": "small_sequence_cnn",
                "test_auroc": test_metrics["auroc"],
                "test_accuracy": test_metrics["accuracy"],
            }
        )
        counterfactual_rows.append({"dataset": dataset_name, **cf_metrics})

        (RESULTS_ROOT / f"{dataset_name}_history.json").write_text(json.dumps(artifacts.history, indent=2))
        (RESULTS_ROOT / f"{dataset_name}_metrics.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))

    pd.DataFrame(model_rows).to_csv(RESULTS_ROOT / "model_metrics.csv", index=False)
    pd.DataFrame(counterfactual_rows).to_csv(RESULTS_ROOT / "counterfactual_metrics.csv", index=False)


if __name__ == "__main__":
    main()
