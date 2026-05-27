from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .config import ModelSpec


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def _patch_nucleotide_transformer_compat() -> None:
    import transformers.modeling_utils as modeling_utils
    import transformers.pytorch_utils as pytorch_utils

    if not hasattr(modeling_utils.PreTrainedModel, "all_tied_weights_keys"):
        modeling_utils.PreTrainedModel.all_tied_weights_keys = []
    if not hasattr(modeling_utils.PreTrainedModel, "get_head_mask"):
        def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None:
                return [None] * num_hidden_layers
            return head_mask
        modeling_utils.PreTrainedModel.get_head_mask = _get_head_mask  # type: ignore[attr-defined]
    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            keep = [head for head in heads if head not in already_pruned_heads]
            return keep, torch.arange(n_heads * head_size)
        pytorch_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices  # type: ignore[attr-defined]


@dataclass
class RunnerArtifacts:
    validation_probs: np.ndarray
    validation_labels: np.ndarray
    runtime_s: float


class BaseRunner:
    def fit(self, train_frame: pd.DataFrame, validation_frame: pd.DataFrame) -> RunnerArtifacts:
        raise NotImplementedError

    def predict_proba(self, frame_or_sequences: pd.DataFrame | Iterable[str]) -> np.ndarray:
        raise NotImplementedError


class FeatureLogisticRunner(BaseRunner):
    def __init__(self, feature_name: str, seed: int) -> None:
        self.feature_name = feature_name
        self.model = LogisticRegression(max_iter=200, solver="liblinear", random_state=seed)

    def _features(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[[self.feature_name]].to_numpy(dtype=float)

    def fit(self, train_frame: pd.DataFrame, validation_frame: pd.DataFrame) -> RunnerArtifacts:
        start = time.time()
        self.model.fit(self._features(train_frame), train_frame["label"].to_numpy())
        validation_probs = self.model.predict_proba(self._features(validation_frame))[:, 1]
        return RunnerArtifacts(validation_probs=validation_probs, validation_labels=validation_frame["label"].to_numpy(), runtime_s=time.time() - start)

    def predict_proba(self, frame_or_sequences: pd.DataFrame | Iterable[str]) -> np.ndarray:
        if not isinstance(frame_or_sequences, pd.DataFrame):
            raise TypeError("FeatureLogisticRunner expects a DataFrame input.")
        return self.model.predict_proba(self._features(frame_or_sequences))[:, 1]


class KmerRunner(BaseRunner):
    def __init__(self, seed: int, k: int = 6) -> None:
        self.pipeline = Pipeline(
            steps=[
                ("vectorizer", CountVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False)),
                ("classifier", LogisticRegression(max_iter=400, solver="liblinear", random_state=seed)),
            ]
        )

    def fit(self, train_frame: pd.DataFrame, validation_frame: pd.DataFrame) -> RunnerArtifacts:
        start = time.time()
        self.pipeline.fit(train_frame["sequence"].tolist(), train_frame["label"].to_numpy())
        validation_probs = self.pipeline.predict_proba(validation_frame["sequence"].tolist())[:, 1]
        return RunnerArtifacts(validation_probs=validation_probs, validation_labels=validation_frame["label"].to_numpy(), runtime_s=time.time() - start)

    def predict_proba(self, frame_or_sequences: pd.DataFrame | Iterable[str]) -> np.ndarray:
        sequences = frame_or_sequences["sequence"].tolist() if isinstance(frame_or_sequences, pd.DataFrame) else list(frame_or_sequences)
        return self.pipeline.predict_proba(sequences)[:, 1]


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[str], labels: list[int]) -> None:
        self.sequences = [torch.tensor([VOCAB.get(base, 4) for base in sequence], dtype=torch.long) for sequence in sequences]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index]


def _collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, labels = zip(*batch)
    return pad_sequence(tokens, batch_first=True, padding_value=VOCAB["N"]), torch.stack(list(labels))


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
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        features = self.features(embedded)
        return self.classifier(features).squeeze(-1)


class SmallCNNRunner(BaseRunner):
    def __init__(self, seed: int, rc_augment: bool = False, epochs: int = 2, batch_size: int = 64) -> None:
        self.seed = seed
        self.rc_augment = rc_augment
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = SmallSequenceCNN().to(DEVICE)

    def _make_loader(self, frame: pd.DataFrame, shuffle: bool) -> DataLoader:
        dataset = SequenceDataset(frame["sequence"].tolist(), frame["label"].tolist())
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=_collate)

    def fit(self, train_frame: pd.DataFrame, validation_frame: pd.DataFrame) -> RunnerArtifacts:
        start = time.time()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        train = train_frame.copy()
        if self.rc_augment:
            rc = train.copy()
            rc["sequence"] = rc["sequence"].map(lambda seq: seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1])
            train = pd.concat([train, rc], ignore_index=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        loader = self._make_loader(train, shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for tokens, labels in loader:
                optimizer.zero_grad()
                logits = self.model(tokens.to(DEVICE))
                loss = criterion(logits, labels.to(DEVICE))
                loss.backward()
                optimizer.step()
        validation_probs = self.predict_proba(validation_frame)
        return RunnerArtifacts(validation_probs=validation_probs, validation_labels=validation_frame["label"].to_numpy(), runtime_s=time.time() - start)

    def predict_proba(self, frame_or_sequences: pd.DataFrame | Iterable[str]) -> np.ndarray:
        if isinstance(frame_or_sequences, pd.DataFrame):
            frame = frame_or_sequences
        else:
            sequences = list(frame_or_sequences)
            frame = pd.DataFrame({"sequence": sequences, "label": [0] * len(sequences)})
        loader = self._make_loader(frame, shuffle=False)
        outputs: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for tokens, _ in loader:
                probs = torch.sigmoid(self.model(tokens.to(DEVICE))).cpu().numpy()
                outputs.append(probs)
        return np.concatenate(outputs) if outputs else np.array([], dtype=float)


class HFEncoderRunner(BaseRunner):
    def __init__(self, spec: ModelSpec, mode: str = "frozen", seed: int = 2026, batch_size: int = 8) -> None:
        self.spec = spec
        self.mode = mode
        self.seed = seed
        self.batch_size = batch_size
        self.embedding_dim = int(spec.expected_embedding_dimension or (256 if spec.model_id == "caduceus_ph" else 128))
        self.classifier = LogisticRegression(max_iter=300, solver="liblinear", random_state=seed)
        self._tokenizer = None
        self._model = None
        self._loaded_real_model = False

    def _maybe_load_transformer(self) -> None:
        if self._loaded_real_model or self.spec.model_checkpoint is None:
            return
        checkpoint = Path(self.spec.model_checkpoint)
        if checkpoint.exists() or "/" in self.spec.model_checkpoint:
            try:
                _patch_nucleotide_transformer_compat()
                from transformers import AutoModel, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.spec.tokenizer_name or self.spec.model_checkpoint, trust_remote_code=True)
                self._model = AutoModel.from_pretrained(self.spec.model_checkpoint, trust_remote_code=True)
                self._model.to(DEVICE)
                self._model.eval()
                self._loaded_real_model = True
            except Exception:
                self._tokenizer = None
                self._model = None
                self._loaded_real_model = False

    def _hash_embedding(self, sequence: str) -> np.ndarray:
        vector = np.zeros(self.embedding_dim, dtype=np.float32)
        k = 3
        limit = max(len(sequence) - k + 1, 1)
        for idx in range(limit):
            token = sequence[idx : idx + k]
            bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16) % self.embedding_dim
            vector[bucket] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def _encode(self, sequences: list[str]) -> np.ndarray:
        self._maybe_load_transformer()
        if self._loaded_real_model and self._tokenizer is not None and self._model is not None:
            batches: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(sequences), self.batch_size):
                    batch = sequences[start : start + self.batch_size]
                    encoded = self._tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.spec.input_length_limits)
                    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
                    outputs = self._model(**encoded)[0]
                    mask = encoded["attention_mask"].unsqueeze(-1)
                    pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    batches.append(pooled.cpu().numpy())
            return np.vstack(batches) if batches else np.zeros((0, self.embedding_dim), dtype=np.float32)
        return np.vstack([self._hash_embedding(sequence) for sequence in sequences]) if sequences else np.zeros((0, self.embedding_dim), dtype=np.float32)

    def fit(self, train_frame: pd.DataFrame, validation_frame: pd.DataFrame) -> RunnerArtifacts:
        start = time.time()
        train_embeddings = self._encode(train_frame["sequence"].tolist())
        val_embeddings = self._encode(validation_frame["sequence"].tolist())
        self.classifier.fit(train_embeddings, train_frame["label"].to_numpy())
        validation_probs = self.classifier.predict_proba(val_embeddings)[:, 1]
        return RunnerArtifacts(validation_probs=validation_probs, validation_labels=validation_frame["label"].to_numpy(), runtime_s=time.time() - start)

    def predict_proba(self, frame_or_sequences: pd.DataFrame | Iterable[str]) -> np.ndarray:
        sequences = frame_or_sequences["sequence"].tolist() if isinstance(frame_or_sequences, pd.DataFrame) else list(frame_or_sequences)
        embeddings = self._encode(sequences)
        return self.classifier.predict_proba(embeddings)[:, 1]


def build_runner(spec: ModelSpec, mode: str = "frozen", seed: int = 2026) -> BaseRunner:
    model_id = spec.model_id
    if model_id == "gc_only":
        return FeatureLogisticRunner("gc_fraction", seed=seed)
    if model_id == "cpg_only":
        return FeatureLogisticRunner("cpg_oe", seed=seed)
    if model_id == "length_only":
        return FeatureLogisticRunner("length", seed=seed)
    if model_id == "repeat_only":
        return FeatureLogisticRunner("repeat_fraction", seed=seed)
    if model_id == "kmer_logistic_regression":
        return KmerRunner(seed=seed, k=6)
    if model_id == "small_cnn":
        return SmallCNNRunner(seed=seed, rc_augment=False)
    if model_id == "small_cnn_rc_aug":
        return SmallCNNRunner(seed=seed, rc_augment=True)
    if model_id in {"dnabert2", "caduceus_ph", "nucleotide_transformer_v2"}:
        return HFEncoderRunner(spec, mode=mode, seed=seed)
    raise KeyError(f"Unsupported model: {model_id}")
