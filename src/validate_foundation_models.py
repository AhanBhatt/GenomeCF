from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genomecf.config import get_model_spec, get_split_spec, get_task_spec
from genomecf.data import build_split_frames, load_task_frame
from genomecf.models import HFEncoderRunner

RESULTS_ROOT = PROJECT_ROOT / "results" / "release"
FOCAL_TASKS = ["human_nontata_promoters", "human_enhancers_cohn"]


def sample_background_sequence(length: int, gc_fraction: float, rng: random.Random) -> str:
    sequence = []
    for _ in range(length):
        if rng.random() < gc_fraction:
            sequence.append(rng.choice(["G", "C"]))
        else:
            sequence.append(rng.choice(["A", "T"]))
    return "".join(sequence)


def build_gc_sanity_dataset(seed: int, length: int = 200, n_per_class: int = 128) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    train_rows = []
    test_rows = []
    for split_rows in [train_rows, test_rows]:
        for label, gc in [(1, 0.75), (0, 0.25)]:
            for _ in range(n_per_class):
                split_rows.append({"sequence": sample_background_sequence(length, gc, rng), "label": label})
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def balanced_subset(frame: pd.DataFrame, n_per_label: int, seed: int) -> pd.DataFrame:
    pieces = []
    for label in sorted(frame["label"].unique()):
        subset = frame[frame["label"] == label]
        pieces.append(subset.sample(n=min(n_per_label, len(subset)), random_state=seed))
    return pd.concat(pieces, ignore_index=True)


def logistic_probe(emb_train: np.ndarray, y_train: np.ndarray, emb_test: np.ndarray, y_test: np.ndarray, seed: int) -> float:
    model = LogisticRegression(max_iter=400, solver="liblinear", random_state=seed)
    model.fit(emb_train, y_train)
    probs = model.predict_proba(emb_test)[:, 1]
    return float(roc_auc_score(y_test, probs))


def validate_model(model_id: str) -> dict[str, object]:
    spec = get_model_spec(model_id)
    runner = HFEncoderRunner(spec, mode="frozen", seed=42, batch_size=8)
    runner._load_encoder()
    assert runner.tokenizer is not None
    assert runner.model is not None

    sample_sequences = [
        "ACGT" * 32,
        "TGCATGCA" * 16,
        "GATTACA" * 18,
        "CCCCGGGGAAAATTTT" * 8,
    ]
    tokens = runner.tokenizer(
        sample_sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=spec.max_length,
    )
    input_ids = tokens["input_ids"].cpu().numpy()
    pad_id = getattr(runner.tokenizer, "pad_token_id", None)
    unk_id = getattr(runner.tokenizer, "unk_token_id", None)
    pad_fraction = float(np.mean(input_ids == pad_id)) if pad_id is not None else 0.0
    unk_fraction = float(np.mean(input_ids == unk_id)) if unk_id is not None else 0.0

    long_sequence = ["ACGT" * ((spec.max_length or 1024) // 4 + 300)]
    long_tokens = runner.tokenizer(
        long_sequence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=spec.max_length,
    )
    truncation_applied = bool(spec.max_length and int(long_tokens["input_ids"].shape[1]) == int(spec.max_length))

    embeddings_a = runner._encode(sample_sequences)
    embeddings_b = runner._encode(sample_sequences)
    reproducible = bool(np.allclose(embeddings_a, embeddings_b, atol=1e-6))
    embedding_variance = float(np.var(embeddings_a))

    task_checks = []
    official_spec = get_split_spec("official")
    for task_name in FOCAL_TASKS:
        frame = load_task_frame(get_task_spec(task_name), PROJECT_ROOT)
        splits = build_split_frames(frame, official_spec, seed=2026)
        train = balanced_subset(splits["train"], n_per_label=128, seed=11)
        test = balanced_subset(splits["test"], n_per_label=128, seed=13)
        emb_train = runner._encode(train["sequence"].tolist())
        emb_test = runner._encode(test["sequence"].tolist())
        auroc_1 = logistic_probe(emb_train, train["label"].to_numpy(), emb_test, test["label"].to_numpy(), seed=17)
        auroc_2 = logistic_probe(emb_train, train["label"].to_numpy(), emb_test, test["label"].to_numpy(), seed=17)
        task_checks.append(
            {
                "task_id": task_name,
                "train_positive": int((train["label"] == 1).sum()),
                "train_negative": int((train["label"] == 0).sum()),
                "test_positive": int((test["label"] == 1).sum()),
                "test_negative": int((test["label"] == 0).sum()),
                "max_eval_length": int(max(train["length"].max(), test["length"].max())),
                "max_length_under_limit": bool((spec.max_length is None) or (max(train["length"].max(), test["length"].max()) <= spec.max_length)),
                "embedding_dim": int(emb_train.shape[1]),
                "probe_auroc": auroc_1,
                "probe_reproducible": bool(abs(auroc_1 - auroc_2) < 1e-9),
            }
        )

    gc_train, gc_test = build_gc_sanity_dataset(seed=29)
    gc_emb_train = runner._encode(gc_train["sequence"].tolist())
    gc_emb_test = runner._encode(gc_test["sequence"].tolist())
    gc_auroc = logistic_probe(gc_emb_train, gc_train["label"].to_numpy(), gc_emb_test, gc_test["label"].to_numpy(), seed=31)

    passed = all(
        [
            embeddings_a.shape[0] == len(sample_sequences),
            embeddings_a.shape == embeddings_b.shape,
            embedding_variance > 1e-8,
            reproducible,
            unk_fraction < 0.25,
            gc_auroc > 0.75,
            all(item["probe_reproducible"] for item in task_checks),
            all(item["embedding_dim"] == embeddings_a.shape[1] for item in task_checks),
        ]
    )

    return {
        "model_id": model_id,
        "validated": passed,
        "checkpoint": str(spec.local_model_path or spec.hf_model_id or ""),
        "tokenizer_class": runner.tokenizer.__class__.__name__,
        "model_class": runner.model.__class__.__name__,
        "max_length": spec.max_length,
        "pooling": spec.pooler or "mean",
        "tokenizer_has_attention_mask": "attention_mask" in tokens,
        "pad_fraction": pad_fraction,
        "unk_fraction": unk_fraction,
        "truncation_applied_on_long_sequence": truncation_applied,
        "sample_embedding_shape": list(embeddings_a.shape),
        "sample_embedding_variance": embedding_variance,
        "sample_embeddings_reproducible": reproducible,
        "gc_sanity_auroc": gc_auroc,
        "task_checks": task_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen foundation-model embedding pipelines.")
    parser.add_argument("--model", default="nucleotide_transformer_v2")
    parser.add_argument("--output", type=Path, default=RESULTS_ROOT / "nt_validation_report.json")
    args = parser.parse_args()

    report = validate_model(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
