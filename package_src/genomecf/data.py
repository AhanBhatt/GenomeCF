from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SplitSpec, TaskSpec, get_split_spec
from .paths import CACHE_ROOT, DATA_ROOT, LOCAL_RUNTIME_ROOT, PROJECT_ROOT


CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
CHROM_FOLD_MAP = {
    "A": {"chr1", "chr2"},
    "B": {"chr3", "chr4", "chr5"},
    "C": {"chr6", "chr7", "chr8"},
    "D": {"chr9", "chr10", "chr11", "chr12"},
    "E": {"chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY"},
}
SYNTHETIC_TASKS = {
    "gc_correlated",
    "gc_matched",
    "gc_conflict",
    "two_motif_grammar",
    "motif_position_conflict",
}


def _stable_int(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)


def _sequence_cache_path(task_id: str) -> Path:
    return CACHE_ROOT / "task_frames" / f"{task_id}.csv"


def _cpg_oe(sequence: str) -> float:
    sequence = sequence.upper()
    c = sequence.count("C")
    g = sequence.count("G")
    observed = sum(1 for idx in range(len(sequence) - 1) if sequence[idx : idx + 2] == "CG")
    expected = (c * g) / max(len(sequence), 1)
    if expected == 0:
        return 0.0
    return float(observed / expected)


def _repeat_fraction(sequence: str) -> float:
    matches = re.finditer(r"((A{3,})|(C{3,})|(G{3,})|(T{3,})|(N{3,}))", sequence.upper())
    covered = set()
    for match in matches:
        covered.update(range(match.start(), match.end()))
    return float(len(covered) / max(len(sequence), 1))


def annotate_sequence_confounders(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["sequence"] = enriched["sequence"].astype(str).str.upper()
    if "id" not in enriched.columns:
        enriched["id"] = [f"seq_{idx}" for idx in range(len(enriched))]
    if "length" not in enriched.columns:
        enriched["length"] = enriched["sequence"].str.len()
    if "gc_fraction" not in enriched.columns:
        enriched["gc_fraction"] = enriched["sequence"].map(lambda seq: float((seq.count("G") + seq.count("C")) / max(len(seq), 1)))
    if "cpg_oe" not in enriched.columns:
        enriched["cpg_oe"] = enriched["sequence"].map(_cpg_oe)
    if "n_fraction" not in enriched.columns:
        enriched["n_fraction"] = enriched["sequence"].map(lambda seq: float(seq.count("N") / max(len(seq), 1)))
    if "repeat_fraction" not in enriched.columns:
        enriched["repeat_fraction"] = enriched["sequence"].map(_repeat_fraction)
    if "region" not in enriched.columns:
        enriched["region"] = enriched["id"].map(lambda value: CHROMS[_stable_int(str(value)) % len(CHROMS)])
    else:
        enriched["region"] = enriched["region"].fillna("").replace("", np.nan)
        missing = enriched["region"].isna()
        if missing.any():
            enriched.loc[missing, "region"] = enriched.loc[missing, "id"].map(lambda value: CHROMS[_stable_int(str(value)) % len(CHROMS)])
    if "chromosome_fold" not in enriched.columns:
        enriched["chromosome_fold"] = enriched["region"].map(_chromosome_to_fold)
    else:
        missing = enriched["chromosome_fold"].isna() | (enriched["chromosome_fold"].astype(str) == "")
        if missing.any():
            enriched.loc[missing, "chromosome_fold"] = enriched.loc[missing, "region"].map(_chromosome_to_fold)
    if "tss_distance" not in enriched.columns:
        enriched["tss_distance"] = enriched["id"].map(lambda value: int(_stable_int(str(value)) % 100_000))
    if "orig_split" not in enriched.columns:
        enriched["orig_split"] = "train"
    return enriched.reset_index(drop=True)


def _chromosome_to_fold(region: str) -> str:
    region = str(region)
    for fold, chroms in CHROM_FOLD_MAP.items():
        if region in chroms:
            return fold
    return "E"


def _resolve_data_root(task_spec: TaskSpec, project_root: Path) -> Path:
    candidates = [project_root / "data", LOCAL_RUNTIME_ROOT / "data", DATA_ROOT]
    seen: set[Path] = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        if (
            (root / task_spec.task_id / "train").exists()
            or (root / "gue" / task_spec.task_id).exists()
            or (root / "mavedb" / task_spec.task_id).exists()
        ):
            return root
    return project_root / "data"


def _load_txt_task(task_spec: TaskSpec, data_root: Path) -> pd.DataFrame:
    base = data_root / task_spec.task_id
    rows: list[dict[str, object]] = []
    for split_name in ["train", "test"]:
        for label_name, label_value in [("negative", 0), ("positive", 1)]:
            label_dir = base / split_name / label_name
            for file_path in sorted(label_dir.glob("*.txt")):
                rows.append(
                    {
                        "id": file_path.stem,
                        "orig_split": split_name,
                        "label": label_value,
                        "sequence": file_path.read_text(encoding="utf-8").strip().upper(),
                        "task_id": task_spec.task_id,
                    }
                )
    frame = pd.DataFrame(rows)
    return annotate_sequence_confounders(frame)


def _load_csv_task(task_spec: TaskSpec, base: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for split_name in ["train", "validation", "test"]:
        path = base / f"{split_name}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "orig_split" not in frame.columns:
            frame["orig_split"] = split_name
        if "id" not in frame.columns:
            frame["id"] = [f"{task_spec.task_id}_{split_name}_{idx}" for idx in range(len(frame))]
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No split CSV files found for {task_spec.task_id} at {base}")
    return annotate_sequence_confounders(pd.concat(rows, ignore_index=True))


def _sample_background(length: int, gc_fraction: float, rng: random.Random) -> str:
    return "".join(rng.choice(["G", "C"]) if rng.random() < gc_fraction else rng.choice(["A", "T"]) for _ in range(length))


def _plant(sequence: str, motif: str, start: int) -> str:
    return sequence[:start] + motif + sequence[start + len(motif) :]


def _sample_without_motif(length: int, gc_fraction: float, motif: str, rng: random.Random) -> str:
    while True:
        sequence = _sample_background(length, gc_fraction, rng)
        if motif not in sequence:
            return sequence


def _generate_synthetic_task(task_id: str, seed: int = 2026) -> pd.DataFrame:
    rng = random.Random(seed)
    seq_len = 200
    motif_a = "CACGTG"
    motif_b = "GGAA"
    rows: list[dict[str, object]] = []
    if task_id == "gc_conflict":
        specs = {
            "train": [(1, 2500, 0.70), (0, 2500, 0.30)],
            "validation": [(1, 500, 0.70), (0, 500, 0.30)],
            "test": [(1, 1000, 0.30), (0, 1000, 0.70)],
        }
        for split_name, configs in specs.items():
            for label, count, gc_fraction in configs:
                for idx in range(count):
                    seq = _plant(_sample_background(seq_len, gc_fraction, rng), motif_a, rng.randint(40, 140)) if label == 1 else _sample_without_motif(seq_len, gc_fraction, motif_a, rng)
                    rows.append({"id": f"{task_id}_{split_name}_{label}_{idx}", "orig_split": split_name, "label": label, "sequence": seq, "task_id": task_id, "shortcut_label": int(gc_fraction > 0.5)})
    elif task_id in {"gc_correlated", "gc_matched"}:
        positive_gc, negative_gc = (0.70, 0.30) if task_id == "gc_correlated" else (0.50, 0.50)
        for split_name, count in {"train": 2500, "validation": 500, "test": 1000}.items():
            for label in [0, 1]:
                gc_fraction = positive_gc if label == 1 else negative_gc
                for idx in range(count):
                    seq = _plant(_sample_background(seq_len, gc_fraction, rng), motif_a, rng.randint(40, 140)) if label == 1 else _sample_without_motif(seq_len, gc_fraction, motif_a, rng)
                    rows.append({"id": f"{task_id}_{split_name}_{label}_{idx}", "orig_split": split_name, "label": label, "sequence": seq, "task_id": task_id, "shortcut_label": int(gc_fraction > 0.5)})
    elif task_id == "two_motif_grammar":
        for split_name, count in {"train": 2500, "validation": 500, "test": 1000}.items():
            for label in [0, 1]:
                for idx in range(count):
                    background = _sample_background(seq_len, 0.5, rng)
                    if label == 1:
                        start_a = rng.randint(30, 60)
                        start_b = start_a + 18
                        seq = _plant(_plant(background, motif_a, start_a), motif_b, start_b)
                    else:
                        start_a = rng.randint(30, 60)
                        if idx % 2 == 0:
                            seq = _plant(_plant(background, motif_a, start_a), motif_b, start_a + 8)
                        else:
                            seq = _plant(background, motif_a, start_a)
                    rows.append({"id": f"{task_id}_{split_name}_{label}_{idx}", "orig_split": split_name, "label": label, "sequence": seq, "task_id": task_id, "shortcut_label": int(motif_b in seq)})
    elif task_id == "motif_position_conflict":
        for split_name, count in {"train": 2500, "validation": 500, "test": 1000}.items():
            for label in [0, 1]:
                for idx in range(count):
                    background = _sample_background(seq_len, 0.5, rng)
                    start_positive, start_negative = (25, 95) if split_name == "test" else (95, 25)
                    if label == 1:
                        seq = _plant(background, motif_a, start_positive)
                        shortcut = int(start_positive == 95)
                    else:
                        seq = _plant(background, motif_b, start_negative)
                        shortcut = int(start_negative == 95)
                    rows.append({"id": f"{task_id}_{split_name}_{label}_{idx}", "orig_split": split_name, "label": label, "sequence": seq, "task_id": task_id, "shortcut_label": shortcut})
    else:
        raise KeyError(f"Unsupported synthetic task: {task_id}")
    return annotate_sequence_confounders(pd.DataFrame(rows))


def load_task_frame(task_spec: TaskSpec, project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    cache_path = _sequence_cache_path(task_spec.task_id)
    if cache_path.exists():
        return pd.read_csv(cache_path)

    data_root = _resolve_data_root(task_spec, project_root)
    if task_spec.task_id in SYNTHETIC_TASKS:
        frame = _generate_synthetic_task(task_spec.task_id)
    elif (data_root / task_spec.task_id / "train").exists():
        frame = _load_txt_task(task_spec, data_root)
    elif (data_root / "gue" / task_spec.task_id).exists():
        frame = _load_csv_task(task_spec, data_root / "gue" / task_spec.task_id)
    elif (data_root / "mavedb" / task_spec.task_id).exists():
        frame = _load_csv_task(task_spec, data_root / "mavedb" / task_spec.task_id)
    else:
        raise FileNotFoundError(f"Unable to locate task data for {task_spec.task_id}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def _sample_per_label(frame: pd.DataFrame, n_per_label: int | None, seed: int, *, reset: bool = True) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True) if reset else frame.copy()
    if n_per_label is None:
        return frame.reset_index(drop=True) if reset else frame.copy()
    parts: list[pd.DataFrame] = []
    for label_value in sorted(frame["label"].unique()):
        subset = frame[frame["label"] == label_value]
        if len(subset) <= n_per_label:
            parts.append(subset)
        else:
            parts.append(subset.sample(n=n_per_label, random_state=seed))
    sampled = pd.concat(parts, axis=0)
    return sampled.reset_index(drop=True) if reset else sampled


def _official_split(frame: pd.DataFrame, split_spec: SplitSpec, seed: int) -> dict[str, pd.DataFrame]:
    if set(frame["orig_split"].unique()) >= {"train", "validation", "test"}:
        train = frame[frame["orig_split"] == "train"].copy()
        validation = frame[frame["orig_split"] == "validation"].copy()
        test = frame[frame["orig_split"] == "test"].copy()
    else:
        train_pool = frame[frame["orig_split"] == "train"].copy()
        test = frame[frame["orig_split"] == "test"].copy()
        if train_pool.empty or test.empty:
            train_pool, test = train_test_split(frame, test_size=0.25, random_state=seed, stratify=frame["label"])
        min_per_label = int(train_pool["label"].value_counts().min()) if not train_pool.empty else 0
        if split_spec.val_per_label is not None and min_per_label > split_spec.val_per_label:
            validation = _sample_per_label(train_pool, split_spec.val_per_label, seed=seed + 1, reset=False)
        else:
            n_classes = int(train_pool["label"].nunique())
            validation_count = max(n_classes, int(round(len(train_pool) * 0.2)))
            validation_count = min(validation_count, max(n_classes, len(train_pool) - n_classes))
            validation, _ = train_test_split(
                train_pool,
                test_size=validation_count / max(len(train_pool), 1),
                random_state=seed + 1,
                stratify=train_pool["label"],
            )
        train = train_pool.drop(index=validation.index, errors="ignore")
    train = _sample_per_label(train, split_spec.train_per_label, seed=seed + 2)
    validation = _sample_per_label(validation, split_spec.val_per_label, seed=seed + 3)
    test = _sample_per_label(test, split_spec.test_per_label, seed=seed + 4)
    return {"train": train.reset_index(drop=True), "validation": validation.reset_index(drop=True), "test": test.reset_index(drop=True)}


def _match_negative_indices(frame: pd.DataFrame, split_spec: SplitSpec, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    positives = frame[frame["label"] == 1].copy()
    negatives = frame[frame["label"] == 0].copy()
    used: set[int] = set()
    matches: list[tuple[int, int]] = []
    same_region = bool(split_spec.matching_rules.get("same_chromosome_or_fold", False))
    gc_tol = float(split_spec.matching_rules.get("gc_tolerance", 0.01))
    len_tol = float(split_spec.matching_rules.get("length_tolerance", 10))
    repeat_tol = float(split_spec.matching_rules.get("repeat_tolerance", 0.05))
    cpg_tol = float(split_spec.matching_rules.get("cpg_tolerance", 0.25))
    for pos_index, pos_row in positives.iterrows():
        candidates = negatives[~negatives.index.isin(used)].copy()
        if same_region and "region" in candidates.columns:
            same = candidates[candidates["region"] == pos_row.get("region")]
            if not same.empty:
                candidates = same
        if "gc_fraction" in candidates.columns:
            subset = candidates[np.abs(candidates["gc_fraction"] - pos_row.get("gc_fraction", 0.0)) <= gc_tol]
            if not subset.empty:
                candidates = subset
        if "length" in candidates.columns:
            subset = candidates[np.abs(candidates["length"] - pos_row.get("length", 0.0)) <= len_tol]
            if not subset.empty:
                candidates = subset
        if "repeat_fraction" in candidates.columns:
            subset = candidates[np.abs(candidates["repeat_fraction"] - pos_row.get("repeat_fraction", 0.0)) <= repeat_tol]
            if not subset.empty:
                candidates = subset
        if "cpg_oe" in candidates.columns:
            subset = candidates[np.abs(candidates["cpg_oe"] - pos_row.get("cpg_oe", 0.0)) <= cpg_tol]
            if not subset.empty:
                candidates = subset
        if candidates.empty:
            continue
        candidates = candidates.assign(
            _distance=(
                np.abs(candidates["gc_fraction"] - pos_row.get("gc_fraction", 0.0))
                + np.abs(candidates["length"] - pos_row.get("length", 0.0)) / 1000.0
                + np.abs(candidates["repeat_fraction"] - pos_row.get("repeat_fraction", 0.0))
            )
        ).sort_values(["_distance", "id"])
        choice = candidates.iloc[rng.randrange(min(3, len(candidates)))]
        used.add(int(choice.name))
        matches.append((int(pos_index), int(choice.name)))
    return matches


def matched_negative_subset(frame: pd.DataFrame, split_spec: SplitSpec, seed: int = 2026) -> pd.DataFrame:
    frame = annotate_sequence_confounders(frame)
    matches = _match_negative_indices(frame, split_spec, seed=seed)
    selected = [pair[0] for pair in matches] + [pair[1] for pair in matches]
    if not selected:
        return frame.copy().reset_index(drop=True)
    return frame.loc[selected].reset_index(drop=True)


def matched_negative_replacement_sequences(frame: pd.DataFrame, split_spec: SplitSpec, seed: int = 2026) -> list[str]:
    frame = annotate_sequence_confounders(frame)
    replacements = frame["sequence"].tolist()
    for pos_idx, neg_idx in _match_negative_indices(frame, split_spec, seed=seed):
        replacements[pos_idx] = str(frame.loc[neg_idx, "sequence"])
    return replacements


def build_split_frames(frame: pd.DataFrame, split_spec: SplitSpec, seed: int = 2026, split_fold: str | None = None) -> dict[str, pd.DataFrame]:
    frame = annotate_sequence_confounders(frame)
    if split_spec.kind == "official":
        return _official_split(frame, split_spec, seed)
    if split_spec.kind == "synthetic":
        return _official_split(frame, split_spec, seed)
    if split_spec.kind == "matched_test":
        base = _official_split(frame, get_split_spec("official"), seed)
        base["test"] = matched_negative_subset(base["test"], split_spec, seed=seed)
        return base
    if split_spec.kind == "chromosome_holdout":
        test_chroms = set(split_spec.chrom_folds.get("test", []))
        test = frame[frame["region"].isin(test_chroms)].copy()
        train_pool = frame[~frame["region"].isin(test_chroms)].copy()
        validation = _sample_per_label(train_pool, split_spec.val_per_label, seed=seed + 1)
        train = train_pool.drop(index=validation.index, errors="ignore")
        train = _sample_per_label(train, split_spec.train_per_label, seed=seed + 2)
        test = _sample_per_label(test, split_spec.test_per_label, seed=seed + 3)
        return {"train": train.reset_index(drop=True), "validation": validation.reset_index(drop=True), "test": test.reset_index(drop=True)}
    if split_spec.kind == "chromosome_cv":
        fold_order = list(split_spec.chrom_folds.keys())
        test_fold = split_fold or fold_order[0]
        validation_fold = fold_order[(fold_order.index(test_fold) + 1) % len(fold_order)]
        test = frame[frame["chromosome_fold"] == test_fold].copy()
        validation = frame[frame["chromosome_fold"] == validation_fold].copy()
        train = frame[~frame["chromosome_fold"].isin({test_fold, validation_fold})].copy()
        train = _sample_per_label(train, split_spec.train_per_label, seed=seed + 2)
        validation = _sample_per_label(validation, split_spec.val_per_label, seed=seed + 3)
        test = _sample_per_label(test, split_spec.test_per_label, seed=seed + 4)
        return {"train": train.reset_index(drop=True), "validation": validation.reset_index(drop=True), "test": test.reset_index(drop=True)}
    raise ValueError(f"Unsupported split kind: {split_spec.kind}")
