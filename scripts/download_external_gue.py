from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset


TASKS = {
    "gue_human_tf_0": "human_tf_0",
    "gue_human_tf_1": "human_tf_1",
    "gue_emp_h3k4me3": "emp_H3K4me3",
    "gue_emp_h3k14ac": "emp_H3K14ac",
}

SPLIT_MAP = {
    "train": "train",
    "validation": "dev",
    "test": "test",
}


def export_subset(task_id: str, subset_name: str, root: Path) -> None:
    out_dir = root / "data" / "gue" / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for local_split, hf_split in SPLIT_MAP.items():
        dataset = load_dataset("leannmlindsey/GUE", subset_name, split=hf_split)
        frame = dataset.to_pandas()
        if "sequence" not in frame.columns or "label" not in frame.columns:
            raise ValueError(f"{subset_name}:{hf_split} is missing required columns; found {frame.columns.tolist()}")
        normalized = frame[["sequence", "label"]].copy()
        normalized["id"] = [f"{task_id}_{local_split}_{idx}" for idx in range(len(normalized))]
        normalized["subset_name"] = subset_name
        normalized["orig_split"] = local_split
        normalized.to_csv(out_dir / f"{local_split}.csv", index=False)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for task_id, subset_name in TASKS.items():
        print(f"Exporting {subset_name} -> {task_id}")
        export_subset(task_id, subset_name, root)


if __name__ == "__main__":
    main()
