from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import get_task_spec, list_specs
from .paths import PROJECT_ROOT, RELEASE_ROOT


VARIANT_TASKS = {
    task_id: get_task_spec(task_id)
    for task_id in list_specs("tasks")
    if get_task_spec(task_id).classification_type == "binary_variant_effect_prioritization"
}


def variant_task_ids() -> list[str]:
    return sorted(VARIANT_TASKS)


def ensure_variant_tasks_prepared(project_root: Path = PROJECT_ROOT, task_ids: list[str] | None = None, force: bool = False) -> dict[str, dict[str, str]]:
    task_ids = task_ids or variant_task_ids()
    prepared: dict[str, dict[str, str]] = {}
    for task_id in task_ids:
        base = project_root / "data" / "mavedb" / task_id
        if not base.exists():
            raise FileNotFoundError(f"Missing variant task directory: {base}")
        prepared[task_id] = {
            "metadata": str(base / "metadata.json"),
            "train": str(base / "train.csv"),
            "validation": str(base / "validation.csv"),
            "test": str(base / "test.csv"),
        }
    return prepared


def run_variant_evaluation(
    *,
    task_name: str,
    model_name: str,
    split_name: str,
    mode: str,
    seed: int,
    calibration: str = "none",
    intervention_id: str = "standard",
    project_root: Path = PROJECT_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    variant_root = project_root / "results" / "release" / "variant_effect"
    stem = f"{task_name}__{model_name}__{split_name}__{intervention_id}__{calibration}__seed{seed}_summary.csv"
    path = variant_root / stem
    if not path.exists():
        raise FileNotFoundError(f"Missing variant summary artifact: {path}")
    summary = pd.read_csv(path)
    return summary, summary.copy(), path
