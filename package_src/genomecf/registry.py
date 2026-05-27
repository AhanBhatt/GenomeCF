from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schemas import ResultRow


def append_result_rows(rows: list[ResultRow], path: Path) -> Path:
    frame = pd.DataFrame([row.to_dict() for row in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True, sort=False)
    frame.to_csv(path, index=False)
    return path


def read_registry(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
