from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .paths import PUBLICATION_ROOT, RELEASE_ROOT
from .release import build_release_registry


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    report_path: Path | None = None


def validate_release_results(project_root: Path | None = None) -> ValidationReport:
    build_release_registry(project_root)
    release_root = RELEASE_ROOT if project_root is None else Path(project_root) / "results" / "release"
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        release_root / "benchmark_registry.csv",
        release_root / "benchmark_summary.csv",
        release_root / "model_task_matrix.csv",
        release_root / "external_validation_summary.csv",
        release_root / "chromosome_cv_summary.csv",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"Missing required artifact: {path}")
    if not errors:
        registry = pd.read_csv(release_root / "benchmark_registry.csv")
        if registry.empty:
            errors.append("Release registry is empty.")
        summary = pd.read_csv(release_root / "benchmark_summary.csv")
        if summary.empty:
            errors.append("Release summary is empty.")
        if (PUBLICATION_ROOT / "key_numbers.json").exists():
            payload = json.loads((PUBLICATION_ROOT / "key_numbers.json").read_text(encoding="utf-8"))
            if not payload:
                warnings.append("Publication key numbers file exists but is empty.")
    report = ValidationReport(ok=not errors, errors=errors, warnings=warnings, report_path=release_root / "validation_report.json")
    report.report_path.write_text(json.dumps({"ok": report.ok, "errors": errors, "warnings": warnings}, indent=2), encoding="utf-8")
    return report
