from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from genomecf.config import get_task_spec
from genomecf.data import load_task_frame
from genomecf.nature_methods import (
    build_website,
    check_reporting_standard,
    normalize_split_name,
    summarize_nature_methods,
    trace_paper,
    variant_task_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.fast
def test_reporting_checklist_passes_on_release_registry(tmp_path: Path) -> None:
    output_path = tmp_path / "check_report.json"
    payload = check_reporting_standard(
        results_path=PROJECT_ROOT / "results" / "release" / "benchmark_registry.csv",
        output_path=output_path,
    )
    assert payload["passed"] is True
    assert output_path.exists()


@pytest.mark.fast
def test_build_website_without_regeneration(tmp_path: Path) -> None:
    index_path = build_website(output_dir=tmp_path / "site", regenerate=False)
    assert index_path.exists()
    assert (tmp_path / "site" / "leaderboard.csv").exists()
    leaderboard = pd.read_csv(tmp_path / "site" / "leaderboard.csv")
    assert not leaderboard.empty
    assert {"model_id", "mean_auroc", "mean_shortcut_score", "external_validation_score"}.issubset(leaderboard.columns)


@pytest.mark.fast
def test_summarize_nature_methods_without_regeneration(tmp_path: Path) -> None:
    summary, path = summarize_nature_methods(output_dir=tmp_path / "nm", regenerate=False)
    assert path.exists()
    assert len(summary) >= 4


@pytest.mark.fast
def test_variant_tasks_and_external_release_stats_exist() -> None:
    task_ids = variant_task_ids()
    assert "mpra_bcl11a_enhancer" in task_ids
    assert len(task_ids) >= 5

    external_summary = pd.read_csv(PROJECT_ROOT / "results" / "release" / "external_validation_summary.csv")
    assert "Variant effect" in set(external_summary["external_family"])

    stats = json.loads((PROJECT_ROOT / "results" / "release" / "external_transfer_stats.json").read_text())
    assert int(stats["pair_count"]) >= 50


@pytest.mark.fast
def test_trace_paper_writes_validated_output(tmp_path: Path) -> None:
    payload = trace_paper(output_path=tmp_path / "paper_claim_traceability.csv", strict=True)
    assert payload["validated"] is True
    assert payload["claim_count"] >= 10
    assert (tmp_path / "paper_claim_traceability.csv").exists()
    assert (tmp_path / "paper_claim_traceability.html").exists()


@pytest.mark.fast
def test_synthetic_task_loader_and_split_alias() -> None:
    assert normalize_split_name("chromosome_cv") == "chromosome_5fold_cv"
    assert normalize_split_name("synthetic_default") == "official"
    frame = load_task_frame(get_task_spec("gc_conflict"), PROJECT_ROOT)
    assert not frame.empty
    assert {"train", "validation", "test"} == set(frame["orig_split"].unique())
