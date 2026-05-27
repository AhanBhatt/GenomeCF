from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from genomecf.build_publication import build_publication
from genomecf.release import build_release_registry
from genomecf.validation import validate_release_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.fast
def test_release_registry_and_validation() -> None:
    paths = build_release_registry()
    assert Path(paths["registry_csv"]).exists()
    assert Path(paths["registry_jsonl"]).exists()
    assert Path(paths["summary_csv"]).exists()
    assert Path(paths["matrix_csv"]).exists()
    report = validate_release_results()
    assert report.ok, report.errors


@pytest.mark.fast
def test_cli_summarize_and_appendix(tmp_path: Path) -> None:
    output_dir = tmp_path / "summary"
    result = subprocess.run(
        [sys.executable, "-m", "genomecf.cli", "summarize", "--suite", "core", "--output-dir", str(output_dir)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Wrote summary" in result.stdout
    created = list(output_dir.glob("*.csv"))
    assert created


@pytest.mark.fast
def test_cli_validate_results() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "genomecf.cli", "validate-results"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Release validation passed." in result.stdout


@pytest.mark.fast
def test_generate_release_upgrade_artifacts() -> None:
    result = subprocess.run(
        [sys.executable, "src/generate_release_upgrade_artifacts.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "chromosome_cv_summary.csv" in result.stdout
    assert (PROJECT_ROOT / "results" / "release" / "chromosome_cv_summary.csv").exists()
    assert (PROJECT_ROOT / "figures" / "genomecf_mitigation_summary.png").exists()
    assert (PROJECT_ROOT / "results" / "release" / "gc_bin_summary.csv").exists()
    assert (PROJECT_ROOT / "results" / "release" / "synthetic_extended_summary.csv").exists()


@pytest.mark.slow
def test_cli_smoke_test(tmp_path: Path) -> None:
    output_dir = tmp_path / "smoke"
    result = subprocess.run(
        [sys.executable, "-m", "genomecf.cli", "smoke-test", "--output-dir", str(output_dir)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Smoke test completed" in result.stdout
    report_path = output_dir / "smoke_report.json"
    assert report_path.exists()


@pytest.mark.paper
def test_build_publication_without_latex() -> None:
    build_publication(skip_tests=True, skip_latex=True, skip_validation=False)
    assert (PROJECT_ROOT / "results" / "publication" / "table2_main_results.csv").exists()
    assert (PROJECT_ROOT / "results" / "publication" / "table6_gc_bin_summary.csv").exists()
    assert (PROJECT_ROOT / "results" / "publication" / "table7_motif_summary.csv").exists()
    assert (PROJECT_ROOT / "paper" / "genomecf_report.tex").exists()
    assert (PROJECT_ROOT / "figures" / "genomecf_gc_bin_robustness.png").exists()
    assert (PROJECT_ROOT / "figures" / "genomecf_shortcut_score.png").exists()


@pytest.mark.gpu
def test_gpu_marker_skips_cleanly() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in the current environment.")
    assert torch.cuda.is_available()
