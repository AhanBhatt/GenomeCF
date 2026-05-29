from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from genomecf.submission_data import build_submission_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.fast
def test_build_submission_data_function() -> None:
    payload = build_submission_data(run_validation=True)
    source_zip = Path(payload["source_zip"])
    supplementary_zip = Path(payload["supplementary_zip"])
    assert source_zip.exists()
    assert supplementary_zip.exists()
    assert Path(payload["source_workbook"]).exists()
    assert Path(payload["registry_workbook"]).exists()
    assert payload["canonical_registry_included"] is True
    assert (PROJECT_ROOT / "submission_uploads" / "SUBMISSION_DATA_PACKAGE_REPORT.md").exists()

    with zipfile.ZipFile(source_zip) as zf:
        names = set(zf.namelist())
    assert "source_data/GenomeCF_Source_Data.xlsx" in names
    assert "source_data/Fig1/panel_definitions.csv" in names
    assert "source_data/Fig6/fig6_genomecf_synth.csv" in names

    with zipfile.ZipFile(supplementary_zip) as zf:
        names = set(zf.namelist())
    assert "supplementary_data_registry/GenomeCF_Supplementary_Data_and_Registry.xlsx" in names
    assert "supplementary_data_registry/results/release/benchmark_registry.csv" in names
    assert "supplementary_data_registry/FILE_MANIFEST.csv" in names


@pytest.mark.fast
def test_cli_build_submission_data() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "genomecf.cli", "build-submission-data"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert Path(payload["source_zip"]).exists()
    assert Path(payload["supplementary_zip"]).exists()
    assert payload["main_figure_count"] == 6
