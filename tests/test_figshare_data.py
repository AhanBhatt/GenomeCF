from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from genomecf.figshare_data import build_figshare_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.fast
def test_build_figshare_data_function() -> None:
    payload = build_figshare_data(run_validation=True)
    zip_path = Path(payload["zip_file"])
    assert zip_path.exists()
    assert payload["benchmark_registry_included"] is True
    assert payload["source_data_included"] is True
    assert Path(payload["checksums_path"]).exists()
    assert Path(payload["manifest_path"]).exists()
    assert Path(payload["readme_path"]).exists()
    assert Path(payload["report_path"]).exists()
    assert Path(payload["figshare_manifest_json"]).exists()
    assert payload["validation_status"]["valid"] is True

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())

    assert "figshare_data/README.md" in names
    assert "figshare_data/FILE_MANIFEST.csv" in names
    assert "figshare_data/CHECKSUMS_SHA256.txt" in names
    assert "figshare_data/registry/benchmark_registry.csv" in names
    assert any(name.startswith("figshare_data/source_data/Fig1/") for name in names)
    assert not any(name.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")) for name in names)
    assert not any("paper/" in name.lower() for name in names)
    assert not any(
        any(token in name.lower() for token in ("checkpoint", ".bin", ".pth", ".pt"))
        for name in names
    )


@pytest.mark.fast
def test_cli_build_figshare_data() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "genomecf.cli", "build-figshare-data"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert Path(payload["zip_file"]).exists()
    assert payload["benchmark_registry_included"] is True
    assert payload["source_data_included"] is True
    assert payload["validation_status"]["valid"] is True
