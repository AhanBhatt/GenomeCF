from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.fast
def test_nt_validation_report_exists_and_has_expected_fields() -> None:
    path = PROJECT_ROOT / "results" / "release" / "nt_validation_report.json"
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["model_id"] == "nucleotide_transformer_v2"
    assert "validated" in payload
    assert "task_checks" in payload
    assert len(payload["task_checks"]) >= 2
    for item in payload["task_checks"]:
        assert {"task_id", "embedding_dim", "probe_auroc", "probe_reproducible"}.issubset(item.keys())


@pytest.mark.fast
def test_release_tracks_caduceus_and_dnabert_cv_rows() -> None:
    foundation_status = pd.read_csv(PROJECT_ROOT / "results" / "release" / "foundation_loader_status.csv")
    caduceus = foundation_status[foundation_status["model_id"] == "caduceus_ph"]
    assert not caduceus.empty
    assert caduceus.iloc[0]["status"] == "completed"

    cv_summary = pd.read_csv(PROJECT_ROOT / "results" / "release" / "chromosome_cv_summary.csv")
    dnabert_rows = cv_summary[(cv_summary["model_id"] == "dnabert2") & (cv_summary["task_id"].isin(["human_nontata_promoters", "human_enhancers_cohn"]))]
    assert len(dnabert_rows) == 2
    caduceus_rows = cv_summary[(cv_summary["model_id"] == "caduceus_ph") & (cv_summary["task_id"].isin(["human_nontata_promoters", "human_enhancers_cohn", "human_enhancers_ensembl", "human_ocr_ensembl"]))]
    assert len(caduceus_rows) == 4


@pytest.mark.fast
def test_release_tracks_foundation_mitigation_and_extended_synthetic() -> None:
    mitigation = pd.read_csv(PROJECT_ROOT / "results" / "release" / "mitigation_summary.csv")
    caduceus_mn = mitigation[
        (mitigation["task_id"] == "human_nontata_promoters")
        & (mitigation["model_id"] == "caduceus_ph")
        & (mitigation["intervention_id"] == "matched_negative_retraining")
    ]
    assert not caduceus_mn.empty

    synth = pd.read_csv(PROJECT_ROOT / "results" / "release" / "synthetic_extended_summary.csv")
    assert {"gc_conflict", "two_motif_grammar", "motif_position_conflict"}.issubset(set(synth["task_id"]))


@pytest.mark.gpu
def test_caduceus_loader_smoke_or_skip() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in the current environment.")
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        pytest.skip("mamba_ssm is not installed in the current environment.")

    from genomecf.config import get_model_spec
    from genomecf.models import HFEncoderRunner

    runner = HFEncoderRunner(get_model_spec("caduceus_ph"), mode="frozen", seed=7, batch_size=2)
    embeddings = runner._encode(["ACGTACGTACGT", "TGCATGCATGCA"])
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0
