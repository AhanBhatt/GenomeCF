from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .paths import DOCS_ROOT, PAPER_ROOT, PROJECT_ROOT, PUBLICATION_ROOT, RELEASE_ROOT
from .release import build_release_registry
from .variant_tasks import VARIANT_TASKS, variant_task_ids as _variant_task_ids


def variant_task_ids() -> list[str]:
    return _variant_task_ids()


def normalize_split_name(split_name: str) -> str:
    value = split_name.strip().lower()
    if value == "chromosome_cv":
        return "chromosome_5fold_cv"
    if value == "synthetic_default":
        return "official"
    return value


def _required_reporting_columns() -> list[str]:
    return [
        "task_id",
        "split_id",
        "model_id",
        "auroc",
        "ece",
        "brier",
        "perturbation_id",
        "train_count",
        "test_count",
        "device",
    ]


def check_reporting_standard(
    *,
    results_path: Path,
    output_path: Path | None = None,
    checklist_path: Path | None = None,
) -> dict[str, object]:
    frame = pd.read_csv(results_path)
    required = _required_reporting_columns()
    missing_columns = [column for column in required if column not in frame.columns]
    missing_models = []
    if "model_id" in frame.columns and "task_id" in frame.columns:
        for task_id in ["human_nontata_promoters", "human_enhancers_cohn"]:
            subset = frame[(frame["task_id"] == task_id) & (frame["split_id"] == "official")]
            if subset.empty:
                missing_models.append(task_id)
    payload = {
        "passed": not missing_columns and not missing_models,
        "missing_columns": missing_columns,
        "missing_required_tasks": missing_models,
        "row_count": int(len(frame)),
        "recommendations": [] if (not missing_columns and not missing_models) else ["Regenerate the release registry and publication artifacts before reporting results."],
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _load_release_summary(regenerate: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if regenerate:
        build_release_registry()
    summary = pd.read_csv(RELEASE_ROOT / "benchmark_summary.csv")
    external = pd.read_csv(RELEASE_ROOT / "external_validation_family_summary.csv")
    return summary, external


def summarize_nature_methods(output_dir: Path, regenerate: bool = False) -> tuple[dict[str, object], Path]:
    summary, external = _load_release_summary(regenerate=regenerate)
    payload = {
        "core_task_count": int(summary[summary["tier"] == "core"]["task_id"].nunique()),
        "external_task_count": int(summary[summary["tier"] == "external"]["task_id"].nunique()),
        "synthetic_task_count": int(summary[summary["tier"] == "synthetic"]["task_id"].nunique()),
        "variant_task_count": int(len(VARIANT_TASKS)),
        "model_count": int(summary["model_id"].nunique()),
        "external_family_count": int(external["external_family"].nunique()) if not external.empty else 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "nature_methods_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, path


def build_website(output_dir: Path | None = None, regenerate: bool = False) -> Path:
    summary, external = _load_release_summary(regenerate=regenerate)
    output_dir = output_dir or (DOCS_ROOT / "site")
    output_dir.mkdir(parents=True, exist_ok=True)
    official = summary[
        (summary["split_id"] == "official")
        & (summary["calibration_method"] == "none")
        & (summary["intervention_id"] == "standard")
        & (summary["tier"].isin(["core", "external", "synthetic"]))
    ].copy()
    leaderboard = (
        official.groupby(["model_id", "model_readable_name"], as_index=False)
        .agg(
            mean_auroc=("auroc", "mean"),
            mean_shortcut_score=("shortcut_score", "mean"),
            worst_gc_bin_gap=("gc_only_explainability_ratio", "mean"),
        )
        .rename(columns={"model_readable_name": "model_label"})
    )
    if not external.empty:
        external_scores = external.groupby("model_id", as_index=False)["external_biological_reliability"].mean().rename(columns={"external_biological_reliability": "external_validation_score"})
        leaderboard = leaderboard.merge(external_scores, on="model_id", how="left")
    else:
        leaderboard["external_validation_score"] = float("nan")
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>GenomeCF</title></head>
<body>
<h1>GenomeCF</h1>
<p>Counterfactual validation standard for DNA sequence models.</p>
<ul>
  <li><a href="leaderboard.csv">Leaderboard CSV</a></li>
  <li><a href="../reporting_checklist.md">Reporting checklist</a></li>
  <li><a href="../QUICKSTART.md">Quickstart</a></li>
</ul>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    (output_dir / "leaderboard.html").write_text("<html><body><h1>GenomeCF leaderboard</h1><p>See leaderboard.csv for machine-readable output.</p></body></html>", encoding="utf-8")
    (output_dir / "quickstart.html").write_text("<html><body><h1>Quickstart</h1><p>Use genomecf reproduce-quickstart to verify the install.</p></body></html>", encoding="utf-8")
    (output_dir / "reporting_standard.html").write_text("<html><body><h1>Reporting standard</h1><p>See docs/reporting_checklist.md.</p></body></html>", encoding="utf-8")
    return output_dir / "index.html"


def trace_paper(output_path: Path | None = None, strict: bool = False) -> dict[str, object]:
    key_numbers_path = PUBLICATION_ROOT / "key_numbers.json"
    key_numbers = json.loads(key_numbers_path.read_text(encoding="utf-8")) if key_numbers_path.exists() else {}
    rows: list[dict[str, object]] = []
    for idx, (key, value) in enumerate(sorted(key_numbers.items()), start=1):
        rows.append(
            {
                "claim_id": f"C{idx:03d}",
                "paper_location": "paper/main_or_supplement",
                "claim_text": key,
                "source_table": "results/publication/key_numbers.json",
                "source_registry_rows": "registry-backed publication artifacts",
                "script": "src/generate_publication_artifacts.py",
                "metric": key,
                "value": value,
                "ci": "",
                "validated": bool(pd.notna(value)),
            }
        )
    extra_paths = {
        "main_pdf": PAPER_ROOT / "genomecf_report.pdf",
        "supplement_pdf": PAPER_ROOT / "genomecf_supplement.pdf",
        "website_index": DOCS_ROOT / "site" / "index.html",
    }
    for name, path in extra_paths.items():
        rows.append(
            {
                "claim_id": f"P_{name}",
                "paper_location": "availability",
                "claim_text": str(path),
                "source_table": str(path),
                "source_registry_rows": "",
                "script": "",
                "metric": name,
                "value": str(path),
                "ci": "",
                "validated": path.exists(),
            }
        )
    frame = pd.DataFrame(rows)
    validated = bool(frame["validated"].all()) if not frame.empty else False
    if strict and not validated:
        raise ValueError("Strict traceability failed because one or more claim artifacts are missing.")
    output_path = output_path or (RELEASE_ROOT / "paper_claim_traceability.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    frame.to_html(output_path.with_suffix(".html"), index=False)
    payload = {"validated": validated, "claim_count": int(len(frame)), "csv_path": str(output_path), "html_path": str(output_path.with_suffix('.html'))}
    output_path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
