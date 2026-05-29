from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .paths import CONFIG_ROOT, DOCS_ROOT, PROJECT_ROOT, RELEASE_ROOT, PUBLICATION_ROOT
from .validation import validate_release_results


FIGSHARE_DATA_ROOT = PROJECT_ROOT / "figshare_data"
FIGSHARE_UPLOADS_ROOT = PROJECT_ROOT / "figshare_uploads"
SOURCE_DATA_ROOT = PROJECT_ROOT / "source_data"
PAPER_ROOT = PROJECT_ROOT / "paper"
FIGURES_ROOT = PROJECT_ROOT / "figures"

EXCLUDED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
EXCLUDED_PATTERNS = [
    "paper/",
    ".git/",
    "__pycache__/",
    "cache/",
    "checkpoint",
    ".bin",
    ".pt",
    ".pth",
]


@dataclass
class ManifestRow:
    file_path: str
    file_name: str
    file_type: str
    category: str
    description: str
    source_or_derived: str
    related_manuscript_item: str
    source_script_or_origin: str
    upstream_data_dependency: str
    license_notes: str
    include_in_figshare: str = "yes"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _copy(src: Path, dst: Path) -> None:
    _ensure_parent(dst)
    shutil.copy2(src, dst)


def _write_text(path: Path, text: str) -> None:
    _ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def _should_copy_source_data(path: Path) -> bool:
    if path.is_dir():
        return False
    if path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return False
    return True


def _file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return "tabular text"
    if suffix in {".json", ".jsonl"}:
        return "JSON"
    if suffix in {".yaml", ".yml"}:
        return "YAML"
    if suffix == ".md":
        return "Markdown"
    if suffix == ".xlsx":
        return "Excel workbook"
    if suffix == ".html":
        return "HTML"
    if suffix == ".txt":
        return "plain text"
    return suffix.lstrip(".") or "file"


def _manifest_to_csv(rows: list[ManifestRow], path: Path) -> None:
    _ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ManifestRow.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _copy_with_manifest(
    src: Path,
    dst: Path,
    rows: list[ManifestRow],
    *,
    category: str,
    description: str,
    source_or_derived: str,
    related: str,
    origin: str,
    upstream: str,
    license_notes: str,
) -> None:
    _copy(src, dst)
    rows.append(
        ManifestRow(
            file_path=dst.relative_to(FIGSHARE_DATA_ROOT).as_posix(),
            file_name=dst.name,
            file_type=_file_type(dst),
            category=category,
            description=description,
            source_or_derived=source_or_derived,
            related_manuscript_item=related,
            source_script_or_origin=origin,
            upstream_data_dependency=upstream,
            license_notes=license_notes,
        )
    )


def _copy_optional(
    rel_src: str,
    rel_dst: str,
    rows: list[ManifestRow],
    missing: list[str],
    *,
    category: str,
    description: str,
    source_or_derived: str,
    related: str,
    origin: str = "existing artifact",
    upstream: str = "none",
    license_notes: str = "Derived GenomeCF output; upstream datasets retain original licenses and terms.",
) -> None:
    src = PROJECT_ROOT / rel_src
    dst = FIGSHARE_DATA_ROOT / rel_dst
    if not src.exists():
        missing.append(rel_src)
        return
    _copy_with_manifest(
        src,
        dst,
        rows,
        category=category,
        description=description,
        source_or_derived=source_or_derived,
        related=related,
        origin=origin,
        upstream=upstream,
        license_notes=license_notes,
    )


def _copy_source_data(rows: list[ManifestRow]) -> None:
    for path in sorted(SOURCE_DATA_ROOT.rglob("*")):
        if path.is_dir():
            continue
        if not _should_copy_source_data(path):
            continue
        rel = path.relative_to(SOURCE_DATA_ROOT)
        dst = FIGSHARE_DATA_ROOT / "source_data" / rel
        desc = "Source data supporting main manuscript display items."
        if path.name == "GenomeCF_Source_Data.xlsx":
            desc = "Workbook containing source data for all main manuscript figures."
        elif path.name == "Data_Dictionary.csv":
            desc = "Column dictionary for the manuscript source-data package."
        _copy_with_manifest(
            path,
            dst,
            rows,
            category="source_data",
            description=desc,
            source_or_derived="derived",
            related="Main figures",
            origin="source_data generated by genomecf build-submission-data",
            upstream="none",
            license_notes="Derived GenomeCF output; upstream datasets retain original licenses and terms.",
        )


def _write_readmes() -> None:
    _write_text(
        FIGSHARE_DATA_ROOT / "README.md",
        "\n".join(
            [
                "# GenomeCF Research Data",
                "",
                "This package contains supporting research data for “GenomeCF: a counterfactual validation standard for DNA sequence models.” It includes registry-backed result tables, source data for manuscript figures/tables, external validation summaries, MPRA variant-effect summaries, GenomeCF-Synth outputs, statistical support files, reporting checklist files and traceability metadata.",
                "",
                "GitHub repository: https://github.com/AhanBhatt/GenomeCF",
                "",
                "This package does not contain the manuscript, supplement, cover letter or figure image files.",
                "This package does not redistribute raw upstream public datasets or pretrained model checkpoints.",
                "Raw/public data should be obtained from the original sources cited in the manuscript.",
                "The canonical result registry is in `registry/benchmark_registry.csv`.",
                "Source data for main display items are in `source_data/`.",
                "External validation outputs are in `external_validation/` and `release_summaries/`.",
                "MPRA variant-effect outputs are in `mpra_variant_effect/`.",
                "Statistical support and traceability files are in `statistical_support/` and `traceability/`.",
                "Reporting checklist and reproducibility metadata are included.",
                "",
                "Suggested citation placeholder: If this manuscript is accepted, please cite the associated article and the GenomeCF repository.",
                "",
                "License note: Use the repository license for code; derived research-data outputs are suitable for CC BY 4.0 deposition if desired. Upstream datasets retain their original licenses and terms.",
            ]
        ),
    )
    _write_text(
        FIGSHARE_DATA_ROOT / "registry" / "README.md",
        "\n".join(
            [
                "# Registry",
                "",
                "`benchmark_registry.csv` is the canonical row-level GenomeCF result registry.",
                "`benchmark_summary.csv` is a derived summary table.",
                "Raw upstream datasets and pretrained models are not included in this package and should be obtained from their original sources listed in the manuscript and repository documentation.",
            ]
        ),
    )
    shared_note = "\n".join(
        [
            "External raw data are not redistributed in this package. The package includes derived GenomeCF outputs and metadata. Original data sources should be accessed through GUE/DNABERT-2, MaveDB, Genomic Benchmarks, or the cited upstream repositories.",
        ]
    )
    _write_text(FIGSHARE_DATA_ROOT / "external_validation" / "README.md", "# External validation\n\n" + shared_note)
    _write_text(FIGSHARE_DATA_ROOT / "mpra_variant_effect" / "README.md", "# MPRA variant-effect outputs\n\n" + shared_note)
    _write_text(
        FIGSHARE_DATA_ROOT / "synthetic_benchmark" / "README.md",
        "\n".join(
            [
                "# GenomeCF-Synth",
                "",
                "This folder contains derived GenomeCF-Synth summaries and synthetic-task configuration metadata. These are safe to redistribute because they are generated synthetic outputs rather than upstream assay data.",
            ]
        ),
    )


def _stage_figshare_data() -> tuple[list[ManifestRow], list[str]]:
    _clean_dir(FIGSHARE_DATA_ROOT)
    for subdir in [
        "registry",
        "source_data",
        "release_summaries",
        "publication_tables",
        "external_validation",
        "mpra_variant_effect",
        "synthetic_benchmark",
        "statistical_support",
        "traceability",
        "reporting_checklist",
        "manifests_and_metadata",
        "reproducibility",
    ]:
        (FIGSHARE_DATA_ROOT / subdir).mkdir(parents=True, exist_ok=True)
    rows: list[ManifestRow] = []
    missing: list[str] = []

    _write_readmes()
    for readme_rel, category, description in [
        ("README.md", "package_metadata", "Top-level description of the figshare research-data package."),
        ("registry/README.md", "registry", "Registry folder guide."),
        ("external_validation/README.md", "external_validation", "External validation redistribution note."),
        ("mpra_variant_effect/README.md", "mpra_variant_effect", "MPRA redistribution note."),
        ("synthetic_benchmark/README.md", "synthetic_benchmark", "GenomeCF-Synth package note."),
    ]:
        path = FIGSHARE_DATA_ROOT / readme_rel
        rows.append(
            ManifestRow(
                file_path=readme_rel.replace("\\", "/"),
                file_name=path.name,
                file_type=_file_type(path),
                category=category,
                description=description,
                source_or_derived="derived",
                related_manuscript_item="Package documentation",
                source_script_or_origin="package_src/genomecf/figshare_data.py",
                upstream_data_dependency="none",
                license_notes="Derived package metadata; upstream datasets retain original licenses and terms.",
            )
        )

    # Canonical registry.
    for rel_src, desc in [
        ("results/release/benchmark_registry.csv", "Canonical row-level benchmark registry."),
        ("results/release/benchmark_registry.jsonl", "JSONL version of the canonical benchmark registry."),
        ("results/release/benchmark_summary.csv", "Derived benchmark summary table."),
    ]:
        _copy_optional(
            rel_src,
            "registry/" + Path(rel_src).name,
            rows,
            missing,
            category="registry",
            description=desc,
            source_or_derived="derived",
            related="Supplementary Data / Registry",
            origin="results/release",
        )

    # Source data.
    _copy_source_data(rows)

    # Release summaries.
    release_files = [
        ("results/release/external_validation_summary.csv", "External validation summary by task/model/configuration."),
        ("results/release/external_validation_family_summary.csv", "External assay-family summary."),
        ("results/release/external_transfer_prediction.csv", "Point-level external transfer prediction inputs."),
        ("results/release/external_transfer_stats.json", "External transfer prediction statistics."),
        ("results/release/mitigation_summary.csv", "Mitigation summary."),
        ("results/release/chromosome_cv_summary.csv", "Chromosome cross-validation summary."),
        ("results/release/chromosome_cv_fold_metrics.csv", "Per-fold chromosome cross-validation metrics."),
        ("results/release/matched_negative_model_summary.csv", "Matched-negative model summary."),
        ("results/release/matched_negative_confounders.csv", "Matched-negative confounder summary."),
        ("results/release/gc_bin_summary.csv", "GC-bin robustness summary."),
        ("results/release/motif_summary.csv", "Legacy expected motif summary alias."),
        ("results/release/real_motif_probe_summary.csv", "Real-task motif probe summary."),
        ("results/release/real_task_motif_disruption.csv", "Real-task motif disruption outputs."),
        ("results/release/synthetic_extended_summary.csv", "GenomeCF-Synth summary."),
        ("results/release/biological_case_study.csv", "MPRA biological case-study summary."),
        ("results/release/model_task_matrix.csv", "Model-task coverage matrix."),
        ("results/release/validation_report.json", "Release validation report."),
        ("results/release/reporting_check_report.json", "Reporting-check checklist output."),
    ]
    for rel_src, desc in release_files:
        _copy_optional(
            rel_src,
            "release_summaries/" + Path(rel_src).name,
            rows,
            missing,
            category="release_summaries",
            description=desc,
            source_or_derived="derived",
            related="Release summaries",
            origin="results/release",
        )

    # Publication tables: CSV only.
    for src in sorted(PUBLICATION_ROOT.glob("*.csv")):
        _copy_with_manifest(
            src,
            FIGSHARE_DATA_ROOT / "publication_tables" / src.name,
            rows,
            category="publication_tables",
            description=f"Publication-facing derived table {src.name}.",
            source_or_derived="derived",
            related="Main paper / Supplementary tables",
            origin="results/publication",
            upstream="none",
            license_notes="Derived GenomeCF output; upstream datasets retain original licenses and terms.",
        )

    # External validation focused copies.
    for rel_src, desc in [
        ("results/release/external_validation_summary.csv", "External validation task-level summary."),
        ("results/release/external_validation_family_summary.csv", "External validation family-level summary."),
        ("results/release/external_transfer_prediction.csv", "External prediction point summary."),
        ("results/release/external_transfer_stats.json", "External prediction statistics and model fits."),
        ("results/release/external_core_profile.csv", "Core-profile summary paired with external outputs."),
        ("results/release/external_prediction_robustness.csv", "External prediction robustness summary."),
        ("results/release/external_gc_bin_summary.csv", "External GC-bin summary."),
        ("results/release/external_gc_bin_by_bin.csv", "External GC-bin by-bin metrics."),
    ]:
        _copy_optional(
            rel_src,
            "external_validation/" + Path(rel_src).name,
            rows,
            missing,
            category="external_validation",
            description=desc,
            source_or_derived="derived",
            related="Fig. 4 / External validation",
            origin="results/release",
        )

    # MPRA summaries only: *_summary.csv and *_gc_bins.csv, plus case-study file.
    variant_dir = RELEASE_ROOT / "variant_effect"
    if variant_dir.exists():
        for src in sorted(variant_dir.glob("*_summary.csv")) + sorted(variant_dir.glob("*_gc_bins.csv")):
            _copy_with_manifest(
                src,
                FIGSHARE_DATA_ROOT / "mpra_variant_effect" / src.name,
                rows,
                category="mpra_variant_effect",
                description="Derived MPRA variant-effect summary or GC-bin table.",
                source_or_derived="derived",
                related="Fig. 5 / MPRA variant-effect",
                origin="results/release/variant_effect",
                upstream="MaveDB / MPRA upstream assays (not redistributed as raw data)",
                license_notes="Derived GenomeCF output; upstream assay data retain original licenses and terms.",
            )
    else:
        missing.append("results/release/variant_effect/")
    _copy_optional(
        "results/release/biological_case_study.csv",
        "mpra_variant_effect/biological_case_study.csv",
        rows,
        missing,
        category="mpra_variant_effect",
        description="MPRA biological case-study summary.",
        source_or_derived="derived",
        related="Fig. 5",
        origin="results/release",
        upstream="MaveDB / MPRA upstream assays (not redistributed as raw data)",
    )

    # Synthetic.
    _copy_optional(
        "results/release/synthetic_extended_summary.csv",
        "synthetic_benchmark/synthetic_extended_summary.csv",
        rows,
        missing,
        category="synthetic_benchmark",
        description="GenomeCF-Synth summary.",
        source_or_derived="derived",
        related="Fig. 6",
        origin="results/release",
    )
    _copy_optional(
        "results/release/core_matrix_synthetic.csv",
        "synthetic_benchmark/core_matrix_synthetic.csv",
        rows,
        missing,
        category="synthetic_benchmark",
        description="Synthetic task matrix summary.",
        source_or_derived="derived",
        related="Fig. 6",
        origin="results/release",
    )
    synthetic_config_dir = CONFIG_ROOT / "synthetic"
    if synthetic_config_dir.exists():
        for src in sorted(synthetic_config_dir.rglob("*")):
            if src.is_dir():
                continue
            _copy_with_manifest(
                src,
                FIGSHARE_DATA_ROOT / "synthetic_benchmark" / "configs" / src.relative_to(synthetic_config_dir),
                rows,
                category="synthetic_benchmark",
                description="Synthetic benchmark configuration file.",
                source_or_derived="source",
                related="Fig. 6 / Synthetic benchmark configs",
                origin="configs/synthetic",
                upstream="none",
                license_notes="Repository configuration file under repository license.",
            )
    else:
        missing.append("configs/synthetic/")

    # Statistical support.
    for rel_src, desc in [
        ("results/release/statistical_claims.csv", "Statistical support for headline manuscript claims."),
        ("results/release/external_transfer_stats.json", "External prediction statistics including bootstrap and permutation summaries."),
        ("results/release/external_prediction_robustness.csv", "External prediction robustness summary."),
        ("results/release/reporting_check_report.json", "Reporting checklist validation output."),
    ]:
        _copy_optional(
            rel_src,
            "statistical_support/" + Path(rel_src).name,
            rows,
            missing,
            category="statistical_support",
            description=desc,
            source_or_derived="derived",
            related="Statistical support",
            origin="results/release",
        )

    # Traceability.
    for rel_src, desc in [
        ("results/release/paper_claim_traceability.csv", "Paper-claim traceability CSV."),
        ("results/release/paper_claim_traceability.html", "Paper-claim traceability HTML report."),
        ("results/release/paper_claim_traceability.json", "Paper-claim traceability summary JSON."),
        ("paper/claims.yaml", "Manual claim map used by trace-paper."),
    ]:
        _copy_optional(
            rel_src,
            "traceability/" + Path(rel_src).name,
            rows,
            missing,
            category="traceability",
            description=desc,
            source_or_derived="derived" if rel_src.startswith("results/") else "source",
            related="Traceability",
            origin="trace-paper" if rel_src.startswith("results/") else "paper",
        )

    # Reporting checklist.
    for rel_src, desc in [
        ("docs/reporting_checklist.yaml", "Machine-readable GenomeCF reporting checklist."),
        ("docs/reporting_checklist_schema.json", "JSON schema for the reporting checklist."),
        ("docs/example_completed_checklist.md", "Example completed reporting checklist."),
        ("results/release/reporting_check_report.json", "Report produced by the checklist validator."),
    ]:
        _copy_optional(
            rel_src,
            "reporting_checklist/" + Path(rel_src).name,
            rows,
            missing,
            category="reporting_checklist",
            description=desc,
            source_or_derived="source" if rel_src.startswith("docs/") else "derived",
            related="Reporting checklist",
            origin="docs" if rel_src.startswith("docs/") else "results/release",
        )

    # Manifests and metadata.
    manifest_files = [
        ("configs/task_manifests.jsonl", "Task manifests.", "Task manifests"),
        ("configs/model_manifests.jsonl", "Model manifests.", "Model manifests"),
        ("configs/perturbation_manifests.jsonl", "Perturbation manifests.", "Perturbations"),
        ("configs/split_manifests.jsonl", "Split manifests.", "Split definitions"),
        ("configs/manifest_summary.md", "Manifest summary.", "Manifests"),
        ("docs/RESULT_SCHEMA.md", "Result schema documentation.", "Registry schema"),
        ("docs/METRICS.md", "Metric definitions.", "Methods"),
        ("docs/SPLITS.md", "Split definitions.", "Methods"),
        ("docs/MODELS.md", "Model descriptions.", "Methods"),
        ("docs/TASKS.md", "Task descriptions.", "Methods"),
        ("docs/TASK_MANIFESTS.md", "Task-manifest overview.", "Methods"),
        ("pyproject.toml", "Package metadata.", "Software metadata"),
        ("README.md", "Repository README.", "Repository metadata"),
    ]
    for rel_src, desc, related in manifest_files:
        _copy_optional(
            rel_src,
            "manifests_and_metadata/" + Path(rel_src).name,
            rows,
            missing,
            category="manifests_and_metadata",
            description=desc,
            source_or_derived="source",
            related=related,
            origin=rel_src,
            license_notes="Repository documentation/configuration under repository license.",
        )

    # Reproducibility docs.
    for rel_src, desc in [
        ("docs/DATA_AVAILABILITY.md", "Data availability statement."),
        ("docs/CODE_AVAILABILITY.md", "Code availability statement."),
        ("docs/REPRODUCIBILITY.md", "Reproducibility guide."),
        ("docs/PROTOCOL.md", "Protocol overview."),
        ("docs/REPRODUCIBILITY_PROTOCOL.md", "Reproducibility protocol."),
        ("docs/SYNTHETIC_TASKS.md", "GenomeCF-Synth documentation."),
        ("docs/EXTERNAL_VALIDATION.md", "External validation documentation."),
        ("docs/BIOLOGICAL_CASE_STUDY.md", "Biological case-study documentation."),
        ("docs/GC_BIN_ROBUSTNESS.md", "GC-bin robustness documentation."),
        ("docs/MOTIF_ANALYSIS.md", "Motif-analysis documentation."),
        ("docs/NATURE_METHODS_RELEASE.md", "Nature Methods release notes."),
        ("docs/NATURE_METHODS_SUBMISSION_CHECKLIST.md", "Nature Methods submission checklist."),
    ]:
        _copy_optional(
            rel_src,
            "reproducibility/" + Path(rel_src).name,
            rows,
            missing,
            category="reproducibility",
            description=desc,
            source_or_derived="source",
            related="Reproducibility metadata",
            origin="docs",
            license_notes="Repository documentation under repository license.",
        )

    if missing:
        _write_text(
            FIGSHARE_DATA_ROOT / "MISSING_EXPECTED_FILES.md",
            "\n".join(["# Missing expected files", ""] + [f"- `{item}`" for item in missing]),
        )
        rows.append(
            ManifestRow(
                file_path="MISSING_EXPECTED_FILES.md",
                file_name="MISSING_EXPECTED_FILES.md",
                file_type="Markdown",
                category="package_metadata",
                description="Expected files that were not present in the repository when packaging figshare data.",
                source_or_derived="derived",
                related_manuscript_item="Package validation",
                source_script_or_origin="package_src/genomecf/figshare_data.py",
                upstream_data_dependency="none",
                license_notes="Derived package metadata.",
            )
        )

    return rows, missing


def _write_checksums() -> Path:
    checksum_path = FIGSHARE_DATA_ROOT / "CHECKSUMS_SHA256.txt"
    lines = []
    for path in sorted(p for p in FIGSHARE_DATA_ROOT.rglob("*") if p.is_file() and p.name != "CHECKSUMS_SHA256.txt"):
        lines.append(f"{_sha256_file(path)}  {path.relative_to(FIGSHARE_DATA_ROOT).as_posix()}")
    _write_text(checksum_path, "\n".join(lines) + ("\n" if lines else ""))
    return checksum_path


def _zip_figshare_data(zip_path: Path) -> tuple[int, int]:
    file_count = 0
    total_size = 0
    _ensure_parent(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in FIGSHARE_DATA_ROOT.rglob("*") if p.is_file()):
            arcname = path.relative_to(FIGSHARE_DATA_ROOT.parent).as_posix()
            zf.write(path, arcname=arcname)
            file_count += 1
            total_size += path.stat().st_size
    return file_count, total_size


def _validate_figshare_zip(zip_path: Path) -> dict[str, object]:
    forbidden_members: list[str] = []
    required_members = {
        "figshare_data/README.md",
        "figshare_data/FILE_MANIFEST.csv",
        "figshare_data/CHECKSUMS_SHA256.txt",
        "figshare_data/registry/benchmark_registry.csv",
    }
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for name in names:
            lower = name.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
                forbidden_members.append(name)
            if lower.startswith("figshare_data/paper/") or lower.startswith("paper/"):
                forbidden_members.append(name)
            if "__pycache__" in lower or ".git/" in lower or "cache/" in lower:
                forbidden_members.append(name)
            if any(token in lower for token in ("checkpoint", ".bin", ".pth", ".pt")):
                forbidden_members.append(name)
        missing_required = sorted(required_members - names)
    return {
        "valid": not forbidden_members and not missing_required,
        "forbidden_members": sorted(set(forbidden_members)),
        "missing_required": missing_required,
    }


def build_figshare_data(*, run_validation: bool = True) -> dict[str, object]:
    if run_validation:
        report = validate_release_results()
        if not report.ok:
            raise ValueError("Release validation failed before building figshare data: " + "; ".join(report.errors))

    rows, missing = _stage_figshare_data()
    manifest_path = FIGSHARE_DATA_ROOT / "FILE_MANIFEST.csv"
    _manifest_to_csv(rows, manifest_path)
    checksum_path = _write_checksums()

    FIGSHARE_UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = FIGSHARE_UPLOADS_ROOT / "GenomeCF_Research_Data.zip"
    file_count, total_size = _zip_figshare_data(zip_path)
    zip_sha256 = _sha256_file(zip_path)
    validation = _validate_figshare_zip(zip_path)

    _copy(FIGSHARE_DATA_ROOT / "README.md", FIGSHARE_UPLOADS_ROOT / "README_figshare.md")
    _copy(manifest_path, FIGSHARE_UPLOADS_ROOT / "FIGSHARE_FILE_MANIFEST.csv")
    _copy(checksum_path, FIGSHARE_UPLOADS_ROOT / "CHECKSUMS_SHA256.txt")

    report_text = "\n".join(
        [
            "# Figshare Upload Report",
            "",
            f"- Created ZIP path: `{zip_path}`",
            f"- Number of files: {file_count}",
            f"- Total staged size (bytes before ZIP compression): {total_size}",
            "- Included categories: registry, source_data, release_summaries, publication_tables, external_validation, mpra_variant_effect, synthetic_benchmark, statistical_support, traceability, reporting_checklist, manifests_and_metadata, reproducibility",
            "- Excluded categories: manuscript PDFs/TeX, cover letter, figure images, raw upstream datasets, pretrained checkpoints, cache folders, logs not needed for reproducibility",
            f"- Missing expected files: {len(missing)}",
            f"- Checksum file path: `{checksum_path}`",
            "",
            "Recommended figshare title: GenomeCF Research Data",
            "",
            "Recommended figshare description:",
            "Supporting research data for “GenomeCF: a counterfactual validation standard for DNA sequence models.” This dataset contains registry-backed GenomeCF result tables, source data for main display items, external validation summaries, MPRA variant-effect summaries, GenomeCF-Synth outputs, statistical support files, reporting checklist files and paper-claim traceability metadata. Manuscript files, figure images, raw upstream public datasets and pretrained model checkpoints are not included. Code and documentation are available at https://github.com/AhanBhatt/GenomeCF.",
            "",
            "Recommended keywords:",
            "- GenomeCF",
            "- DNA sequence models",
            "- genomics",
            "- machine learning",
            "- counterfactual validation",
            "- benchmark",
            "- MPRA",
            "- variant effect prediction",
            "- model calibration",
            "- source data",
            "",
            "Recommended license:",
            "- CC BY 4.0 for derived GenomeCF research data, while code in the repository remains MIT-licensed. Upstream datasets retain their original licenses and terms.",
            "",
            f"Strict validation status: {'pass' if validation['valid'] else 'fail'}",
        ]
    )
    _write_text(FIGSHARE_UPLOADS_ROOT / "FIGSHARE_UPLOAD_REPORT.md", report_text)

    manifest_json = {
        "zip_file": str(zip_path),
        "created_at": _utc_timestamp(),
        "file_count": file_count,
        "total_size_bytes": total_size,
        "sha256_of_zip": zip_sha256,
        "included_top_level_dirs": sorted({p.relative_to(FIGSHARE_DATA_ROOT).parts[0] for p in FIGSHARE_DATA_ROOT.iterdir()}),
        "excluded_patterns": EXCLUDED_PATTERNS,
        "validation_status": validation,
        "notes": "Package contains derived research data only; no manuscript files, figure images, raw upstream datasets or model checkpoints are included.",
    }
    _write_text(FIGSHARE_UPLOADS_ROOT / "figshare_data_manifest.json", json.dumps(manifest_json, indent=2))

    payload = {
        "zip_file": str(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "file_count": file_count,
        "benchmark_registry_included": (FIGSHARE_DATA_ROOT / "registry" / "benchmark_registry.csv").exists(),
        "source_data_included": (FIGSHARE_DATA_ROOT / "source_data").exists(),
        "checksums_path": str(checksum_path),
        "manifest_path": str(manifest_path),
        "readme_path": str(FIGSHARE_UPLOADS_ROOT / "README_figshare.md"),
        "report_path": str(FIGSHARE_UPLOADS_ROOT / "FIGSHARE_UPLOAD_REPORT.md"),
        "figshare_manifest_json": str(FIGSHARE_UPLOADS_ROOT / "figshare_data_manifest.json"),
        "missing_expected_files": missing,
        "validation_status": validation,
        "recommended_title": "GenomeCF Research Data",
        "recommended_description": "Supporting research data for “GenomeCF: a counterfactual validation standard for DNA sequence models.” This dataset contains registry-backed GenomeCF result tables, source data for main display items, external validation summaries, MPRA variant-effect summaries, GenomeCF-Synth outputs, statistical support files, reporting checklist files and paper-claim traceability metadata. Manuscript files, figure images, raw upstream public datasets and pretrained model checkpoints are not included. Code and documentation are available at https://github.com/AhanBhatt/GenomeCF.",
        "recommended_keywords": [
            "GenomeCF",
            "DNA sequence models",
            "genomics",
            "machine learning",
            "counterfactual validation",
            "benchmark",
            "MPRA",
            "variant effect prediction",
            "model calibration",
            "source data",
        ],
        "recommended_license": "CC BY 4.0 for derived GenomeCF research data; repository code remains MIT-licensed.",
    }
    return payload
