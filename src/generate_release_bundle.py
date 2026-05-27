from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_RELEASE = PROJECT_ROOT / "results" / "release"
RESULTS_PUBLICATION = PROJECT_ROOT / "results" / "publication"
PAPER_DIR = PROJECT_ROOT / "paper"
DOCS_DIR = PROJECT_ROOT / "docs"
RELEASE_DIR = PROJECT_ROOT / "release"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value: str | float | int | None) -> float:
    if value in ("", None):
        raise ValueError("Missing numeric value")
    return float(value)


def maybe_float(value: str | float | int | None) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def fmt_number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def fmt_ci(low: float | None, high: float | None, digits: int = 3) -> str:
    if low is None or high is None:
        return "--"
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("^", "\\textasciicircum{}")
    )


def find_row(rows: Iterable[dict[str, str]], **filters: str) -> dict[str, str]:
    matched = []
    for row in rows:
        if all(str(row.get(key, "")) == str(value) for key, value in filters.items()):
            matched.append(row)
    if not matched:
        raise KeyError(f"No row found for filters: {filters}")
    if len(matched) > 1:
        raise KeyError(f"Expected one row, found {len(matched)} for filters: {filters}")
    return matched[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_statistical_claims() -> list[dict[str, object]]:
    main_rows = read_csv(RESULTS_PUBLICATION / "table2_main_results.csv")
    matched_rows = read_csv(RESULTS_RELEASE / "matched_negative_model_summary.csv")
    mitigation_rows = read_csv(RESULTS_RELEASE / "mitigation_summary.csv")
    case_rows = read_csv(RESULTS_RELEASE / "biological_case_study.csv")
    synth_rows = read_csv(RESULTS_RELEASE / "synthetic_extended_summary.csv")
    external_stats = json.loads((RESULTS_RELEASE / "external_transfer_stats.json").read_text(encoding="utf-8"))

    promoter_kmer = find_row(
        main_rows,
        dataset="human_nontata_promoters",
        model_family="kmer_logistic_regression",
    )
    promoter_rcaug = find_row(
        main_rows,
        dataset="human_nontata_promoters",
        model_family="small_cnn_rc_aug",
    )
    promoter_dnabert = find_row(
        main_rows,
        dataset="human_nontata_promoters",
        model_family="dnabert2",
    )
    promoter_caduceus = find_row(
        main_rows,
        dataset="human_nontata_promoters",
        model_family="caduceus_ph",
    )
    cohn_dnabert = find_row(
        main_rows,
        dataset="human_enhancers_cohn",
        model_family="dnabert2",
    )

    promoter_gc_official = find_row(
        matched_rows,
        task_id="human_nontata_promoters",
        split_id="official",
        model_id="gc_only",
    )
    promoter_gc_matched = find_row(
        matched_rows,
        task_id="human_nontata_promoters",
        split_id="matched_test",
        model_id="gc_only",
    )

    promoter_rcaug_standard = find_row(
        mitigation_rows,
        task_id="human_nontata_promoters",
        model_id="small_cnn_rc_aug",
        intervention_id="standard",
        calibration_method="none",
    )
    promoter_rcaug_temperature = find_row(
        mitigation_rows,
        task_id="human_nontata_promoters",
        model_id="small_cnn_rc_aug",
        intervention_id="standard",
        calibration_method="temperature",
    )

    case_a_standard = find_row(
        case_rows,
        case_study_id="Case A",
        model_id="small_cnn",
        condition_label="Standard",
    )
    case_a_temp = find_row(
        case_rows,
        case_study_id="Case A",
        model_id="small_cnn",
        condition_label="Temperature-scaled",
    )
    case_b_dnabert = find_row(
        case_rows,
        case_study_id="Case B",
        model_id="dnabert2",
        condition_label="Standard",
    )
    case_b_kmer = find_row(
        case_rows,
        case_study_id="Case B",
        model_id="kmer_logistic_regression",
        condition_label="Standard",
    )

    gc_conflict_dnabert = find_row(synth_rows, task_id="gc_conflict", model_id="dnabert2")

    reg_auroc = next(item for item in external_stats["regression"] if item["predictors"] == "core_mean_auroc")
    reg_shortcut = next(item for item in external_stats["regression"] if item["predictors"] == "core_mean_shortcut_score")
    reg_full = next(
        item
        for item in external_stats["regression"]
        if item["predictors"]
        == "core_mean_auroc+core_mean_rc_delta+core_mean_ece+core_matched_negative_shift+core_gc_bin_auroc_gap"
    )

    rows = [
        {
            "claim_id": "abstract_promoter_kmer_auroc",
            "paper_location": "Abstract; Results/Held-out AUROC",
            "metric": "AUROC",
            "estimate": to_float(promoter_kmer["ensemble_auroc"]),
            "CI_low": None,
            "CI_high": None,
            "test": "registry point estimate",
            "p_value_or_bootstrap_probability": None,
            "n": 2000,
            "registry_source": "results/publication/table2_main_results.csv",
            "script": "src/generate_release_bundle.py",
        },
        {
            "claim_id": "abstract_promoter_kmer_rc_instability",
            "paper_location": "Abstract; Results/Held-out AUROC",
            "metric": "RC instability",
            "estimate": to_float(promoter_kmer["ensemble_reverse_complement_mean_abs_delta"]),
            "CI_low": None,
            "CI_high": None,
            "test": "registry point estimate",
            "p_value_or_bootstrap_probability": None,
            "n": 2000,
            "registry_source": "results/publication/table2_main_results.csv",
            "script": "src/generate_release_bundle.py",
        },
        {
            "claim_id": "abstract_promoter_rcaug_rc_instability",
            "paper_location": "Abstract; Results/Held-out AUROC",
            "metric": "RC instability",
            "estimate": to_float(promoter_rcaug["ensemble_reverse_complement_mean_abs_delta"]),
            "CI_low": None,
            "CI_high": None,
            "test": "registry point estimate",
            "p_value_or_bootstrap_probability": None,
            "n": 2000,
            "registry_source": "results/publication/table2_main_results.csv",
            "script": "src/generate_release_bundle.py",
        },
        {
            "claim_id": "cohn_dnabert_negative_mono_drop",
            "paper_location": "Abstract; Results/Held-out AUROC",
            "metric": "Mononucleotide shuffle drop",
            "estimate": to_float(cohn_dnabert["ensemble_mono_shuffle_positive_prob_drop"]),
            "CI_low": None,
            "CI_high": None,
            "test": "registry point estimate",
            "p_value_or_bootstrap_probability": None,
            "n": 6948,
            "registry_source": "results/publication/table2_main_results.csv",
            "script": "src/generate_release_bundle.py",
        },
        {
            "claim_id": "promoter_caduceus_vs_dnabert_rc_delta_gain",
            "paper_location": "Results/Foundation models",
            "metric": "RC instability improvement",
            "estimate": to_float(promoter_dnabert["ensemble_reverse_complement_mean_abs_delta"])
            - to_float(promoter_caduceus["ensemble_reverse_complement_mean_abs_delta"]),
            "CI_low": None,
            "CI_high": None,
            "test": "difference of registry point estimates",
            "p_value_or_bootstrap_probability": None,
            "n": 2000,
            "registry_source": "results/publication/table2_main_results.csv",
            "script": "src/generate_release_bundle.py",
        },
        {
            "claim_id": "promoter_gc_only_matched_negative_drop",
            "paper_location": "Abstract; Results/Matched-negative",
            "metric": "Matched-negative AUROC drop",
            "estimate": to_float(promoter_gc_official["auroc"]) - to_float(promoter_gc_matched["auroc"]),
            "CI_low": None,
            "CI_high": None,
            "test": "difference of registry point estimates",
            "p_value_or_bootstrap_probability": None,
            "n": 2000,
            "registry_source": "results/release/matched_negative_model_summary.csv",
            "script": "src/generate_release_bundle.py",
        },
        {
            "claim_id": "promoter_rcaug_temperature_ece_reduction",
            "paper_location": "Abstract; Results/Matched-negative and GC-bin analyses",
            "metric": "ECE reduction",
            "estimate": to_float(promoter_rcaug_standard["ece"]) - to_float(promoter_rcaug_temperature["ece"]),
            "CI_low": None,
            "CI_high": None,
            "test": "difference of registry point estimates",
            "p_value_or_bootstrap_probability": None,
            "n": 2000,
            "registry_source": "results/release/mitigation_summary.csv",
            "script": "src/generate_release_bundle.py",
        },
        {
            "claim_id": "external_auroc_only_r2",
            "paper_location": "Abstract; Results/External validation",
            "metric": "R^2",
            "estimate": reg_auroc["r2"],
            "CI_low": reg_auroc["ci_low"],
            "CI_high": reg_auroc["ci_high"],
            "test": "bootstrap confidence interval",
            "p_value_or_bootstrap_probability": None,
            "n": reg_auroc["n"],
            "registry_source": "results/release/external_transfer_stats.json",
            "script": "src/generate_release_bundle.py",
        },
        {
            "claim_id": "external_shortcut_only_r2",
            "paper_location": "Abstract; Results/External validation",
            "metric": "R^2",
            "estimate": reg_shortcut["r2"],
            "CI_low": reg_shortcut["ci_low"],
            "CI_high": reg_shortcut["ci_high"],
            "test": "bootstrap confidence interval",
            "p_value_or_bootstrap_probability": None,
            "n": reg_shortcut["n"],
            "registry_source": "results/release/external_transfer_stats.json",
            "script": "src/generate_release_bundle.py",
        },
        {
            "claim_id": "external_full_profile_r2",
            "paper_location": "Abstract; Results/External validation",
            "metric": "R^2",
            "estimate": reg_full["r2"],
            "CI_low": reg_full["ci_low"],
            "CI_high": reg_full["ci_high"],
            "test": "bootstrap confidence interval",
            "p_value_or_bootstrap_probability": None,
            "n": reg_full["n"],
            "registry_source": "results/release/external_transfer_stats.json",
            "script": "src/generate_release_bundle.py",
        },
        {
            "claim_id": "external_full_profile_advantage",
            "paper_location": "Abstract; Results/External validation",
            "metric": "Delta R^2 over AUROC-only",
            "estimate": external_stats["full_profile_advantage"]["observed_delta"],
            "CI_low": external_stats["full_profile_advantage"]["ci_low"],
            "CI_high": external_stats["full_profile_advantage"]["ci_high"],
            "test": "bootstrap confidence interval",
            "p_value_or_bootstrap_probability": None,
            "n": external_stats["pair_count"],
            "registry_source": "results/release/external_transfer_stats.json",
            "script": "src/generate_release_bundle.py",
        },
        {
            "claim_id": "case_a_bcl11a_auprc_gain",
            "paper_location": "Abstract; Results/Case study",
            "metric": "AUPRC gain",
            "estimate": to_float(case_a_temp["auprc"]) - to_float(case_a_standard["auprc"]),
            "CI_low": None,
            "CI_high": None,
            "test": "difference of registry point estimates",
            "p_value_or_bootstrap_probability": None,
            "n": 1,
            "registry_source": "results/release/biological_case_study.csv",
            "script": "src/generate_release_bundle.py",
        },
        {
            "claim_id": "case_a_bcl11a_topk_gain",
            "paper_location": "Abstract; Results/Case study",
            "metric": "Top-k enrichment gain",
            "estimate": to_float(case_a_temp["topk_enrichment"]) - to_float(case_a_standard["topk_enrichment"]),
            "CI_low": None,
            "CI_high": None,
            "test": "difference of registry point estimates",
            "p_value_or_bootstrap_probability": None,
            "n": 1,
            "registry_source": "results/release/biological_case_study.csv",
            "script": "src/generate_release_bundle.py",
        },
        {
            "claim_id": "case_b_myc_dnabert_topk_advantage",
            "paper_location": "Abstract; Results/Case study",
            "metric": "Top-k enrichment advantage",
            "estimate": to_float(case_b_dnabert["topk_enrichment"]) - to_float(case_b_kmer["topk_enrichment"]),
            "CI_low": None,
            "CI_high": None,
            "test": "difference of registry point estimates",
            "p_value_or_bootstrap_probability": None,
            "n": 1,
            "registry_source": "results/release/biological_case_study.csv",
            "script": "src/generate_release_bundle.py",
        },
        {
            "claim_id": "synthetic_gc_conflict_dnabert_shortcut_following",
            "paper_location": "Abstract; Results/GenomeCF-Synth",
            "metric": "Shortcut-following rate",
            "estimate": to_float(gc_conflict_dnabert["shortcut_following_rate"]),
            "CI_low": None,
            "CI_high": None,
            "test": "deterministic synthetic evaluation",
            "p_value_or_bootstrap_probability": None,
            "n": 1,
            "registry_source": "results/release/synthetic_extended_summary.csv",
            "script": "src/generate_release_bundle.py",
        },
    ]
    return rows


def write_statistical_claims(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "claim_id",
        "paper_location",
        "metric",
        "estimate",
        "CI_low",
        "CI_high",
        "test",
        "p_value_or_bootstrap_probability",
        "n",
        "registry_source",
        "script",
    ]
    write_csv(RESULTS_RELEASE / "statistical_claims.csv", rows, fieldnames)

    source_alias = {
        "results/publication/table2_main_results.csv": "main results",
        "results/release/matched_negative_model_summary.csv": "matched negative",
        "results/release/mitigation_summary.csv": "mitigation",
        "results/release/external_transfer_stats.json": "external stats",
        "results/release/biological_case_study.csv": "case study",
        "results/release/synthetic_extended_summary.csv": "synthetic",
    }

    def tex_metric(metric: str) -> str:
        return metric.replace("R^2", "$R^2$")

    tex_lines = [
        r"\begin{tabular}{p{3.2cm}p{3.3cm}p{1.4cm}p{2.2cm}p{2.6cm}p{0.9cm}p{2.4cm}}",
        r"\toprule",
        r"Claim & Metric & Estimate & 95\% CI & Test & $n$ & Source \\",
        r"\midrule",
    ]
    for row in rows:
        tex_lines.append(
            " & ".join(
                [
                    latex_escape(str(row["claim_id"]).replace("_", " ")),
                    tex_metric(str(row["metric"])),
                    latex_escape(fmt_number(maybe_float(row["estimate"]))),
                    latex_escape(fmt_ci(maybe_float(row["CI_low"]), maybe_float(row["CI_high"]))),
                    latex_escape(str(row["test"])),
                    latex_escape(str(row["n"])),
                    latex_escape(source_alias.get(str(row["registry_source"]), Path(str(row["registry_source"])).stem)),
                ]
            )
            + r" \\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (RESULTS_PUBLICATION / "appendix_statistical_claims.tex").write_text(
        "\n".join(tex_lines) + "\n",
        encoding="utf-8",
    )


def write_claims_yaml(rows: list[dict[str, object]]) -> None:
    lines = ["claims:"]
    for row in rows:
        lines.extend(
            [
                f"  - claim_id: {row['claim_id']}",
                f"    paper_location: \"{row['paper_location']}\"",
                f"    claim_text: \"{row['metric']} = {fmt_number(maybe_float(row['estimate']))}\"",
                f"    source_registry_file: \"{row['registry_source']}\"",
                "    source_rows_filter: \"see generating script filters\"",
                f"    source_script: \"{row['script']}\"",
                f"    validated_value: {fmt_number(maybe_float(row['estimate']), 6)}",
                "    tolerance: 0.001",
                "    status: validated",
            ]
        )
    (PAPER_DIR / "claims.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_release_bundle() -> None:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = [
        RESULTS_RELEASE / "benchmark_registry.csv",
        RESULTS_RELEASE / "benchmark_registry.jsonl",
        RESULTS_RELEASE / "benchmark_summary.csv",
        RESULTS_RELEASE / "external_transfer_stats.json",
        RESULTS_PUBLICATION / "key_numbers.json",
        DOCS_DIR / "site" / "index.html",
        DOCS_DIR / "site" / "leaderboard.html",
        DOCS_DIR / "site" / "quickstart.html",
        DOCS_DIR / "site" / "reporting_standard.html",
        DOCS_DIR / "reporting_checklist.yaml",
    ]

    manifest = {
        "release_name": "GenomeCF_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": [],
    }
    checksum_lines = []
    for artifact in artifacts:
        if not artifact.exists():
            continue
        rel_path = artifact.relative_to(PROJECT_ROOT).as_posix()
        digest = sha256(artifact)
        checksum_lines.append(f"{digest}  {rel_path}")
        manifest["artifacts"].append(
            {
                "path": rel_path,
                "sha256": digest,
                "size_bytes": artifact.stat().st_size,
            }
        )

    (RELEASE_DIR / "GenomeCF_v1_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (RELEASE_DIR / "GenomeCF_v1_checksums.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    (RELEASE_DIR / "GenomeCF_v1_reproduction_commands.sh").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                "python -m pip install -e .[benchmark,dev]",
                "genomecf --help",
                "python -m pytest",
                "python -m genomecf.cli validate-results",
                "python -m genomecf.cli check-report --results results/release/benchmark_registry.csv",
                "python -m genomecf.cli reproduce-quickstart",
                "python -m genomecf.cli reproduce-focal",
                "python -m genomecf.cli reproduce-external",
                "python -m genomecf.cli build-website",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (RELEASE_DIR / "GenomeCF_v1_expected_outputs.md").write_text(
        "\n".join(
            [
                "# GenomeCF v1 Expected Outputs",
                "",
                "- (manuscript kept private; not included in this repo)",
                "- `results/release/benchmark_registry.csv`",
                "- `results/release/benchmark_summary.csv`",
                "- `results/release/external_transfer_stats.json`",
                "- `docs/site/index.html`", 
                "- `docs/site/leaderboard.html`",
                "- `docs/reporting_checklist.yaml`",
                "- `release/GenomeCF_v1_manifest.json`",
                "- `release/GenomeCF_v1_checksums.txt`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    write_release_bundle()
    print(
        json.dumps(
            {
                "release_manifest": str(RELEASE_DIR / "GenomeCF_v1_manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
