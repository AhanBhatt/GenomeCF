from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .paths import DOCS_ROOT, FIGURES_ROOT, PAPER_ROOT, PROJECT_ROOT, PUBLICATION_ROOT, RELEASE_ROOT
from .validation import validate_release_results


SOURCE_DATA_ROOT = PROJECT_ROOT / "source_data"
SUPPLEMENTARY_ROOT = PROJECT_ROOT / "supplementary_data_registry"
SUBMISSION_UPLOADS_ROOT = PROJECT_ROOT / "submission_uploads"


@dataclass
class DisplayItem:
    kind: str
    number: int
    caption: str
    includegraphics: list[str]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    _ensure_parent(dst)
    shutil.copy2(src, dst)
    return True


def _normalize_caption(text: str) -> str:
    cleaned = text.replace("\n", " ")
    cleaned = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\[a-zA-Z]+\*?", "", cleaned)
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def parse_display_items(tex_path: Path, kind_prefix: str) -> list[DisplayItem]:
    text = tex_path.read_text(encoding="utf-8")
    pattern = re.compile(rf"\\begin\{{({kind_prefix}\*?)\}}(.*?)\\end\{{\1\}}", re.S)
    items: list[DisplayItem] = []
    counter = 0
    for match in pattern.finditer(text):
        env_body = match.group(2)
        captions = re.findall(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", env_body, re.S)
        if not captions:
            continue
        counter += 1
        graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", env_body)
        items.append(
            DisplayItem(
                kind=kind_prefix,
                number=counter,
                caption=_normalize_caption(captions[-1]),
                includegraphics=graphics,
            )
        )
    return items


def parse_main_display_items() -> tuple[list[DisplayItem], list[DisplayItem]]:
    main_figures = parse_display_items(PAPER_ROOT / "genomecf_report.tex", "figure")
    main_tables = parse_display_items(PAPER_ROOT / "genomecf_report.tex", "table")
    return main_figures, main_tables


def parse_supplementary_tables() -> list[DisplayItem]:
    return parse_display_items(PAPER_ROOT / "genomecf_supplement.tex", "table")


def _find_full_profile_model(stats: dict[str, object]) -> dict[str, object]:
    for row in stats.get("regression", []):
        if row.get("predictors") == "core_mean_auroc+core_mean_rc_delta+core_mean_ece+core_matched_negative_shift+core_gc_bin_auroc_gap":
            return row
    raise ValueError("Full GenomeCF profile regression model not found in external_transfer_stats.json")


def _flatten_external_model_fits(stats: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in stats.get("regression", []):
        rows.append(
            {
                "fit_type": "in_sample_regression",
                "group": "all_pairs",
                "predictors": row.get("predictors"),
                "outcome": row.get("outcome"),
                "r2": row.get("r2"),
                "ci_low": row.get("ci_low"),
                "ci_high": row.get("ci_high"),
                "n": row.get("n"),
                "p_value_or_permutation_p": None,
                "notes": "",
            }
        )
    for row in stats.get("leave_one_family_out", []):
        rows.append(
            {
                "fit_type": "leave_one_family_out",
                "group": "all_pairs",
                "predictors": row.get("predictors"),
                "outcome": row.get("outcome"),
                "r2": row.get("cv_r2"),
                "ci_low": None,
                "ci_high": None,
                "n": None,
                "p_value_or_permutation_p": None,
                "notes": json.dumps(row.get("folds", [])),
            }
        )
    for row in stats.get("leave_one_task_out", []):
        rows.append(
            {
                "fit_type": "leave_one_task_out",
                "group": row.get("group_col", "task_id"),
                "predictors": row.get("predictors"),
                "outcome": row.get("outcome"),
                "r2": row.get("cv_r2"),
                "ci_low": None,
                "ci_high": None,
                "n": None,
                "p_value_or_permutation_p": None,
                "notes": json.dumps(row.get("folds", [])),
            }
        )
    for row in stats.get("family_stratified_regression", []):
        rows.append(
            {
                "fit_type": "family_stratified_regression",
                "group": row.get("external_family"),
                "predictors": row.get("predictors"),
                "outcome": row.get("outcome"),
                "r2": row.get("r2"),
                "ci_low": row.get("ci_low"),
                "ci_high": row.get("ci_high"),
                "n": row.get("n"),
                "p_value_or_permutation_p": None,
                "notes": "",
            }
        )
    for key in ("full_profile_advantage", "shortcut_permutation", "permutation", "in_sample_permutation"):
        row = stats.get(key)
        if not row:
            continue
        rows.append(
            {
                "fit_type": key,
                "group": "all_pairs",
                "predictors": None,
                "outcome": None,
                "r2": row.get("observed_delta"),
                "ci_low": row.get("ci_low"),
                "ci_high": row.get("ci_high"),
                "n": row.get("n_boot") or row.get("n_perm"),
                "p_value_or_permutation_p": row.get("p_value"),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def _sheet_name(name: str) -> str:
    return name[:31]


def _write_workbook(workbook_path: Path, sheets: list[tuple[str, pd.DataFrame]]) -> None:
    _ensure_parent(workbook_path)
    with pd.ExcelWriter(workbook_path, engine="xlsxwriter") as writer:
        for sheet_name, frame in sheets:
            frame.to_excel(writer, sheet_name=_sheet_name(sheet_name), index=False)
            worksheet = writer.sheets[_sheet_name(sheet_name)]
            worksheet.freeze_panes(1, 0)
            for idx, column in enumerate(frame.columns):
                max_len = max(len(str(column)), *(len(str(v)) for v in frame[column].head(200).fillna("").tolist()))
                worksheet.set_column(idx, idx, min(max(max_len + 2, 12), 40))


def _make_markdown_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| Field | Value |", "|---|---|"]
    lines.extend(f"| {k} | {v} |" for k, v in rows)
    return "\n".join(lines)


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    _ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def _source_registry_key(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [c for c in cols if c in frame.columns]
    if not present:
        return pd.Series([""] * len(frame))
    return frame[present].astype(str).agg("|".join, axis=1)


def _build_fig1_source(main_figures: list[DisplayItem]) -> dict[str, object]:
    folder = SOURCE_DATA_ROOT / "Fig1"
    folder.mkdir(parents=True, exist_ok=True)
    figure = next(item for item in main_figures if item.number == 1)
    panel_df = pd.DataFrame(
        [
            {
                "panel_or_box": "Input 1",
                "label": "Core benchmark",
                "description": "Core human tasks with held-out and chromosome-grouped evaluation.",
                "input_artifact_or_doc": "results/publication/table1_task_overview.csv",
                "notes": "Main counterfactual benchmark block.",
            },
            {
                "panel_or_box": "Input 2",
                "label": "External biological validation",
                "description": "TF-binding, histone-mark and MPRA variant-effect assays.",
                "input_artifact_or_doc": "results/release/external_validation_summary.csv",
                "notes": "Supports external transfer and biological validation panels.",
            },
            {
                "panel_or_box": "Input 3",
                "label": "GenomeCF-Synth",
                "description": "Controlled shortcut-conflict and rule-learning tasks.",
                "input_artifact_or_doc": "results/release/synthetic_extended_summary.csv",
                "notes": "Used for mechanism-focused stress tests.",
            },
            {
                "panel_or_box": "Input 4",
                "label": "Software/reporting resource",
                "description": "CLI, registry, website and reporting checklist.",
                "input_artifact_or_doc": "docs/reporting_checklist.yaml",
                "notes": "Submission-ready software layer.",
            },
            {
                "panel_or_box": "Output 1",
                "label": "Model reliability profile",
                "description": "Counterfactual, calibration and robustness summary for each model.",
                "input_artifact_or_doc": "results/release/benchmark_registry.csv",
                "notes": "Generated from release registry summaries.",
            },
            {
                "panel_or_box": "Output 2",
                "label": "Variant-prioritization guidance",
                "description": "Case-study recommendations for MPRA-supported variant ranking tasks.",
                "input_artifact_or_doc": "results/release/biological_case_study.csv",
                "notes": "Main Fig. 5 downstream-use case.",
            },
            {
                "panel_or_box": "Output 3",
                "label": "Registry / leaderboard",
                "description": "Public-facing summaries and website downloads.",
                "input_artifact_or_doc": "docs/site/leaderboard.csv",
                "notes": "Website-facing derivative artifact.",
            },
        ]
    )
    panel_df.to_csv(folder / "panel_definitions.csv", index=False)
    _write_text(
        folder / "README.md",
        "\n".join(
            [
                "# Fig. 1 source data",
                "",
                "This folder documents the schematic source mapping for Fig. 1 in the GenomeCF manuscript.",
                "Fig. 1 is a resource overview schematic rather than a quantitative plot, so the key source-data file is `panel_definitions.csv`.",
                "",
                "- Manuscript figure: Fig. 1",
                f"- Figure title: {figure.caption}",
                "- Source files: `results/publication/table1_task_overview.csv`, `results/release/benchmark_registry.csv`, `results/release/external_validation_summary.csv`, `results/release/biological_case_study.csv`, `results/release/synthetic_extended_summary.csv`, `docs/reporting_checklist.yaml`, `docs/site/leaderboard.csv`",
                "- Generation script: `package_src/genomecf/submission_data.py` via `genomecf build-submission-data`",
                "",
                "Columns:",
                "- `panel_or_box`: logical block in the schematic.",
                "- `label`: short visible label used in the schematic.",
                "- `description`: summary of the block content.",
                "- `input_artifact_or_doc`: registry-backed artifact or documentation source used for the block.",
                "- `notes`: additional interpretation guidance.",
            ]
        ),
    )
    _write_metadata(
        folder / "metadata.json",
        {
            "figure_or_table": "Fig. 1",
            "title": figure.caption,
            "manuscript_file": "paper/genomecf_report.tex",
            "source_files": [
                "results/publication/table1_task_overview.csv",
                "results/release/benchmark_registry.csv",
                "results/release/external_validation_summary.csv",
                "results/release/biological_case_study.csv",
                "results/release/synthetic_extended_summary.csv",
                "docs/reporting_checklist.yaml",
                "docs/site/leaderboard.csv",
            ],
            "generation_script": "package_src/genomecf/submission_data.py",
            "date_generated": _utc_timestamp(),
            "registry_hash": _sha256_file(RELEASE_ROOT / "benchmark_registry.csv"),
            "notes": "Schematic-only figure; no direct plotted numeric series.",
            "license_or_restrictions": "No raw upstream datasets redistributed in this folder.",
        },
    )
    _write_text(
        folder / "source_script.txt",
        "\n".join(
            [
                "Command:",
                "python -m genomecf.cli build-submission-data",
                "",
                "Logic:",
                "- Parse final main-paper display items from paper/genomecf_report.tex.",
                "- Map Fig. 1 schematic blocks to the registry-backed artifacts and public docs they summarize.",
                "- Write panel_definitions.csv as the source-data equivalent for this schematic figure.",
            ]
        ),
    )
    return {
        "sheet_name": "Fig1_panel_definitions",
        "sheet_df": panel_df,
        "data_files": [folder / "panel_definitions.csv"],
        "manifest_rows": [
            {
                "figure_or_table": "Fig. 1",
                "folder": "source_data/Fig1",
                "file_name": "panel_definitions.csv",
                "description": "Schematic panel and artifact mapping for the resource overview.",
            }
        ],
    }


def _build_fig2_source(main_figures: list[DisplayItem], benchmark_summary: pd.DataFrame) -> dict[str, object]:
    folder = SOURCE_DATA_ROOT / "Fig2"
    folder.mkdir(parents=True, exist_ok=True)
    figure = next(item for item in main_figures if item.number == 2)
    table2 = pd.read_csv(PUBLICATION_ROOT / "table2_main_results.csv")
    official = benchmark_summary[
        (benchmark_summary["split_id"] == "official")
        & (benchmark_summary["intervention_id"] == "standard")
        & (benchmark_summary["calibration_method"] == "none")
    ][["task_id", "model_id", "mode", "split_id", "seeds"]].rename(columns={"seeds": "seed_list"})
    fig2 = table2.rename(
        columns={
            "dataset": "task_id",
            "model_family": "model_id",
            "ensemble_auroc": "AUROC",
            "ensemble_ece": "ECE",
            "ensemble_brier": "Brier",
            "ensemble_reverse_complement_mean_abs_delta": "RC_delta",
            "ensemble_mono_shuffle_positive_prob_drop": "mono_drop",
            "ensemble_dinuc_shuffle_positive_prob_drop": "dinuc_drop",
        }
    ).copy()
    fig2 = fig2.merge(official, on=["task_id", "model_id"], how="left")
    fig2["split"] = fig2["split_id"].fillna("official")
    fig2["source_registry_key"] = _source_registry_key(fig2, ["task_id", "model_id", "split", "mode"])
    fig2["source_file"] = "results/publication/table2_main_results.csv"
    fig2 = fig2[
        [
            "task_id",
            "task_label",
            "model_id",
            "model_label",
            "AUROC",
            "RC_delta",
            "mono_drop",
            "ECE",
            "Brier",
            "split",
            "mode",
            "source_registry_key",
            "source_file",
        ]
    ]
    fig2.to_csv(folder / "fig2_tradeoff_points.csv", index=False)
    registry_rows = benchmark_summary[
        (benchmark_summary["task_id"].isin(fig2["task_id"]))
        & (benchmark_summary["model_id"].isin(fig2["model_id"]))
        & (benchmark_summary["split_id"] == "official")
        & (benchmark_summary["intervention_id"] == "standard")
        & (benchmark_summary["calibration_method"] == "none")
    ].copy()
    registry_rows.to_csv(folder / "source_registry_rows.csv", index=False)
    _write_text(
        folder / "README.md",
        "\n".join(
            [
                "# Fig. 2 source data",
                "",
                "This folder contains the numerical source data for Fig. 2 (held-out AUROC versus GenomeCF reliability axes).",
                f"- Manuscript figure: Fig. 2",
                f"- Figure title: {figure.caption}",
                "- Source files: `results/publication/table2_main_results.csv`, `results/release/benchmark_summary.csv`",
                "- Generation script: `package_src/genomecf/submission_data.py` via `genomecf build-submission-data`",
                "",
                "Columns:",
                "- `AUROC`: held-out AUROC on the official split.",
                "- `RC_delta`: reverse-complement mean absolute probability delta.",
                "- `mono_drop`: positive-class probability drop after mononucleotide shuffle.",
                "- `ECE`: expected calibration error on the official split.",
                "- `Brier`: Brier score on the official split.",
                "- `source_registry_key`: composite identifier linking the point back to the benchmark summary rows.",
            ]
        ),
    )
    _write_metadata(
        folder / "metadata.json",
        {
            "figure_or_table": "Fig. 2",
            "title": figure.caption,
            "manuscript_file": "paper/genomecf_report.tex",
            "source_files": [
                "results/publication/table2_main_results.csv",
                "results/release/benchmark_summary.csv",
            ],
            "generation_script": "package_src/genomecf/submission_data.py",
            "date_generated": _utc_timestamp(),
            "registry_hash": _sha256_file(RELEASE_ROOT / "benchmark_registry.csv"),
            "notes": "Direct plotted points plus publication summary subset.",
            "license_or_restrictions": "Derived from registry-backed benchmark summaries.",
        },
    )
    _write_text(
        folder / "source_script.txt",
        "\n".join(
            [
                "Command:",
                "python -m genomecf.cli build-submission-data",
                "",
                "Logic:",
                "- Load results/publication/table2_main_results.csv.",
                "- Merge official split / mode metadata from results/release/benchmark_summary.csv.",
                "- Export plotted values and the corresponding benchmark summary rows.",
            ]
        ),
    )
    return {
        "sheet_name": "Fig2_tradeoff_points",
        "sheet_df": fig2,
        "data_files": [folder / "fig2_tradeoff_points.csv", folder / "source_registry_rows.csv"],
        "manifest_rows": [
            {
                "figure_or_table": "Fig. 2",
                "folder": "source_data/Fig2",
                "file_name": "fig2_tradeoff_points.csv",
                "description": "Official-split AUROC, RC instability, shuffle response and calibration values for the core tradeoff figure.",
            }
        ],
    }


def _build_fig3_source(main_figures: list[DisplayItem]) -> dict[str, object]:
    folder = SOURCE_DATA_ROOT / "Fig3"
    folder.mkdir(parents=True, exist_ok=True)
    figure = next(item for item in main_figures if item.number == 3)
    mitigation = pd.read_csv(PUBLICATION_ROOT / "table5_mitigation_summary.csv")
    focal = mitigation[
        mitigation["model_id"].isin(["dnabert2", "caduceus_ph"])
        & mitigation["task_id"].isin(["human_nontata_promoters", "human_enhancers_cohn"])
    ].copy()
    rows: list[dict[str, object]] = []
    metric_map = {
        "official_AUROC": ("auroc_before", "auroc_after"),
        "matched_negative_AUROC": ("matched_auroc_before", "matched_auroc_after"),
        "ECE": ("ece_before", "ece_after"),
        "Brier": ("brier_before", "brier_after"),
        "RC_delta": ("rc_before", "rc_after"),
    }
    for _, row in focal.iterrows():
        for metric, (before_col, after_col) in metric_map.items():
            rows.append(
                {
                    "task_id": row["task_id"],
                    "task_label": row["task_label"],
                    "model_id": row["model_id"],
                    "model_label": row["model_label"],
                    "configuration": row["intervention"],
                    "metric": metric,
                    "value": row[before_col],
                    "before_or_after": "before",
                    "official_AUROC": row["auroc_before"],
                    "matched_negative_AUROC": row["matched_auroc_before"],
                    "ECE": row["ece_before"],
                    "Brier": row["brier_before"],
                    "RC_delta": row["rc_before"],
                    "intervention": row["intervention"],
                    "source_file": "results/publication/table5_mitigation_summary.csv",
                }
            )
            rows.append(
                {
                    "task_id": row["task_id"],
                    "task_label": row["task_label"],
                    "model_id": row["model_id"],
                    "model_label": row["model_label"],
                    "configuration": row["intervention"],
                    "metric": metric,
                    "value": row[after_col],
                    "before_or_after": "after",
                    "official_AUROC": row["auroc_after"],
                    "matched_negative_AUROC": row["matched_auroc_after"],
                    "ECE": row["ece_after"],
                    "Brier": row["brier_after"],
                    "RC_delta": row["rc_after"],
                    "intervention": row["intervention"],
                    "source_file": "results/publication/table5_mitigation_summary.csv",
                }
            )
    fig3 = pd.DataFrame(rows)
    fig3.to_csv(folder / "fig3_foundation_comparison.csv", index=False)
    focal.to_csv(folder / "source_registry_rows.csv", index=False)
    _write_text(
        folder / "README.md",
        "\n".join(
            [
                "# Fig. 3 source data",
                "",
                "This folder contains the source data for the foundation-model comparison and adaptation figure.",
                f"- Manuscript figure: Fig. 3",
                f"- Figure title: {figure.caption}",
                "- Source files: `results/publication/table5_mitigation_summary.csv`",
                "- Generation script: `package_src/genomecf/submission_data.py` via `genomecf build-submission-data`",
                "",
                "Rows are in long format. Each row stores one metric value for one task/model/intervention phase while also carrying the phase-wide AUROC, matched-negative AUROC, calibration and RC context values.",
            ]
        ),
    )
    _write_metadata(
        folder / "metadata.json",
        {
            "figure_or_table": "Fig. 3",
            "title": figure.caption,
            "manuscript_file": "paper/genomecf_report.tex",
            "source_files": ["results/publication/table5_mitigation_summary.csv"],
            "generation_script": "package_src/genomecf/submission_data.py",
            "date_generated": _utc_timestamp(),
            "registry_hash": _sha256_file(RELEASE_ROOT / "benchmark_registry.csv"),
            "notes": "Long-form mitigation summary for focal DNABERT-2 and Caduceus-Ph rows.",
            "license_or_restrictions": "Derived summary only.",
        },
    )
    _write_text(
        folder / "source_script.txt",
        "\n".join(
            [
                "Command:",
                "python -m genomecf.cli build-submission-data",
                "",
                "Logic:",
                "- Load results/publication/table5_mitigation_summary.csv.",
                "- Filter to focal DNABERT-2 and Caduceus-Ph rows.",
                "- Expand before/after metrics into long format for the figure panels.",
            ]
        ),
    )
    return {
        "sheet_name": "Fig3_foundation_comparison",
        "sheet_df": fig3,
        "data_files": [folder / "fig3_foundation_comparison.csv", folder / "source_registry_rows.csv"],
        "manifest_rows": [
            {
                "figure_or_table": "Fig. 3",
                "folder": "source_data/Fig3",
                "file_name": "fig3_foundation_comparison.csv",
                "description": "Foundation-model official and intervention results in long format for the main comparison figure.",
            }
        ],
    }


def _build_fig4_source(main_figures: list[DisplayItem]) -> dict[str, object]:
    folder = SOURCE_DATA_ROOT / "Fig4"
    folder.mkdir(parents=True, exist_ok=True)
    figure = next(item for item in main_figures if item.number == 4)
    family = pd.read_csv(RELEASE_ROOT / "external_validation_family_summary.csv").copy()
    family["source_file"] = "results/release/external_validation_family_summary.csv"
    family.to_csv(folder / "fig4_external_assay_family_summary.csv", index=False)

    points = pd.read_csv(RELEASE_ROOT / "external_transfer_prediction.csv").copy()
    detailed = pd.read_csv(RELEASE_ROOT / "external_validation_summary.csv")[
        [
            "task_id",
            "model_id",
            "condition_label",
            "auroc",
            "auprc",
            "ece",
            "brier",
            "worst_bin_auroc",
            "external_biological_reliability",
            "external_reliability_risk",
        ]
    ].rename(
        columns={
            "auroc": "external_AUROC",
            "auprc": "external_AUPRC",
            "ece": "external_ECE",
            "brier": "external_Brier",
            "worst_bin_auroc": "worst_GC_bin_AUROC",
        }
    )
    points = points.merge(
        detailed,
        on=["task_id", "model_id", "condition_label", "external_biological_reliability", "external_reliability_risk"],
        how="left",
    )
    stats = _load_json(RELEASE_ROOT / "external_transfer_stats.json")
    full_profile = _find_full_profile_model(stats)
    predictors = full_profile["predictors"].split("+")
    coefficients = full_profile["coef"]
    intercept = full_profile["intercept"]
    linear_score = sum(points[predictor].fillna(0.0) * coef for predictor, coef in zip(predictors, coefficients))
    points["full_GenomeCF_profile_score"] = linear_score
    points["predicted_external_reliability"] = intercept + linear_score
    points["external_metric"] = "external_biological_reliability"
    points["source_file"] = "results/release/external_transfer_prediction.csv"
    points.to_csv(folder / "fig4_external_prediction_points.csv", index=False)

    fits = _flatten_external_model_fits(stats)
    fits.to_csv(folder / "fig4_external_prediction_model_fits.csv", index=False)
    points.to_csv(folder / "source_registry_rows.csv", index=False)
    _write_text(
        folder / "README.md",
        "\n".join(
            [
                "# Fig. 4 source data",
                "",
                "This folder contains the source data for the external validation and external-reliability prediction figure.",
                f"- Manuscript figure: Fig. 4",
                f"- Figure title: {figure.caption}",
                "- Source files: `results/release/external_validation_family_summary.csv`, `results/release/external_validation_summary.csv`, `results/release/external_transfer_prediction.csv`, `results/release/external_transfer_stats.json`",
                "- Generation script: `package_src/genomecf/submission_data.py` via `genomecf build-submission-data`",
                "",
                "Files:",
                "- `fig4_external_assay_family_summary.csv`: assay-family summary metrics plotted on the left side of Fig. 4.",
                "- `fig4_external_prediction_points.csv`: model-configuration-task points used for the right-side prediction panels.",
                "- `fig4_external_prediction_model_fits.csv`: fitted regressions, LOFO analyses and permutation summaries supporting the prediction comparisons.",
            ]
        ),
    )
    _write_metadata(
        folder / "metadata.json",
        {
            "figure_or_table": "Fig. 4",
            "title": figure.caption,
            "manuscript_file": "paper/genomecf_report.tex",
            "source_files": [
                "results/release/external_validation_family_summary.csv",
                "results/release/external_validation_summary.csv",
                "results/release/external_transfer_prediction.csv",
                "results/release/external_transfer_stats.json",
            ],
            "generation_script": "package_src/genomecf/submission_data.py",
            "date_generated": _utc_timestamp(),
            "registry_hash": _sha256_file(RELEASE_ROOT / "benchmark_registry.csv"),
            "notes": "Includes fitted full-profile predictions computed from coefficients stored in external_transfer_stats.json.",
            "license_or_restrictions": "Derived summaries only; raw public assay files are not redistributed.",
        },
    )
    _write_text(
        folder / "source_script.txt",
        "\n".join(
            [
                "Command:",
                "python -m genomecf.cli build-submission-data",
                "",
                "Logic:",
                "- Load external assay-family summaries and model-configuration-task point summaries.",
                "- Merge point-level metrics from external_validation_summary.csv onto external_transfer_prediction.csv.",
                "- Compute the full GenomeCF profile linear score and predicted external reliability using the coefficients recorded in external_transfer_stats.json.",
                "- Flatten the regression and robustness summaries into fig4_external_prediction_model_fits.csv.",
            ]
        ),
    )
    return {
        "sheet_name": "Fig4_external_family",
        "sheet_df": family,
        "extra_sheets": [
            ("Fig4_external_prediction", points),
            ("Fig4_model_fits", fits),
        ],
        "data_files": [
            folder / "fig4_external_assay_family_summary.csv",
            folder / "fig4_external_prediction_points.csv",
            folder / "fig4_external_prediction_model_fits.csv",
            folder / "source_registry_rows.csv",
        ],
        "manifest_rows": [
            {
                "figure_or_table": "Fig. 4",
                "folder": "source_data/Fig4",
                "file_name": "fig4_external_assay_family_summary.csv",
                "description": "Assay-family summary metrics for the left-hand panel of the external validation figure.",
            },
            {
                "figure_or_table": "Fig. 4",
                "folder": "source_data/Fig4",
                "file_name": "fig4_external_prediction_points.csv",
                "description": "Model-configuration-task prediction points for the external reliability analysis.",
            },
            {
                "figure_or_table": "Fig. 4",
                "folder": "source_data/Fig4",
                "file_name": "fig4_external_prediction_model_fits.csv",
                "description": "Regression, LOFO and permutation summaries for external prediction analyses.",
            },
        ],
    }


def _build_fig5_source(main_figures: list[DisplayItem]) -> dict[str, object]:
    folder = SOURCE_DATA_ROOT / "Fig5"
    folder.mkdir(parents=True, exist_ok=True)
    figure = next(item for item in main_figures if item.number == 5)
    cases = pd.read_csv(RELEASE_ROOT / "biological_case_study.csv").copy()
    ext = pd.read_csv(RELEASE_ROOT / "external_validation_summary.csv")[
        ["task_id", "model_id", "condition_label", "topk_precision"]
    ].copy()
    merged = cases.merge(ext, on=["task_id", "model_id", "condition_label"], how="left")
    merged["objective"] = merged["task_id"].map(
        {
            "mpra_bcl11a_enhancer": "variant_prioritization",
            "mpra_myc_enhancer": "top_k_nomination",
        }
    ).fillna("variant_prioritization")
    merged["GenomeCF_recommendation"] = merged["decision_role"].str.contains("GenomeCF-aware", case=False, na=False)
    merged["AUROC_only_choice"] = merged["decision_role"].str.contains("AUROC-only", case=False, na=False)
    merged["configuration"] = merged["condition_label"]
    merged["source_file"] = "results/release/biological_case_study.csv"
    fig5 = merged[
        [
            "case_study_id",
            "task_id",
            "model_id",
            "configuration",
            "decision_role",
            "objective",
            "auroc",
            "auprc",
            "topk_enrichment",
            "topk_precision",
            "spearman_abs_effect",
            "worst_bin_auroc",
            "ece",
            "GenomeCF_recommendation",
            "AUROC_only_choice",
            "source_file",
        ]
    ].rename(
        columns={
            "case_study_id": "case_id",
            "auroc": "AUROC",
            "auprc": "AUPRC",
            "topk_enrichment": "top_k_enrichment",
            "topk_precision": "top_k_precision",
            "spearman_abs_effect": "Spearman",
            "worst_bin_auroc": "worst_GC_bin_AUROC",
        }
    )
    fig5.to_csv(folder / "fig5_mpra_case_studies.csv", index=False)
    merged.to_csv(folder / "source_registry_rows.csv", index=False)
    _write_text(
        folder / "README.md",
        "\n".join(
            [
                "# Fig. 5 source data",
                "",
                "This folder contains the case-study source data for the MPRA biological-use figure.",
                f"- Manuscript figure: Fig. 5",
                f"- Figure title: {figure.caption}",
                "- Source files: `results/release/biological_case_study.csv`, `results/release/external_validation_summary.csv`",
                "- Generation script: `package_src/genomecf/submission_data.py` via `genomecf build-submission-data`",
                "",
                "The `objective` column distinguishes BCL11A variant prioritization from MYC top-k nomination, matching the manuscript wording.",
            ]
        ),
    )
    _write_metadata(
        folder / "metadata.json",
        {
            "figure_or_table": "Fig. 5",
            "title": figure.caption,
            "manuscript_file": "paper/genomecf_report.tex",
            "source_files": [
                "results/release/biological_case_study.csv",
                "results/release/external_validation_summary.csv",
            ],
            "generation_script": "package_src/genomecf/submission_data.py",
            "date_generated": _utc_timestamp(),
            "registry_hash": _sha256_file(RELEASE_ROOT / "benchmark_registry.csv"),
            "notes": "Merges case-study summaries with top-k precision values from external validation results.",
            "license_or_restrictions": "Derived summary only; no raw MPRA records redistributed here.",
        },
    )
    _write_text(
        folder / "source_script.txt",
        "\n".join(
            [
                "Command:",
                "python -m genomecf.cli build-submission-data",
                "",
                "Logic:",
                "- Load results/release/biological_case_study.csv.",
                "- Merge top-k precision from results/release/external_validation_summary.csv using task, model and condition.",
                "- Encode decision-role flags for GenomeCF-aware and AUROC-only choices.",
            ]
        ),
    )
    return {
        "sheet_name": "Fig5_MPRA_case_studies",
        "sheet_df": fig5,
        "data_files": [folder / "fig5_mpra_case_studies.csv", folder / "source_registry_rows.csv"],
        "manifest_rows": [
            {
                "figure_or_table": "Fig. 5",
                "folder": "source_data/Fig5",
                "file_name": "fig5_mpra_case_studies.csv",
                "description": "Case-study metrics for BCL11A and MYC MPRA decision-change panels.",
            }
        ],
    }


def _build_fig6_source(main_figures: list[DisplayItem]) -> dict[str, object]:
    folder = SOURCE_DATA_ROOT / "Fig6"
    folder.mkdir(parents=True, exist_ok=True)
    figure = next(item for item in main_figures if item.number == 6)
    synth = pd.read_csv(RELEASE_ROOT / "synthetic_extended_summary.csv").copy()
    synth["condition"] = "official"
    synth["AUPRC"] = pd.NA
    synth["source_file"] = "results/release/synthetic_extended_summary.csv"
    fig6 = synth[
        [
            "task_id",
            "task_label",
            "model_id",
            "model_label",
            "condition",
            "auroc",
            "AUPRC",
            "ece",
            "brier",
            "rc_mean_abs_delta",
            "mono_positive_prob_drop",
            "dinuc_positive_prob_drop",
            "motif_positive_prob_drop",
            "rule_following_rate",
            "shortcut_following_rate",
            "shortcut_conflict_accuracy",
            "source_file",
        ]
    ].rename(
        columns={
            "task_id": "synthetic_task",
            "auroc": "AUROC",
            "ece": "ECE",
            "brier": "Brier",
            "rc_mean_abs_delta": "RC_delta",
            "mono_positive_prob_drop": "mono_drop",
            "dinuc_positive_prob_drop": "dinuc_drop",
            "motif_positive_prob_drop": "motif_disruption_drop",
            "shortcut_conflict_accuracy": "conflict_accuracy",
        }
    )
    fig6.to_csv(folder / "fig6_genomecf_synth.csv", index=False)
    synth.to_csv(folder / "source_registry_rows.csv", index=False)
    _write_text(
        folder / "README.md",
        "\n".join(
            [
                "# Fig. 6 source data",
                "",
                "This folder contains the source data for the GenomeCF-Synth figure.",
                f"- Manuscript figure: Fig. 6",
                f"- Figure title: {figure.caption}",
                "- Source files: `results/release/synthetic_extended_summary.csv`",
                "- Generation script: `package_src/genomecf/submission_data.py` via `genomecf build-submission-data`",
                "",
                "Note: the current extended synthetic summary does not store AUPRC, so the `AUPRC` column is left blank rather than back-filled from another source.",
            ]
        ),
    )
    _write_metadata(
        folder / "metadata.json",
        {
            "figure_or_table": "Fig. 6",
            "title": figure.caption,
            "manuscript_file": "paper/genomecf_report.tex",
            "source_files": ["results/release/synthetic_extended_summary.csv"],
            "generation_script": "package_src/genomecf/submission_data.py",
            "date_generated": _utc_timestamp(),
            "registry_hash": _sha256_file(RELEASE_ROOT / "benchmark_registry.csv"),
            "notes": "Extended synthetic summary includes AUROC, calibration and shortcut/rule-following metrics.",
            "license_or_restrictions": "Derived benchmark summary only.",
        },
    )
    _write_text(
        folder / "source_script.txt",
        "\n".join(
            [
                "Command:",
                "python -m genomecf.cli build-submission-data",
                "",
                "Logic:",
                "- Load results/release/synthetic_extended_summary.csv.",
                "- Rename columns to match the manuscript-facing source-data field names.",
                "- Preserve blank AUPRC values because the extended synthetic summary does not record them.",
            ]
        ),
    )
    return {
        "sheet_name": "Fig6_GenomeCF_Synth",
        "sheet_df": fig6,
        "data_files": [folder / "fig6_genomecf_synth.csv", folder / "source_registry_rows.csv"],
        "manifest_rows": [
            {
                "figure_or_table": "Fig. 6",
                "folder": "source_data/Fig6",
                "file_name": "fig6_genomecf_synth.csv",
                "description": "GenomeCF-Synth task metrics for shortcut and rule-following comparisons.",
            }
        ],
    }


def _column_descriptions() -> dict[str, str]:
    return {
        "task_id": "Internal task identifier used throughout the release registry.",
        "task_label": "Human-readable task label used in manuscript figures.",
        "model_id": "Internal model identifier.",
        "model_label": "Human-readable model label used in the paper.",
        "AUROC": "Area under the receiver operating characteristic curve.",
        "AUPRC": "Area under the precision-recall curve.",
        "ECE": "Expected calibration error.",
        "Brier": "Brier score.",
        "RC_delta": "Reverse-complement mean absolute probability delta.",
        "mono_drop": "Positive-class probability drop after mononucleotide shuffle.",
        "dinuc_drop": "Positive-class probability drop after dinucleotide shuffle.",
        "matched_negative_AUROC": "AUROC on the matched-negative evaluation split.",
        "top_k_enrichment": "Enrichment among the highest-ranked variants in the MPRA case study.",
        "top_k_precision": "Precision among the top-ranked variants in the MPRA case study.",
        "Spearman": "Spearman correlation with absolute measured variant effect.",
        "worst_GC_bin_AUROC": "Worst-bin AUROC across GC composition quantile bins.",
        "external_biological_reliability": "Composite external biological reliability score used in the manuscript.",
        "external_reliability_risk": "Risk-oriented complement of the external biological reliability score.",
        "full_GenomeCF_profile_score": "Linear score from the full GenomeCF profile regression (without intercept).",
        "predicted_external_reliability": "Predicted external biological reliability from the fitted full GenomeCF profile regression.",
        "rule_following_rate": "Fraction of synthetic examples on which the model followed the intended mechanistic rule.",
        "shortcut_following_rate": "Fraction of synthetic examples on which the model followed the shortcut cue.",
        "conflict_accuracy": "Accuracy on the synthetic shortcut-conflict evaluation split.",
    }


def _build_source_data_package(main_figures: list[DisplayItem], main_tables: list[DisplayItem]) -> dict[str, object]:
    _clean_dir(SOURCE_DATA_ROOT)
    benchmark_summary = pd.read_csv(RELEASE_ROOT / "benchmark_summary.csv")
    build_results = [
        _build_fig1_source(main_figures),
        _build_fig2_source(main_figures, benchmark_summary),
        _build_fig3_source(main_figures),
        _build_fig4_source(main_figures),
        _build_fig5_source(main_figures),
        _build_fig6_source(main_figures),
    ]

    root_readme = "\n".join(
        [
            "# GenomeCF Source Data",
            "",
            "This archive contains the numerical source data underlying the main figures and tables in “GenomeCF: a counterfactual validation standard for DNA sequence models.” Values are derived from registry-backed result files. Raw public datasets and pretrained model checkpoints are not redistributed here; dataset/model sources are listed in the manuscript and repository manifests.",
        ]
    )
    _write_text(SOURCE_DATA_ROOT / "README.md", root_readme)

    dictionary_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    workbook_sheets: list[tuple[str, pd.DataFrame]] = [
        ("README", pd.DataFrame({"note": root_readme.splitlines()})),
    ]
    for item in build_results:
        workbook_sheets.append((item["sheet_name"], item["sheet_df"]))
        for extra in item.get("extra_sheets", []):
            workbook_sheets.append(extra)
        manifest_rows.extend(item["manifest_rows"])
        for data_file in item["data_files"]:
            if data_file.suffix.lower() != ".csv":
                continue
            frame = pd.read_csv(data_file)
            for column in frame.columns:
                dictionary_rows.append(
                    {
                        "figure_or_table": data_file.parent.name,
                        "file_name": data_file.name,
                        "column_name": column,
                        "description": _column_descriptions().get(column, f"Column copied from {data_file.name}."),
                    }
                )

    if main_tables:
        # The current manuscript has no main tables, but this branch preserves future compatibility.
        manifest_rows.append(
            {
                "figure_or_table": "Table 1",
                "folder": "source_data/Table1",
                "file_name": "table1_source_data.csv",
                "description": "Placeholder for main-paper table source data if a main table is restored.",
            }
        )

    dictionary_df = pd.DataFrame(dictionary_rows).drop_duplicates()
    dictionary_df.to_csv(SOURCE_DATA_ROOT / "Data_Dictionary.csv", index=False)
    workbook_sheets.append(("Data_Dictionary", dictionary_df))
    workbook_path = SOURCE_DATA_ROOT / "GenomeCF_Source_Data.xlsx"
    _write_workbook(workbook_path, workbook_sheets)
    manifest_payload = {
        "generated_at": _utc_timestamp(),
        "main_figures": [item.caption for item in main_figures],
        "main_tables": [item.caption for item in main_tables],
        "files": manifest_rows,
    }
    _write_metadata(SOURCE_DATA_ROOT / "manifest.json", manifest_payload)
    return {
        "workbook": workbook_path,
        "manifest_rows": manifest_rows,
        "main_figures": main_figures,
        "main_tables": main_tables,
        "file_count": sum(1 for path in SOURCE_DATA_ROOT.rglob("*") if path.is_file()),
    }


def _copy_package_file(rel_path: str, missing: list[str], manifest_rows: list[dict[str, object]], description: str, related: str, source_or_derived: str, notes: str = "") -> None:
    src = PROJECT_ROOT / rel_path
    dst = SUPPLEMENTARY_ROOT / rel_path
    if not src.exists():
        missing.append(rel_path)
        return
    _copy_if_exists(src, dst)
    manifest_rows.append(
        {
            "file_path": rel_path.replace("\\", "/"),
            "description": description,
            "source_or_derived": source_or_derived,
            "related_figure_or_table": related,
            "source_script": "existing release/publication artifact",
            "notes": notes,
        }
    )


def _build_reporting_checklist_sheet() -> pd.DataFrame:
    payload = _load_json(DOCS_ROOT / "reporting_checklist.yaml")
    rows: list[dict[str, object]] = []
    for item in payload.get("items", []):
        rows.append(
            {
                "id": item.get("id"),
                "description": item.get("description"),
                "required_columns": ", ".join(item.get("required_columns", [])),
                "filters": json.dumps(item.get("filters", {}), ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def _build_supplementary_package() -> dict[str, object]:
    _clean_dir(SUPPLEMENTARY_ROOT)
    manifest_rows: list[dict[str, object]] = []
    missing: list[str] = []
    expected_files = [
        ("results/release/benchmark_registry.csv", "Canonical long-form benchmark registry.", "derived", "Supplementary Data / Registry"),
        ("results/release/benchmark_summary.csv", "Canonical release summary table.", "derived", "Supplementary Data / Registry"),
        ("results/release/external_validation_summary.csv", "External validation matrix.", "derived", "Fig. 4 / Supplementary Tables"),
        ("results/release/external_validation_family_summary.csv", "External assay-family summary table.", "derived", "Fig. 4"),
        ("results/release/external_transfer_prediction.csv", "Point-level external prediction analysis inputs.", "derived", "Fig. 4"),
        ("results/release/external_transfer_stats.json", "External prediction statistics and fitted models.", "derived", "Fig. 4"),
        ("results/release/biological_case_study.csv", "MPRA biological case-study summary.", "derived", "Fig. 5"),
        ("results/release/synthetic_extended_summary.csv", "GenomeCF-Synth summary.", "derived", "Fig. 6"),
        ("results/release/mitigation_summary.csv", "Mitigation summary.", "derived", "Fig. 3 / Supplementary"),
        ("results/release/chromosome_cv_summary.csv", "Chromosome grouped CV summary.", "derived", "Supplementary"),
        ("results/release/chromosome_cv_fold_metrics.csv", "Per-fold chromosome CV metrics.", "derived", "Supplementary"),
        ("results/release/matched_negative_model_summary.csv", "Matched-negative model summary.", "derived", "Supplementary"),
        ("results/release/matched_negative_confounders.csv", "Matched-negative confounder summary.", "derived", "Supplementary"),
        ("results/release/gc_bin_summary.csv", "GC-bin robustness summary.", "derived", "Supplementary"),
        ("results/release/motif_summary.csv", "Motif summary (legacy expected path).", "derived", "Supplementary"),
        ("results/release/real_motif_probe_summary.csv", "Real-task motif probe summary.", "derived", "Supplementary"),
        ("results/release/paper_claim_traceability.csv", "Paper-claim traceability table.", "derived", "Traceability"),
        ("results/release/paper_claim_traceability.html", "HTML traceability report.", "derived", "Traceability"),
        ("results/release/statistical_claims.csv", "Statistical support table for headline claims.", "derived", "Supplementary"),
        ("results/release/benchmark_registry.jsonl", "JSONL version of the canonical registry.", "derived", "Supplementary Data / Registry"),
        ("docs/reporting_checklist.yaml", "Machine-readable reporting checklist.", "source", "Reporting checklist"),
        ("docs/reporting_checklist_schema.json", "Reporting checklist schema.", "source", "Reporting checklist"),
        ("docs/example_completed_checklist.md", "Example completed checklist.", "source", "Reporting checklist"),
        ("docs/RESULT_SCHEMA.md", "Registry schema documentation.", "source", "Supplementary Data / Registry"),
        ("docs/DATA_AVAILABILITY.md", "Data availability statement.", "source", "Availability"),
        ("docs/CODE_AVAILABILITY.md", "Code availability statement.", "source", "Availability"),
        ("docs/REPRODUCIBILITY.md", "Reproducibility guide.", "source", "Availability"),
        ("docs/METRICS.md", "Metric definitions.", "source", "Methods"),
        ("docs/SPLITS.md", "Split definitions.", "source", "Methods"),
        ("docs/MODELS.md", "Model descriptions.", "source", "Methods"),
    ]
    for rel_path, description, source_or_derived, related in expected_files:
        _copy_package_file(rel_path, missing, manifest_rows, description, related, source_or_derived)

    # Include all publication CSVs.
    publication_dir = PUBLICATION_ROOT
    for csv_path in sorted(publication_dir.glob("*.csv")):
        rel_path = csv_path.relative_to(PROJECT_ROOT).as_posix()
        _copy_package_file(
            rel_path,
            missing,
            manifest_rows,
            f"Publication-facing derived table {csv_path.name}.",
            "Main paper / Supplementary tables",
            "derived",
        )

    if missing:
        _write_text(
            SUPPLEMENTARY_ROOT / "MISSING_EXPECTED_FILES.md",
            "\n".join(["# Missing expected files", ""] + [f"- `{item}`" for item in missing]),
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(SUPPLEMENTARY_ROOT / "FILE_MANIFEST.csv", index=False)

    readme_text = "\n".join(
        [
            "# GenomeCF Supplementary Data and Registry",
            "",
            "This package contains the canonical GenomeCF benchmark registry, derived publication tables, external validation summaries, traceability records, reporting-checklist materials and supporting metadata for the manuscript.",
            "",
            "- Canonical registry: `results/release/benchmark_registry.csv`",
            "- Main release summary: `results/release/benchmark_summary.csv`",
            "- Publication-derived summary tables: `results/publication/*.csv`",
            "- Traceability: `results/release/paper_claim_traceability.csv` and `results/release/statistical_claims.csv`",
            "",
            "How to load the registry:",
            "```python",
            "import pandas as pd",
            "df = pd.read_csv('results/release/benchmark_registry.csv')",
            "```",
            "",
            "How to regenerate figures/tables from the repository:",
            "- `python -m genomecf.cli build-paper`",
            "- `python -m genomecf.cli build-supplement`",
            "- `python -m genomecf.cli build-submission-data`",
            "",
            "Source data versus derived summaries:",
            "- `source_data/` in the companion source-data archive contains the numerical inputs for the main display items.",
            "- This registry archive contains the canonical registry plus derived summary tables and supporting metadata.",
            "",
            "What is not included:",
            "- raw public benchmark datasets",
            "- pretrained model checkpoints",
            "- heavyweight embedding caches",
            "",
            "GitHub repository:",
            "https://github.com/AhanBhatt/GenomeCF",
        ]
    )
    _write_text(SUPPLEMENTARY_ROOT / "README.md", readme_text)

    stats = _load_json(RELEASE_ROOT / "external_transfer_stats.json")
    external_model_fits = _flatten_external_model_fits(stats)
    external_prediction = pd.read_csv(RELEASE_ROOT / "external_transfer_prediction.csv")
    registry = pd.read_csv(RELEASE_ROOT / "benchmark_registry.csv")
    benchmark_summary = pd.read_csv(RELEASE_ROOT / "benchmark_summary.csv")
    external_validation = pd.read_csv(RELEASE_ROOT / "external_validation_summary.csv")
    family_summary = pd.read_csv(RELEASE_ROOT / "external_validation_family_summary.csv")
    case_studies = pd.read_csv(RELEASE_ROOT / "biological_case_study.csv")
    synthetic = pd.read_csv(RELEASE_ROOT / "synthetic_extended_summary.csv")
    mitigation = pd.read_csv(RELEASE_ROOT / "mitigation_summary.csv")
    chromosome_cv = pd.read_csv(RELEASE_ROOT / "chromosome_cv_summary.csv")
    chromosome_folds = pd.read_csv(RELEASE_ROOT / "chromosome_cv_fold_metrics.csv")
    matched_negative = pd.read_csv(RELEASE_ROOT / "matched_negative_model_summary.csv")
    matched_confounders = pd.read_csv(RELEASE_ROOT / "matched_negative_confounders.csv")
    gc_bin = pd.read_csv(RELEASE_ROOT / "gc_bin_summary.csv")
    motif = pd.read_csv(RELEASE_ROOT / "real_motif_probe_summary.csv") if (RELEASE_ROOT / "real_motif_probe_summary.csv").exists() else pd.DataFrame()
    statistical = pd.read_csv(RELEASE_ROOT / "statistical_claims.csv")
    traceability = pd.read_csv(RELEASE_ROOT / "paper_claim_traceability.csv")
    reporting = _build_reporting_checklist_sheet()

    workbook_path = SUPPLEMENTARY_ROOT / "GenomeCF_Supplementary_Data_and_Registry.xlsx"
    _write_workbook(
        workbook_path,
        [
            ("README", pd.DataFrame({"note": readme_text.splitlines()})),
            ("File_Manifest", manifest_df),
            ("Benchmark_Summary", benchmark_summary),
            ("Main_Registry", registry),
            ("External_Validation", external_validation),
            ("External_Family_Summary", family_summary),
            ("External_Prediction", external_prediction),
            ("External_Model_Fits", external_model_fits),
            ("MPRA_Case_Studies", case_studies),
            ("Synthetic_Summary", synthetic),
            ("Mitigation_Summary", mitigation),
            ("Chromosome_CV_Summary", chromosome_cv),
            ("Chromosome_CV_Folds", chromosome_folds),
            ("Matched_Negative", matched_negative),
            ("Matched_Confounders", matched_confounders),
            ("GC_Bin", gc_bin),
            ("Motif_Probes", motif if not motif.empty else pd.DataFrame({"note": ["No motif summary file was available."]})),
            ("Statistical_Claims", statistical),
            ("Claim_Traceability", traceability),
            ("Reporting_Checklist", reporting),
        ],
    )

    checksum_lines: list[str] = []
    for path in sorted(p for p in SUPPLEMENTARY_ROOT.rglob("*") if p.is_file() and p.name != "CHECKSUMS_SHA256.txt"):
        checksum_lines.append(f"{_sha256_file(path)}  {path.relative_to(SUPPLEMENTARY_ROOT).as_posix()}")
    _write_text(SUPPLEMENTARY_ROOT / "CHECKSUMS_SHA256.txt", "\n".join(checksum_lines) + ("\n" if checksum_lines else ""))

    return {
        "workbook": workbook_path,
        "missing": missing,
        "manifest": manifest_df,
        "file_count": sum(1 for path in SUPPLEMENTARY_ROOT.rglob("*") if path.is_file()),
    }


def _zip_directory(src_dir: Path, zip_path: Path) -> int:
    _ensure_parent(zip_path)
    count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in src_dir.rglob("*") if p.is_file()):
            zf.write(path, arcname=path.relative_to(src_dir.parent).as_posix())
            count += 1
    return count


def _build_submission_report(source_payload: dict[str, object], supplementary_payload: dict[str, object], source_zip: Path, supplementary_zip: Path) -> Path:
    report_path = SUBMISSION_UPLOADS_ROOT / "SUBMISSION_DATA_PACKAGE_REPORT.md"
    source_size = source_zip.stat().st_size if source_zip.exists() else 0
    supp_size = supplementary_zip.stat().st_size if supplementary_zip.exists() else 0
    missing = supplementary_payload.get("missing", [])
    missing_lines = [f"- `{item}`" for item in missing] if missing else ["- None"]
    figure_folders = [f"Fig{i}" for i in range(1, 7)]
    validation_lines = []
    for fig_folder in figure_folders:
        folder_path = SOURCE_DATA_ROOT / fig_folder
        has_folder = folder_path.exists()
        has_payload = any(folder_path.glob("*.csv")) or any(folder_path.glob("*.json"))
        validation_lines.append(f"- {fig_folder}: {'pass' if has_folder and has_payload else 'fail'}")
    critical_ok = (SUPPLEMENTARY_ROOT / "results" / "release" / "benchmark_registry.csv").exists()
    report_text = "\n".join(
        [
            "# Submission Data Package Report",
            "",
            "## Created files",
            "",
            f"- `submission_uploads/GenomeCF_Source_Data.zip` ({source_size:,} bytes)",
            f"- `submission_uploads/GenomeCF_Supplementary_Data_and_Registry.zip` ({supp_size:,} bytes)",
            f"- Source-data workbook: `{source_payload['workbook']}`",
            f"- Registry workbook: `{supplementary_payload['workbook']}`",
            "",
            "## Package counts",
            "",
            f"- Source-data files: {source_payload['file_count']}",
            f"- Supplementary registry/data files: {supplementary_payload['file_count']}",
            "",
            "## Validation",
            "",
            "- Main figures with source-data folders:",
            *validation_lines,
            f"- Canonical registry included: {'pass' if critical_ok else 'fail'}",
            f"- Missing expected files: {len(missing)}",
            "",
            "## Missing expected files",
            "",
            *missing_lines,
            "",
            "## Recommended upload entries",
            "",
            "File type: Source Data",
            "File title: Source Data for GenomeCF",
            "Upload file name: GenomeCF_Source_Data.zip",
            "Description: Numerical source data underlying the main figures and tables in “GenomeCF: a counterfactual validation standard for DNA sequence models.”",
            "",
            "File type: Supplementary and Additional Material or Supplementary Data",
            "File title: GenomeCF Supplementary Data and Registry",
            "Upload file name: GenomeCF_Supplementary_Data_and_Registry.zip",
            "Description: Canonical benchmark registry, derived summary tables, external validation results, statistical claims, reporting checklist and traceability files supporting the GenomeCF manuscript.",
        ]
    )
    _write_text(report_path, report_text)
    return report_path


def build_submission_data(*, run_validation: bool = True) -> dict[str, object]:
    if run_validation:
        report = validate_release_results()
        if not report.ok:
            raise ValueError("Release validation failed before building submission data: " + "; ".join(report.errors))

    main_figures, main_tables = parse_main_display_items()
    if len(main_figures) != 6:
        raise ValueError(f"Expected 6 main figures in the final manuscript, found {len(main_figures)}.")

    source_payload = _build_source_data_package(main_figures, main_tables)
    supplementary_payload = _build_supplementary_package()

    _clean_dir(SUBMISSION_UPLOADS_ROOT)
    source_zip = SUBMISSION_UPLOADS_ROOT / "GenomeCF_Source_Data.zip"
    supplementary_zip = SUBMISSION_UPLOADS_ROOT / "GenomeCF_Supplementary_Data_and_Registry.zip"
    source_zip_count = _zip_directory(SOURCE_DATA_ROOT, source_zip)
    supplementary_zip_count = _zip_directory(SUPPLEMENTARY_ROOT, supplementary_zip)
    report_path = _build_submission_report(source_payload, supplementary_payload, source_zip, supplementary_zip)

    payload = {
        "source_zip": str(source_zip),
        "supplementary_zip": str(supplementary_zip),
        "source_workbook": str(source_payload["workbook"]),
        "registry_workbook": str(supplementary_payload["workbook"]),
        "source_zip_file_count": source_zip_count,
        "supplementary_zip_file_count": supplementary_zip_count,
        "canonical_registry_included": (SUPPLEMENTARY_ROOT / "results" / "release" / "benchmark_registry.csv").exists(),
        "missing_expected_files": supplementary_payload["missing"],
        "report_path": str(report_path),
        "main_figure_count": len(main_figures),
        "main_table_count": len(main_tables),
    }
    (SUBMISSION_UPLOADS_ROOT / "submission_data_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
