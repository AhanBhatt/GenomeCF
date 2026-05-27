from __future__ import annotations

import json
import re
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
    summary, external_family = _load_release_summary(regenerate=regenerate)
    output_dir = output_dir or (DOCS_ROOT / "site")
    output_dir.mkdir(parents=True, exist_ok=True)

    official = summary[
        (summary["split_id"] == "official")
        & (summary["calibration_method"] == "none")
        & (summary["intervention_id"] == "standard")
        & (summary["tier"].isin(["core", "external", "synthetic"]))
    ].copy()

    def _safe_read_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    gc_bins = _safe_read_csv(RELEASE_ROOT / "gc_bin_summary.csv")
    matched_neg = _safe_read_csv(RELEASE_ROOT / "matched_negative_model_summary.csv")
    external_pairs = _safe_read_csv(RELEASE_ROOT / "external_transfer_prediction.csv")

    leaderboard_rows = official.rename(
        columns={
            "task_readable_name": "task_label",
            "model_readable_name": "model_label",
        }
    )

    if not gc_bins.empty:
        gc_agg = (
            gc_bins.groupby(["task_id", "model_id"], as_index=False)
            .agg(worst_gc_bin_auroc=("worst_bin_auroc", "mean"), gc_bin_auroc_gap=("gc_bin_auroc_gap", "mean"))
        )
        leaderboard_rows = leaderboard_rows.merge(gc_agg, on=["task_id", "model_id"], how="left")

    if not matched_neg.empty:
        matched_neg = matched_neg[matched_neg.get("split_id", "") == "official"].copy() if "split_id" in matched_neg.columns else matched_neg
        matched_agg = matched_neg[["task_id", "model_id", "gc_only_auroc_official", "gc_only_auroc_matched"]].drop_duplicates()
        matched_agg["matched_negative_gc_only_auroc_drop"] = matched_agg["gc_only_auroc_official"] - matched_agg["gc_only_auroc_matched"]
        leaderboard_rows = leaderboard_rows.merge(matched_agg, on=["task_id", "model_id"], how="left")

    keep_cols = [
        "tier",
        "track",
        "species",
        "task_id",
        "task_label",
        "model_id",
        "model_label",
        "model_family",
        "auroc",
        "auprc",
        "ece",
        "brier",
        "rc_mean_abs_delta",
        "mono_positive_prob_drop",
        "dinuc_positive_prob_drop",
        "gc_only_auroc",
        "gc_only_explainability_ratio",
        "shortcut_score",
        "worst_gc_bin_auroc",
        "gc_bin_auroc_gap",
        "matched_negative_gc_only_auroc_drop",
    ]
    leaderboard_rows = leaderboard_rows[[c for c in keep_cols if c in leaderboard_rows.columns]].copy()
    leaderboard_rows.to_csv(output_dir / "leaderboard_rows.csv", index=False)

    leaderboard_models = (
        official.rename(columns={"model_readable_name": "model_label"})
        .groupby(["model_id", "model_label", "model_family"], as_index=False)
        .agg(
            mean_auroc=("auroc", "mean"),
            mean_auprc=("auprc", "mean"),
            mean_ece=("ece", "mean"),
            mean_shortcut_score=("shortcut_score", "mean"),
            mean_rc_delta=("rc_mean_abs_delta", "mean"),
            mean_gc_only_ratio=("gc_only_explainability_ratio", "mean"),
        )
    )

    if not gc_bins.empty:
        gc_model = gc_bins.groupby(["model_id"], as_index=False).agg(mean_worst_gc_bin_auroc=("worst_bin_auroc", "mean"), mean_gc_bin_auroc_gap=("gc_bin_auroc_gap", "mean"))
        leaderboard_models = leaderboard_models.merge(gc_model, on="model_id", how="left")

    if not matched_neg.empty:
        mn_model = matched_agg.groupby("model_id", as_index=False).agg(mean_matched_negative_gc_only_auroc_drop=("matched_negative_gc_only_auroc_drop", "mean"))
        leaderboard_models = leaderboard_models.merge(mn_model, on="model_id", how="left")

    if not external_family.empty:
        ext_scores = (
            external_family.groupby("model_id", as_index=False)["external_biological_reliability"]
            .mean()
            .rename(columns={"external_biological_reliability": "external_validation_score"})
        )
        leaderboard_models = leaderboard_models.merge(ext_scores, on="model_id", how="left")
        leaderboard_models["mean_external_biological_reliability"] = leaderboard_models["external_validation_score"]

    if not external_pairs.empty:
        ext_pair_model = external_pairs.groupby("model_id", as_index=False).agg(mean_external_reliability_risk=("external_reliability_risk", "mean"))
        leaderboard_models = leaderboard_models.merge(ext_pair_model, on="model_id", how="left")

        feature_cols = [
            "core_mean_auroc",
            "core_mean_shortcut_score",
            "core_mean_rc_delta",
            "core_mean_ece",
            "core_matched_negative_shift",
            "core_gc_bin_auroc_gap",
        ]
        train = external_pairs.dropna(subset=["external_biological_reliability"] + feature_cols).copy()
        if not train.empty:
            import numpy as np

            X = train[feature_cols].to_numpy(dtype=float)
            y = train["external_biological_reliability"].to_numpy(dtype=float)
            X = np.c_[np.ones(len(X)), X]
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            agg = external_pairs.groupby("model_id", as_index=False)[feature_cols].mean()
            Xp = np.c_[np.ones(len(agg)), agg[feature_cols].to_numpy(dtype=float)]
            agg["full_profile_predicted_external_reliability_in_sample"] = Xp @ beta
            leaderboard_models = leaderboard_models.merge(
                agg[["model_id", "full_profile_predicted_external_reliability_in_sample"]],
                on="model_id",
                how="left",
            )

    leaderboard_models.to_csv(output_dir / "leaderboard.csv", index=False)

    nav = """
<nav>
  <a href="index.html">Overview</a>
  <a href="quickstart.html">Quickstart</a>
  <a href="tasks.html">Tasks</a>
  <a href="models.html">Models</a>
  <a href="metrics.html">Metrics</a>
  <a href="external.html">External validation</a>
  <a href="synthetic.html">GenomeCF-Synth</a>
  <a href="leaderboard.html">Leaderboard</a>
  <a href="reporting_standard.html">Reporting checklist</a>
  <a href="reproducibility.html">Reproducibility</a>
  <a href="downloads.html">Downloads</a>
  <a href="citation.html">Citation</a>
</nav>
""".strip()

    index_html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>GenomeCF</title>
  <link rel=\"stylesheet\" href=\"style.css\">
</head>
<body>
{nav}

<div class=\"hero\">
  <h1>GenomeCF</h1>
  <p class=\"small\">Counterfactual stress testing and external validation for DNA sequence models.</p>
  <p>
    <a href=\"leaderboard.html\">Leaderboard</a> ·
    <a href=\"external.html\">External validation</a> ·
    <a href=\"synthetic.html\">GenomeCF-Synth</a>
  </p>
  <p class=\"small\">
    PDFs: <a href=\"../../paper/genomecf_report.pdf\">paper</a>, <a href=\"../../paper/genomecf_supplement.pdf\">supplement</a>
  </p>
</div>

<h2>Key commands</h2>
<pre><code>pip install -e .[benchmark,dev]

genomecf reproduce-quickstart
genomecf validate-results
genomecf check-report --results results/release/benchmark_registry.csv
genomecf trace-paper --strict
</code></pre>

<h2>Downloads</h2>
<ul>
  <li><a href=\"leaderboard.csv\">leaderboard.csv</a> (model-level)</li>
  <li><a href=\"leaderboard_rows.csv\">leaderboard_rows.csv</a> (task-model rows)</li>
  <li><a href=\"../../results/release/benchmark_registry.csv\">benchmark_registry.csv</a></li>
  <li><a href=\"../../results/release/paper_claim_traceability.html\">paper claim traceability</a></li>
</ul>
</body>
</html>
"""
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")

    leaderboard_html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>GenomeCF leaderboard</title>
  <link rel=\"stylesheet\" href=\"style.css\">
</head>
<body>
{nav}
<h1>Leaderboard</h1>
<p class=\"small\">This page is generated from the release registry. Use the search box to filter rows.</p>
<p><a href=\"leaderboard.csv\">leaderboard.csv</a> · <a href=\"leaderboard_rows.csv\">leaderboard_rows.csv</a></p>

<label>Search: <input id=\"q\" style=\"width: 42rem\" placeholder=\"model / task / tier\"></label>
<div id=\"status\" class=\"small\"></div>
<div style=\"overflow-x:auto;\"><table id=\"tbl\"></table></div>

<script>
async function loadCSV(path) {{
  const txt = await (await fetch(path)).text();
  const lines = txt.trim().split(/\r?\n/);
  const headers = lines[0].split(',');
  const rows = lines.slice(1).map(l => {{
    const cols = [];
    let cur = ''; let inQ = false;
    for (let i=0;i<l.length;i++) {{
      const ch = l[i];
      if (ch === '"') {{ inQ = !inQ; continue; }}
      if (ch === ',' && !inQ) {{ cols.push(cur); cur=''; continue; }}
      cur += ch;
    }}
    cols.push(cur);
    const obj = {{}};
    headers.forEach((h,i) => obj[h] = (cols[i] ?? ''));
    return obj;
  }});
  return {{headers, rows}};
}}

function renderTable(headers, rows) {{
  const tbl = document.getElementById('tbl');
  tbl.innerHTML = '';
  const thead = document.createElement('thead');
  const trh = document.createElement('tr');
  headers.forEach(h => {{
    const th = document.createElement('th');
    th.textContent = h;
    trh.appendChild(th);
  }});
  thead.appendChild(trh);
  tbl.appendChild(thead);

  const tbody = document.createElement('tbody');
  rows.forEach(r => {{
    const tr = document.createElement('tr');
    headers.forEach(h => {{
      const td = document.createElement('td');
      td.textContent = r[h] ?? '';
      tr.appendChild(td);
    }});
    tbody.appendChild(tr);
  }});
  tbl.appendChild(tbody);
}}

let data = null;
(async () => {{
  data = await loadCSV('leaderboard_rows.csv');
  renderTable(data.headers, data.rows);
  document.getElementById('status').textContent = data.rows.length + ' rows';
}})();

document.getElementById('q').addEventListener('input', (e) => {{
  if (!data) return;
  const q = e.target.value.toLowerCase();
  const rows = data.rows.filter(r => JSON.stringify(r).toLowerCase().includes(q));
  renderTable(data.headers, rows);
  document.getElementById('status').textContent = rows.length + ' rows (filtered)';
}});
</script>
</body>
</html>
"""
    (output_dir / "leaderboard.html").write_text(leaderboard_html, encoding="utf-8")

    quickstart_html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>GenomeCF quickstart</title>
  <link rel=\"stylesheet\" href=\"style.css\">
</head>
<body>
{nav}
<h1>Quickstart</h1>
<p>Verify the installed release and generate a small JSON report:</p>
<pre><code>genomecf reproduce-quickstart</code></pre>
</body>
</html>
"""
    (output_dir / "quickstart.html").write_text(quickstart_html, encoding="utf-8")

    reporting_html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>GenomeCF reporting checklist</title>
  <link rel=\"stylesheet\" href=\"style.css\">
</head>
<body>
{nav}
<h1>Reporting checklist</h1>
<p>See the machine-readable checklist at <code>docs/reporting_checklist.yaml</code> and the markdown summary in <code>docs/reporting_checklist.md</code>.</p>
</body>
</html>
"""
    (output_dir / "reporting_standard.html").write_text(reporting_html, encoding="utf-8")

    pages = sorted([p.name for p in output_dir.glob("*.html")])
    manifest = {
        "site_root": str(output_dir),
        "leaderboard_csv": str(output_dir / "leaderboard.csv"),
        "leaderboard_rows_csv": str(output_dir / "leaderboard_rows.csv"),
        "pages": pages,
    }
    (output_dir / "site_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return output_dir / "index.html"


def _parse_yaml_scalar(raw: str) -> object:
    value = raw.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    try:
        if any(token in lowered for token in (".", "e")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_claims_yaml(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    claims: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    in_claims = False
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.strip() == "claims:" and not in_claims:
            in_claims = True
            continue
        if not in_claims:
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            if current:
                claims.append(current)
            current = {}
            remainder = stripped[2:]
            if ":" in remainder:
                key, raw = remainder.split(":", 1)
                current[key.strip()] = _parse_yaml_scalar(raw.strip())
            continue
        if current is None:
            continue
        if ":" not in stripped:
            continue
        key, raw = stripped.split(":", 1)
        current[key.strip()] = _parse_yaml_scalar(raw.strip())
    if current:
        claims.append(current)
    return claims


_LATEX_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("input", re.compile(r"\\input\{([^}]+)\}")),
    ("includegraphics", re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")),
    ("addbibresource", re.compile(r"\\addbibresource\{([^}]+)\}")),
    ("bibliography", re.compile(r"\\bibliography\{([^}]+)\}")),
]


def _latex_references(tex_path: Path) -> list[dict[str, str]]:
    try:
        text = tex_path.read_text(encoding="utf-8")
    except OSError:
        return []
    refs: list[dict[str, str]] = []
    for kind, pattern in _LATEX_PATTERNS:
        for match in pattern.findall(text):
            refs.append({"kind": kind, "raw": match})
    return refs


def _resolve_latex_ref(root: Path, ref: dict[str, str]) -> Path | None:
    raw = ref["raw"].strip()
    if not raw:
        return None
    if ref["kind"] == "bibliography":
        candidate = root / f"{raw}.bib"
        return candidate
    candidate = (root / raw)
    if candidate.exists():
        return candidate
    if candidate.suffix == "":
        for suffix in (".tex", ".bib", ".png", ".pdf", ".jpg", ".jpeg"):
            alt = candidate.with_suffix(suffix)
            if alt.exists():
                return alt
    return candidate


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and pd.notna(value)


def trace_paper(output_path: Path | None = None, strict: bool = False) -> dict[str, object]:
    rows: list[dict[str, object]] = []

    key_numbers_path = PUBLICATION_ROOT / "key_numbers.json"
    key_numbers = json.loads(key_numbers_path.read_text(encoding="utf-8")) if key_numbers_path.exists() else {}
    for key, value in sorted(key_numbers.items()):
        rows.append(
            {
                "claim_id": f"key_numbers::{key}",
                "paper_location": "paper/main_or_supplement",
                "claim_text": key,
                "source_artifact": str(key_numbers_path),
                "script": "src/generate_publication_artifacts.py",
                "expected": "",
                "observed": value,
                "tolerance": "",
                "validated": bool(pd.notna(value)),
                "notes": "",
            }
        )

    stats_path = RELEASE_ROOT / "statistical_claims.csv"
    stats = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
    stats_by_id = stats.set_index("claim_id") if (not stats.empty and "claim_id" in stats.columns) else pd.DataFrame()

    claims_path = PAPER_ROOT / "claims.yaml"
    if claims_path.exists():
        try:
            claims = _load_claims_yaml(claims_path)
        except Exception as exc:  # pragma: no cover
            claims = []
            rows.append(
                {
                    "claim_id": "claims_yaml::parse_error",
                    "paper_location": "paper/claims.yaml",
                    "claim_text": "Failed to parse paper/claims.yaml",
                    "source_artifact": str(claims_path),
                    "script": "package_src/genomecf/nature_methods.py",
                    "expected": "",
                    "observed": "",
                    "tolerance": "",
                    "validated": False,
                    "notes": str(exc),
                }
            )
        for claim in claims:
            claim_id = str(claim.get("claim_id", ""))
            expected = claim.get("validated_value", None)
            tol = claim.get("tolerance", 0.0)
            source = str(claim.get("source_registry_file", ""))
            observed = None
            ok = False
            notes = ""
            if isinstance(stats_by_id, pd.DataFrame) and not stats_by_id.empty and claim_id in stats_by_id.index:
                observed = float(stats_by_id.loc[claim_id, "estimate"])
                if _is_number(expected) and _is_number(observed) and _is_number(tol):
                    ok = abs(float(observed) - float(expected)) <= float(tol)
                else:
                    ok = str(observed) == str(expected)
            else:
                notes = "Missing claim_id in results/release/statistical_claims.csv"
            source_path = PROJECT_ROOT / source if source else None
            if source_path is not None and not source_path.exists():
                ok = False
                notes = (notes + "; " if notes else "") + f"Missing source artifact: {source}"
            rows.append(
                {
                    "claim_id": claim_id,
                    "paper_location": str(claim.get("paper_location", "")),
                    "claim_text": str(claim.get("claim_text", "")),
                    "source_artifact": source,
                    "script": str(claim.get("source_script", "")),
                    "expected": expected,
                    "observed": observed,
                    "tolerance": tol,
                    "validated": bool(ok),
                    "notes": notes,
                }
            )
    else:
        rows.append(
            {
                "claim_id": "claims_yaml::missing",
                "paper_location": "paper/claims.yaml",
                "claim_text": "Missing paper/claims.yaml",
                "source_artifact": str(claims_path),
                "script": "",
                "expected": "",
                "observed": "",
                "tolerance": "",
                "validated": False,
                "notes": "",
            }
        )

    for tex_name in ("genomecf_report.tex", "genomecf_supplement.tex"):
        tex_path = PAPER_ROOT / tex_name
        if not tex_path.exists():
            rows.append(
                {
                    "claim_id": f"latex::{tex_name}::missing",
                    "paper_location": "paper",
                    "claim_text": f"Missing LaTeX source {tex_name}",
                    "source_artifact": str(tex_path),
                    "script": "",
                    "expected": "exists",
                    "observed": "missing",
                    "tolerance": "",
                    "validated": False,
                    "notes": "",
                }
            )
            continue
        for ref in _latex_references(tex_path):
            resolved = _resolve_latex_ref(PAPER_ROOT, ref)
            exists = bool(resolved is not None and resolved.exists())
            rows.append(
                {
                    "claim_id": f"latex::{tex_name}::{ref['kind']}::{ref['raw']}",
                    "paper_location": tex_name,
                    "claim_text": f"{ref['kind']}{{{ref['raw']}}}",
                    "source_artifact": ref["raw"],
                    "script": "paper",
                    "expected": "exists",
                    "observed": str(resolved) if resolved is not None else "",
                    "tolerance": "",
                    "validated": exists,
                    "notes": "",
                }
            )

    extra_paths = {
        "paper_pdf": PAPER_ROOT / "genomecf_report.pdf",
        "supplement_pdf": PAPER_ROOT / "genomecf_supplement.pdf",
        "website_index": DOCS_ROOT / "site" / "index.html",
    }
    for name, path in extra_paths.items():
        rows.append(
            {
                "claim_id": f"availability::{name}",
                "paper_location": "availability",
                "claim_text": str(path),
                "source_artifact": str(path),
                "script": "",
                "expected": "exists",
                "observed": "exists" if path.exists() else "missing",
                "tolerance": "",
                "validated": path.exists(),
                "notes": "",
            }
        )

    frame = pd.DataFrame(rows)
    validated = bool(frame["validated"].all()) if not frame.empty else False
    if strict and not validated:
        failures = frame[~frame["validated"]]
        raise ValueError(f"Strict traceability failed with {len(failures)} failing checks.")

    output_path = output_path or (RELEASE_ROOT / "paper_claim_traceability.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    frame.to_html(output_path.with_suffix(".html"), index=False)
    payload = {
        "validated": validated,
        "claim_count": int(len(frame)),
        "row_count": int(len(frame)),
        "failure_count": int((~frame["validated"]).sum()) if not frame.empty else 0,
        "csv_path": str(output_path),
        "html_path": str(output_path.with_suffix(".html")),
        "key_numbers_path": str(key_numbers_path),
        "claims_yaml_path": str(claims_path),
        "statistical_claims_path": str(stats_path),
    }
    output_path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
