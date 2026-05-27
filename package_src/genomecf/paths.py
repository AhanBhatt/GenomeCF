from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_RUNTIME_ROOT = PROJECT_ROOT.parent / "local_runtime_assets"
CONFIG_ROOT = PROJECT_ROOT / "configs"
RESULTS_ROOT = PROJECT_ROOT / "results"
RELEASE_ROOT = RESULTS_ROOT / "release"
PUBLICATION_ROOT = RESULTS_ROOT / "publication"
GENOMECF_REGISTRY_ROOT = RESULTS_ROOT / "genomecf_registry"
FIGURES_ROOT = PROJECT_ROOT / "figures"
DOCS_ROOT = PROJECT_ROOT / "docs"
_local_data_root = PROJECT_ROOT / "data"
_sibling_data_root = LOCAL_RUNTIME_ROOT / "data"
DATA_ROOT = _local_data_root if _local_data_root.exists() else _sibling_data_root
_local_external_root = PROJECT_ROOT / "external"
_sibling_external_root = LOCAL_RUNTIME_ROOT / "external"
EXTERNAL_ROOT = _local_external_root if _local_external_root.exists() else _sibling_external_root
_local_cache_root = RESULTS_ROOT / "cache"
_sibling_cache_root = LOCAL_RUNTIME_ROOT / "results" / "cache"
CACHE_ROOT = _local_cache_root if _local_cache_root.exists() else _sibling_cache_root
_local_paper_root = PROJECT_ROOT / "paper"
_sibling_paper_root = PROJECT_ROOT.parent / "paper"
PAPER_ROOT = _local_paper_root if _local_paper_root.exists() else _sibling_paper_root


def normalize_legacy_path_text(value: str | None) -> str | None:
    """Normalize previously-recorded path strings.

    This repo has existed in a few on-disk layouts during development (sibling results folders,
    duplicated `GenomeCF/GenomeCF` roots, and JSON-escaped backslashes). For public release
    artifacts we also avoid leaking absolute local paths by rewriting paths under the repo
    root to repo-relative strings.
    """

    if value is None:
        return None

    text = str(value)

    def _replace_all(old: str, new: str) -> None:
        nonlocal text
        variants = {
            old,
            old.replace("\\", "/"),
            old.replace("\\", "\\\\"),
        }
        for v in variants:
            if v in text:
                nv = new
                if v.endswith("\\\\") or "\\\\" in v:
                    nv = new.replace("\\", "\\\\")
                text = text.replace(v, nv)

    # Fix legacy repo-root mistakes.
    _replace_all(str(PROJECT_ROOT / "GenomeCF"), str(PROJECT_ROOT))

    # Fix cases where results/figures were recorded under the parent project directory.
    for sub in ("results", "figures", "data", "external"):
        _replace_all(str(PROJECT_ROOT.parent / sub), str(PROJECT_ROOT / sub))

    # Finally, strip absolute repo root prefixes to avoid leaking local paths.
    _replace_all(str(PROJECT_ROOT), ".")

    return text
