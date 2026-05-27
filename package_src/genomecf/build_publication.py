from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .paths import FIGURES_ROOT, PAPER_ROOT, PROJECT_ROOT, PUBLICATION_ROOT
from .validation import validate_release_results


def _run_script(script_name: str) -> None:
    subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / script_name)], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)


def build_publication(*, skip_tests: bool = False, skip_latex: bool = False, skip_validation: bool = False) -> dict[str, str]:
    if not skip_validation:
        report = validate_release_results()
        if not report.ok:
            raise RuntimeError("Release validation failed before publication build.")
    for path in [
        PROJECT_ROOT / "src" / "generate_release_upgrade_artifacts.py",
        PROJECT_ROOT / "src" / "generate_nature_methods_artifacts.py",
        PROJECT_ROOT / "src" / "generate_publication_artifacts.py",
    ]:
        if path.exists():
            _run_script(path.name)
    if not skip_latex:
        tex_path = PAPER_ROOT / "genomecf_report.tex"
        if tex_path.exists():
            subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=PAPER_ROOT, check=False, capture_output=True, text=True)
    return {
        "paper_tex": str(PAPER_ROOT / "genomecf_report.tex"),
        "paper_pdf": str(PAPER_ROOT / "genomecf_report.pdf"),
        "publication_dir": str(PUBLICATION_ROOT),
        "figures_dir": str(FIGURES_ROOT),
    }


def main() -> None:
    build_publication(skip_tests=True, skip_latex=False, skip_validation=False)
