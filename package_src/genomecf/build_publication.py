from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .paths import FIGURES_ROOT, PAPER_ROOT, PROJECT_ROOT, PUBLICATION_ROOT
from .validation import validate_release_results


def _run_script(script_name: str) -> None:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / script_name)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def clean_latex_artifacts(*tex_stems: str) -> None:
    suffixes = [
        ".aux",
        ".bbl",
        ".bcf",
        ".blg",
        ".fdb_latexmk",
        ".fls",
        ".log",
        ".out",
        ".pdf",
        ".run.xml",
        ".toc",
    ]
    for stem in tex_stems:
        for suffix in suffixes:
            path = PAPER_ROOT / f"{stem}{suffix}"
            if path.exists():
                path.unlink()


def _run_pdflatex(tex_path: Path) -> None:
    exe = shutil.which("pdflatex")
    if exe is None:
        raise RuntimeError(
            "pdflatex not found. Install a LaTeX distribution (for example MiKTeX/TeX Live) "
            "or re-run with --skip-latex."
        )
    cmd = [exe, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", tex_path.name]
    result = subprocess.run(cmd, cwd=PAPER_ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        tail = "\n".join(combined.splitlines()[-120:])
        raise RuntimeError(f"pdflatex failed for {tex_path.name}. Output tail:\n{tail}")


def _run_bibtex(tex_path: Path) -> None:
    exe = shutil.which("bibtex")
    if exe is None:
        raise RuntimeError("bibtex not found. Install a LaTeX distribution with BibTeX support.")
    aux_path = tex_path.with_suffix(".aux")
    if not aux_path.exists():
        return
    aux_text = aux_path.read_text(encoding="utf-8", errors="ignore")
    if "\\bibdata" not in aux_text:
        return
    result = subprocess.run([exe, tex_path.stem], cwd=PAPER_ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        tail = "\n".join(combined.splitlines()[-120:])
        raise RuntimeError(f"bibtex failed for {tex_path.stem}. Output tail:\n{tail}")


def build_latex_document(tex_path: Path, *, clean: bool = True) -> Path:
    if clean:
        clean_latex_artifacts(tex_path.stem)
    _run_pdflatex(tex_path)
    _run_bibtex(tex_path)
    _run_pdflatex(tex_path)
    _run_pdflatex(tex_path)
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise RuntimeError(f"Expected PDF was not created: {pdf_path}")
    return pdf_path


def build_publication(*, skip_tests: bool = False, skip_latex: bool = False, skip_validation: bool = False) -> dict[str, str]:
    if not skip_validation:
        report = validate_release_results()
        if not report.ok:
            raise RuntimeError("Release validation failed before publication build.")
    for path in [
        PROJECT_ROOT / "src" / "generate_release_upgrade_artifacts.py",
        PROJECT_ROOT / "src" / "generate_nature_methods_artifacts.py",
        PROJECT_ROOT / "src" / "generate_publication_artifacts.py",
        PROJECT_ROOT / "src" / "generate_release_bundle.py",
    ]:
        if path.exists():
            _run_script(path.name)
    if not skip_latex:
        tex_path = PAPER_ROOT / "genomecf_report.tex"
        if tex_path.exists():
            build_latex_document(tex_path, clean=True)
    return {
        "paper_tex": str(PAPER_ROOT / "genomecf_report.tex"),
        "paper_pdf": str(PAPER_ROOT / "genomecf_report.pdf"),
        "publication_dir": str(PUBLICATION_ROOT),
        "figures_dir": str(FIGURES_ROOT),
    }


def main() -> None:
    build_publication(skip_tests=True, skip_latex=False, skip_validation=False)
