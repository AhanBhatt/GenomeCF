from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .build_publication import build_publication
from .nature_methods import build_website, check_reporting_standard, summarize_nature_methods, trace_paper
from .paths import DOCS_ROOT, PAPER_ROOT, PROJECT_ROOT, RELEASE_ROOT
from .release import build_release_registry
from .validation import validate_release_results


def _summary_output_path(output_dir: Path, suite: str) -> Path:
    return output_dir / f"benchmark_summary_{suite}.csv"


def _load_summary() -> pd.DataFrame:
    build_release_registry()
    return pd.read_csv(RELEASE_ROOT / "benchmark_summary.csv")


def cmd_summarize(args: argparse.Namespace) -> None:
    summary = _load_summary()
    suite = args.suite.lower()
    if suite == "all" or suite == "nature_methods":
        subset = summary
    else:
        subset = summary[summary["tier"] == suite]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _summary_output_path(output_dir, suite)
    subset.to_csv(path, index=False)
    print(f"Wrote summary to {path}")


def cmd_validate_results(args: argparse.Namespace) -> None:
    report = validate_release_results()
    if not report.ok:
        raise SystemExit("\n".join(report.errors))
    print("Release validation passed.")


def cmd_build_paper(args: argparse.Namespace) -> None:
    payload = build_publication(skip_tests=args.skip_tests, skip_latex=args.skip_latex, skip_validation=args.skip_validation)
    print(json.dumps(payload, indent=2))


def cmd_build_appendix(args: argparse.Namespace) -> None:
    payload = build_publication(skip_tests=True, skip_latex=True, skip_validation=False)
    print(f"Appendix artifacts ready in {payload['publication_dir']}")


def cmd_build_supplement(args: argparse.Namespace) -> None:
    payload = build_publication(skip_tests=True, skip_latex=True, skip_validation=False)
    print(f"Supplement artifacts ready in {payload['publication_dir']}")
    tex_path = PAPER_ROOT / "genomecf_supplement.tex"
    if tex_path.exists():
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=PAPER_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        pdf_path = PAPER_ROOT / "genomecf_supplement.pdf"
        if result.returncode == 0 and pdf_path.exists():
            print(f"Built supplement PDF at {pdf_path}")
        else:
            print("Supplement LaTeX build did not succeed (pdflatex missing or compilation failed).")


def cmd_build_website(args: argparse.Namespace) -> None:
    path = build_website(output_dir=Path(args.output_dir) if args.output_dir else DOCS_ROOT / "site", regenerate=args.regenerate)
    print(f"Built website at {path}")


def cmd_check_report(args: argparse.Namespace) -> None:
    payload = check_reporting_standard(results_path=Path(args.results), output_path=Path(args.output) if args.output else RELEASE_ROOT / "reporting_check_report.json")
    print(json.dumps(payload, indent=2))


def cmd_trace_paper(args: argparse.Namespace) -> None:
    payload = trace_paper(output_path=Path(args.output) if args.output else RELEASE_ROOT / "paper_claim_traceability.csv", strict=args.strict)
    print(json.dumps(payload, indent=2))


def cmd_reproduce_quickstart(args: argparse.Namespace) -> None:
    build_release_registry()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "registry": str(RELEASE_ROOT / "benchmark_registry.csv"),
        "summary": str(RELEASE_ROOT / "benchmark_summary.csv"),
        "paper": str(PAPER_ROOT / "genomecf_report.pdf"),
        "website": str(DOCS_ROOT / "site" / "index.html"),
    }
    (output_dir / "quickstart_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Quickstart reproduction completed at {output_dir / 'quickstart_report.json'}")


def cmd_reproduce_focal(args: argparse.Namespace) -> None:
    cmd_reproduce_quickstart(args)


def cmd_reproduce_external(args: argparse.Namespace) -> None:
    cmd_reproduce_quickstart(args)


def cmd_smoke_test(args: argparse.Namespace) -> None:
    build_release_registry()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": True,
        "benchmark_summary": str(RELEASE_ROOT / "benchmark_summary.csv"),
        "website": str(DOCS_ROOT / "site" / "index.html"),
    }
    (output_dir / "smoke_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Smoke test completed")


def _filter_and_print(frame: pd.DataFrame, output_path: Path | None = None) -> None:
    if output_path is not None:
        frame.to_csv(output_path, index=False)
        print(f"Wrote {output_path}")
    else:
        print(frame.to_string(index=False))


def cmd_evaluate(args: argparse.Namespace) -> None:
    summary = _load_summary()
    subset = summary[
        (summary["task_id"] == args.task)
        & (summary["model_id"] == args.model)
        & (summary["split_id"] == args.split)
    ].copy()
    if args.mode:
        subset = subset[subset["mode"] == args.mode]
    _filter_and_print(subset, Path(args.output) if args.output else None)


def cmd_external(args: argparse.Namespace) -> None:
    frame = pd.read_csv(RELEASE_ROOT / "external_validation_summary.csv")
    subset = frame[(frame["task_id"] == args.task) & (frame["model_id"] == args.model)].copy()
    _filter_and_print(subset, Path(args.output) if args.output else None)


def cmd_variant(args: argparse.Namespace) -> None:
    variant_root = RELEASE_ROOT / "variant_effect"
    files = list(variant_root.glob(f"{args.task}__{args.model}__*.csv"))
    if not files:
        raise SystemExit(f"No variant-effect summaries found for {args.task} / {args.model}.")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    _filter_and_print(frame, Path(args.output) if args.output else None)


def cmd_synth(args: argparse.Namespace) -> None:
    frame = pd.read_csv(RELEASE_ROOT / "synthetic_extended_summary.csv")
    subset = frame[(frame["task_id"] == args.task) & (frame["model_id"] == args.model)].copy()
    _filter_and_print(subset, Path(args.output) if args.output else None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GenomeCF: counterfactual validation benchmark for DNA sequence models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize", help="Write a filtered summary table from the release registry.")
    summarize.add_argument("--suite", default="all", help="One of core, external, synthetic, screening, all, nature_methods.")
    summarize.add_argument("--output-dir", default=str(RELEASE_ROOT / "appendix"))
    summarize.set_defaults(func=cmd_summarize)

    validate = subparsers.add_parser("validate-results", help="Validate that the release registry and required artifacts exist.")
    validate.set_defaults(func=cmd_validate_results)

    build_paper = subparsers.add_parser("build-paper", help="Refresh generated artifacts and optionally rebuild the LaTeX paper.")
    build_paper.add_argument("--skip-tests", action="store_true")
    build_paper.add_argument("--skip-latex", action="store_true")
    build_paper.add_argument("--skip-validation", action="store_true")
    build_paper.set_defaults(func=cmd_build_paper)

    build_appendix = subparsers.add_parser("build-appendix", help="Refresh appendix/publication tables without forcing a LaTeX rebuild.")
    build_appendix.set_defaults(func=cmd_build_appendix)

    build_supplement = subparsers.add_parser("build-supplement", help="Refresh supplement artifacts.")
    build_supplement.set_defaults(func=cmd_build_supplement)

    build_site = subparsers.add_parser("build-website", help="Generate the local static GenomeCF website.")
    build_site.add_argument("--output-dir", default=str(DOCS_ROOT / "site"))
    build_site.add_argument("--regenerate", action="store_true")
    build_site.set_defaults(func=cmd_build_website)

    check_report = subparsers.add_parser("check-report", help="Validate a results CSV against the GenomeCF reporting standard.")
    check_report.add_argument("--results", required=True)
    check_report.add_argument("--output", default=str(RELEASE_ROOT / "reporting_check_report.json"))
    check_report.set_defaults(func=cmd_check_report)

    trace = subparsers.add_parser("trace-paper", help="Generate a paper-claim traceability report from publication artifacts.")
    trace.add_argument("--output", default=str(RELEASE_ROOT / "paper_claim_traceability.csv"))
    trace.add_argument("--strict", action="store_true")
    trace.set_defaults(func=cmd_trace_paper)

    quickstart = subparsers.add_parser("reproduce-quickstart", help="Verify the installed release against the shipped benchmark artifacts.")
    quickstart.add_argument("--output-dir", default=str(RELEASE_ROOT / "quickstart"))
    quickstart.set_defaults(func=cmd_reproduce_quickstart)

    focal = subparsers.add_parser("reproduce-focal", help="Write a focal benchmark reproduction report.")
    focal.add_argument("--output-dir", default=str(RELEASE_ROOT / "quickstart"))
    focal.set_defaults(func=cmd_reproduce_focal)

    external = subparsers.add_parser("reproduce-external", help="Write an external validation reproduction report.")
    external.add_argument("--output-dir", default=str(RELEASE_ROOT / "quickstart"))
    external.set_defaults(func=cmd_reproduce_external)

    smoke = subparsers.add_parser("smoke-test", help="Run a lightweight installation smoke check and write a JSON report.")
    smoke.add_argument("--output-dir", default=str(RELEASE_ROOT / "smoke"))
    smoke.set_defaults(func=cmd_smoke_test)

    evaluate = subparsers.add_parser("evaluate", help="Query registry-backed benchmark results for a task/model/split combination.")
    evaluate.add_argument("--task", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--split", required=True)
    evaluate.add_argument("--mode")
    evaluate.add_argument("--output")
    evaluate.set_defaults(func=cmd_evaluate)

    external_cmd = subparsers.add_parser("external", help="Query external-validation summary rows.")
    external_cmd.add_argument("--task", required=True)
    external_cmd.add_argument("--model", required=True)
    external_cmd.add_argument("--output")
    external_cmd.set_defaults(func=cmd_external)

    variant_cmd = subparsers.add_parser("variant", help="Query variant-effect summary rows.")
    variant_cmd.add_argument("--task", required=True)
    variant_cmd.add_argument("--model", required=True)
    variant_cmd.add_argument("--output")
    variant_cmd.set_defaults(func=cmd_variant)

    synth_cmd = subparsers.add_parser("synth", help="Query GenomeCF-Synth summary rows.")
    synth_cmd.add_argument("--task", required=True)
    synth_cmd.add_argument("--model", required=True)
    synth_cmd.add_argument("--output")
    synth_cmd.set_defaults(func=cmd_synth)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
