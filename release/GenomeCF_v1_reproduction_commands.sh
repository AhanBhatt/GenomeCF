#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e .[benchmark,dev]
genomecf --help
python -m pytest
python -m genomecf.cli validate-results
python -m genomecf.cli check-report --results results/release/benchmark_registry.csv
python -m genomecf.cli trace-paper --strict
python -m genomecf.cli reproduce-quickstart
python -m genomecf.cli reproduce-focal
python -m genomecf.cli reproduce-external
python -m genomecf.cli build-supplement
python -m genomecf.cli build-paper
python -m genomecf.cli build-website
