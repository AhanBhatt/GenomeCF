$ErrorActionPreference = "Stop"

python -m genomecf.cli smoke-test
python -m genomecf.cli summarize --suite core
python -m genomecf.cli build-appendix
python -m genomecf.cli validate-results
