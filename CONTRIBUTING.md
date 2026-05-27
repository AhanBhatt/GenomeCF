# Contributing

GenomeCF is organized around registry-backed results and generated artifacts.

Recommended workflow:

1. Install with `pip install -e .[benchmark,dev]`
2. Run `python -m pytest`
3. Validate release outputs with `genomecf validate-results`
4. Rebuild artifacts you touched:
   - `genomecf build-website`
   - `genomecf build-supplement`
   - `genomecf build-paper`

Please avoid editing generated tables or figures by hand; update the source data or generation scripts instead.
