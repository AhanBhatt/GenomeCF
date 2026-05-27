# Quickstart

Run the small end-to-end path:

```bash
genomecf reproduce-quickstart
```

Manual equivalent:

```bash
genomecf smoke-test
genomecf summarize --suite nature_methods
genomecf validate-results
genomecf build-website
```

Outputs:

- `results/release/quickstart/quickstart_report.json`
- `docs/site/index.html` or the custom site output you pass

