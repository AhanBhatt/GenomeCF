This folder keeps the lightweight public task bundles that are safe to ship in the GitHub repo.

Large local-only benchmark inputs are intentionally excluded from version control and can live in the sibling runtime-assets folder:

- `../local_runtime_assets/data/`

GenomeCF will automatically look there when the large raw task folders are not present in this repo checkout.
