from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT / "package_src" / "genomecf"


if "genomecf" not in sys.modules and PACKAGE_DIR.exists() and (PACKAGE_DIR / "__init__.py").exists():
    spec = importlib.util.spec_from_file_location(
        "genomecf",
        PACKAGE_DIR / "__init__.py",
        submodule_search_locations=[str(PACKAGE_DIR)],
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["genomecf"] = module
        spec.loader.exec_module(module)
