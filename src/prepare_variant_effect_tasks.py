from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genomecf.variant_tasks import ensure_variant_tasks_prepared, variant_task_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess public MaveDB regulatory variant-effect tasks for GenomeCF.")
    parser.add_argument("--tasks", nargs="*", default=variant_task_ids() or None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    task_ids = args.tasks or variant_task_ids()
    prepared = ensure_variant_tasks_prepared(project_root=PROJECT_ROOT, task_ids=task_ids, force=args.force)
    for task_id, paths in prepared.items():
        print(task_id)
        for key, value in paths.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
