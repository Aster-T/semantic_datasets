"""Backfill task_type into existing OpenML metadata.json files."""

import argparse
import json
import logging
from pathlib import Path

import openml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DATA_DIR = Path("./data")


def infer_task_type(dataset_id: int) -> str:
    """Infer task type from OpenML tasks associated with this dataset."""
    try:
        tasks = openml.tasks.list_tasks(
            data_id=dataset_id, output_format="dataframe"
        )
        if tasks is not None and "task_type" in tasks.columns:
            task_types = tasks["task_type"].unique().tolist()
            if "Supervised Regression" in task_types:
                return "regression"
            if "Supervised Classification" in task_types:
                return "classification"
    except Exception:
        pass
    return ""


def main():
    parser = argparse.ArgumentParser(description="Backfill task_type into metadata.json")
    parser.add_argument("--source", default="openml", choices=["openml"])
    args = parser.parse_args()

    source_dir = DATA_DIR / args.source
    if not source_dir.exists():
        log.error(f"Source directory not found: {source_dir}")
        return

    ds_dirs = sorted(
        [d for d in source_dir.iterdir() if d.is_dir()],
        key=lambda p: (0, int(p.name)) if p.name.isdigit() else (1, p.name),
    )

    updated = 0
    skipped = 0
    failed = 0

    for ds_dir in ds_dirs:
        meta_path = ds_dir / "metadata.json"
        if not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        # Skip if task_type already exists and is non-empty
        if meta.get("task_type"):
            skipped += 1
            continue

        dataset_id = meta.get("dataset_id", ds_dir.name)
        try:
            task_type = infer_task_type(int(dataset_id))
        except Exception as e:
            log.warning(f"Failed to query task_type for {dataset_id}: {e}")
            failed += 1
            continue

        meta["task_type"] = task_type
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        updated += 1
        if updated % 100 == 0:
            log.info(f"Progress: updated {updated}")

    log.info(f"Done. Updated: {updated}, Skipped (already has task_type): {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
