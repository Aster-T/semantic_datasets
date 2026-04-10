"""Remap placeholder column names to semantic names based on evaluation results."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DATA_DIR = Path("./data")


def load_eval_results(source: str) -> list[dict]:
    """Load evaluation results that have a non-empty column_mapping."""
    eval_file = DATA_DIR / "eval" / f"{source}_eval.jsonl"
    if not eval_file.exists():
        log.error(f"Eval file not found: {eval_file}")
        return []

    results = []
    for line in eval_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("column_mapping"):
            results.append(entry)
    return results


def remap_dataset(source: str, dataset_id: str, column_mapping: dict[str, str]) -> None:
    """Rename columns in parquet files and update metadata for one dataset."""
    ds_dir = DATA_DIR / source / dataset_id
    if not ds_dir.exists():
        log.warning(f"Dataset directory not found: {ds_dir}")
        return

    for parquet_file in ds_dir.glob("*.parquet"):
        df = pd.read_parquet(parquet_file)
        # Only rename columns that exist in the file
        rename_map = {k: v for k, v in column_mapping.items() if k in df.columns}
        if not rename_map:
            continue

        df = df.rename(columns=rename_map)
        df.to_parquet(parquet_file, index=False)
        log.info(f"  Remapped {len(rename_map)} columns in {parquet_file.name}")

    # Update metadata.json if it exists
    meta_path = ds_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["column_mapping"] = column_mapping
        if meta.get("default_target") in column_mapping:
            meta["default_target"] = column_mapping[meta["default_target"]]
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remap placeholder columns to semantic names")
    parser.add_argument(
        "--source",
        choices=["openml", "kaggle"],
        default="openml",
        help="Data source to remap (default: openml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source

    entries = load_eval_results(source)
    log.info(f"[{source}] Found {len(entries)} datasets with column mappings")

    for i, entry in enumerate(entries, 1):
        dataset_id = entry["dataset_id"]
        mapping = entry["column_mapping"]
        log.info(f"[{i}/{len(entries)}] Remapping {dataset_id}: {mapping}")
        try:
            remap_dataset(source, dataset_id, mapping)
        except Exception as e:
            log.warning(f"Failed to remap {dataset_id}: {e}")


if __name__ == "__main__":
    main()
