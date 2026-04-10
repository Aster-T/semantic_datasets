"""Batch evaluate downloaded datasets for column semantic quality."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from column_evaluator import ColumnEvaluator, EvalResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DATA_DIR = Path("./data")


def load_eval_progress(path: Path) -> set[str]:
    """Load already-evaluated dataset keys."""
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def save_eval_progress(path: Path, done: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(done)))


def discover_datasets(source: str) -> list[dict]:
    """Scan data/{source}/ and collect (dataset_id, columns, description) for each dataset."""
    source_dir = DATA_DIR / source
    if not source_dir.exists():
        log.warning(f"Source directory not found: {source_dir}")
        return []

    datasets: list[dict] = []
    for ds_dir in sorted(source_dir.iterdir()):
        if not ds_dir.is_dir():
            continue

        # For kaggle, dataset_id is owner/slug (two-level nesting)
        if source == "kaggle":
            for sub_dir in sorted(ds_dir.iterdir()):
                if not sub_dir.is_dir():
                    continue
                entry = _read_dataset_dir(sub_dir, source, f"{ds_dir.name}/{sub_dir.name}")
                if entry:
                    datasets.append(entry)
        else:
            entry = _read_dataset_dir(ds_dir, source, ds_dir.name)
            if entry:
                datasets.append(entry)

    return datasets


def _read_dataset_dir(ds_dir: Path, source: str, dataset_id: str) -> dict | None:
    """Read columns from parquet and description from metadata.json."""
    # Find a parquet file
    parquet_files = list(ds_dir.glob("*.parquet"))
    if not parquet_files:
        return None

    try:
        columns = pd.read_parquet(parquet_files[0], columns=[]).columns.tolist()
    except Exception as e:
        log.warning(f"Cannot read parquet {parquet_files[0]}: {e}")
        return None

    # Read description from metadata.json
    description = ""
    meta_path = ds_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            description = meta.get("description", "")
        except Exception:
            pass

    return {
        "dataset_id": dataset_id,
        "source": source,
        "columns": columns,
        "description": description,
    }


def save_result(result: EvalResult, output_dir: Path) -> None:
    """Append one evaluation result to the output JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{result.source}_eval.jsonl"
    entry = {
        "dataset_id": result.dataset_id,
        "source": result.source,
        "quality": result.quality,
        "columns": result.columns,
        "column_mapping": result.column_mapping,
    }
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate column semantic quality of downloaded datasets")
    parser.add_argument(
        "--source",
        choices=["openml", "kaggle"],
        default="openml",
        help="Data source to evaluate (default: openml)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: read from OPENAI_API_KEY env var)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source

    evaluator = ColumnEvaluator(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    output_dir = DATA_DIR / "eval"
    progress_path = output_dir / f"{source}_eval_progress.json"
    done = load_eval_progress(progress_path)

    datasets = discover_datasets(source)
    remaining = [ds for ds in datasets if ds["dataset_id"] not in done]
    log.info(f"[{source}] Found {len(datasets)} datasets, already evaluated: {len(done)}, remaining: {len(remaining)}")

    failed: list[tuple[str, str]] = []

    for i, ds in enumerate(remaining, 1):
        dataset_id = ds["dataset_id"]
        log.info(f"[{i}/{len(remaining)}] Evaluating {dataset_id} ...")
        try:
            result = evaluator.evaluate(
                columns=ds["columns"],
                description=ds["description"],
                dataset_id=dataset_id,
                source=source,
            )
            save_result(result, output_dir)
            done.add(dataset_id)
            save_eval_progress(progress_path, done)
            log.info(f"  -> quality={result.quality}, mapping_count={len(result.column_mapping)}")
        except Exception as e:
            log.warning(f"Failed to evaluate {dataset_id}: {e}")
            failed.append((dataset_id, str(e)))
            continue

    log.info(f"Done. Evaluated: {len(done)}, Failed: {len(failed)}")
    if failed:
        fail_path = output_dir / f"{source}_eval_failed.json"
        fail_path.write_text(
            json.dumps([{"id": fid, "error": err} for fid, err in failed],
                       indent=2, ensure_ascii=False)
        )
        log.info(f"Failed list saved to {fail_path}")


if __name__ == "__main__":
    main()
