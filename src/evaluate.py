"""Batch evaluate downloaded datasets for column semantic quality."""

import argparse
import asyncio
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

    def _sort_key(p: Path) -> tuple:
        """Sort numerically when the directory name is a number, otherwise lexicographically."""
        try:
            return (0, int(p.name))
        except ValueError:
            return (1, p.name)

    datasets: list[dict] = []
    for ds_dir in sorted(source_dir.iterdir(), key=_sort_key):
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


def save_result(result: EvalResult, output_dir: Path, model: str) -> None:
    """Append one evaluation result to the output JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = model.replace("/", "_")
    output_file = output_dir / f"{result.source}_{model_tag}_eval.jsonl"
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
        help="OpenAI-compatible API base URL (default: http://localhost:8000/v1 when --local)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: read from OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Use local LLM server at http://localhost:8000/v1",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent LLM requests (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of datasets to evaluate (default: all)",
    )
    return parser.parse_args()


async def evaluate_one(
    evaluator: ColumnEvaluator,
    ds: dict,
    semaphore: asyncio.Semaphore,
) -> EvalResult:
    """Evaluate a single dataset with concurrency control."""
    async with semaphore:
        return await evaluator.async_evaluate(
            columns=ds["columns"],
            description=ds["description"],
            dataset_id=ds["dataset_id"],
            source=ds["source"],
        )


async def async_main():
    args = parse_args()
    source = args.source

    evaluator = ColumnEvaluator(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        local=args.local,
    )

    output_dir = DATA_DIR / "eval"
    model_tag = args.model.replace("/", "_")
    progress_path = output_dir / f"{source}_{model_tag}_eval_progress.json"
    done = load_eval_progress(progress_path)

    datasets = discover_datasets(source)
    remaining = [ds for ds in datasets if ds["dataset_id"] not in done]
    if args.limit is not None:
        remaining = remaining[:args.limit]
    log.info(f"[{source}] Found {len(datasets)} datasets, already evaluated: {len(done)}, remaining: {len(remaining)}")

    if not remaining:
        log.info("Nothing to evaluate.")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    log.info(f"Running with concurrency={args.concurrency}")

    # Create all tasks
    tasks = {
        ds["dataset_id"]: asyncio.create_task(evaluate_one(evaluator, ds, semaphore))
        for ds in remaining
    }

    failed: list[tuple[str, str]] = []
    completed = 0

    for dataset_id, task in tasks.items():
        try:
            result = await task
            save_result(result, output_dir, args.model)
            done.add(dataset_id)
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                save_eval_progress(progress_path, done)
            if completed % 100 == 0 or completed == len(tasks):
                log.info(f"Progress: {completed}/{len(tasks)}")
        except Exception as e:
            log.warning(f"Failed to evaluate {dataset_id}: {e}")
            failed.append((dataset_id, str(e)))

    # Final progress save
    save_eval_progress(progress_path, done)

    log.info(f"Done. Evaluated: {completed}, Failed: {len(failed)}")
    if failed:
        fail_path = output_dir / f"{source}_{model_tag}_eval_failed.json"
        fail_path.write_text(
            json.dumps([{"id": fid, "error": err} for fid, err in failed],
                       indent=2, ensure_ascii=False)
        )
        log.info(f"Failed list saved to {fail_path}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
