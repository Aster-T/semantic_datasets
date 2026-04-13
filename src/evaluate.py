"""Batch evaluate downloaded datasets for column semantic quality."""

import argparse
import asyncio
import json
import logging
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from column_evaluator import ColumnEvaluator, EvalResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DATA_DIR = Path("./data")


def wait_for_local_server(
    base_url: str,
    model: str,
    timeout_seconds: int = 300,
    poll_interval_seconds: int = 2,
) -> None:
    """Wait until the local OpenAI-compatible server is ready."""
    models_url = urllib.parse.urljoin(base_url.rstrip("/") + "/", "models")
    deadline = time.time() + timeout_seconds
    last_error = ""

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
            models = payload.get("data", []) if isinstance(payload, dict) else []
            model_ids = [
                str(item.get("id", ""))
                for item in models
                if isinstance(item, dict) and item.get("id")
            ]
            if model_ids:
                if model not in model_ids:
                    log.warning(
                        "Local server is ready at %s, but model '%s' is not listed. Available models: %s",
                        base_url,
                        model,
                        ", ".join(model_ids),
                    )
                else:
                    log.info("Local server is ready at %s with model %s", base_url, model)
                return
            log.info("Local server at %s is up but has not exposed any models yet.", base_url)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(poll_interval_seconds)

    raise RuntimeError(
        f"Local server at {base_url} was not ready within {timeout_seconds}s. "
        f"Last error: {last_error or 'unknown'}"
    )


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

    # Read metadata from metadata.json
    description = ""
    task_type = ""
    default_target = ""
    meta_path = ds_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            description = meta.get("description", "")
            task_type = meta.get("task_type", "")
            default_target = meta.get("default_target", "")
        except Exception:
            pass

    return {
        "dataset_id": dataset_id,
        "source": source,
        "columns": columns,
        "description": description,
        "task_type": task_type,
        "default_target": default_target,
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
        "description": result.description,
        "columns": result.columns,
        "columns_mapping": result.columns_mapping,
        "Task_type": result.task_type,
        "target_column": result.target_column,
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
        help="OpenAI-compatible API base URL (default: http://127.0.0.1:8000/v1 when --local)",
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
        help="Use local LLM server at http://127.0.0.1:8000/v1",
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
    max_retries: int = 3,
) -> EvalResult:
    """Evaluate a single dataset with concurrency control and retry."""
    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                return await evaluator.async_evaluate(
                    columns=ds["columns"],
                    description=ds["description"],
                    dataset_id=ds["dataset_id"],
                    source=ds["source"],
                    task_type=ds.get("task_type", ""),
                    default_target=ds.get("default_target", ""),
                )
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt + random.uniform(0, 1)
                log.info(f"Retry {attempt + 1}/{max_retries} for {ds['dataset_id']} in {wait:.2f}s: {e}")
                await asyncio.sleep(wait)
            else:
                raise


async def async_main():
    args = parse_args()
    source = args.source

    if args.local:
        base_url = args.base_url or ColumnEvaluator.LOCAL_BASE_URL
        wait_for_local_server(base_url=base_url, model=args.model)
        if args.concurrency > 8:
            log.warning(
                "Local concurrency=%s is aggressive for Qwen/Qwen3.5-35B-A3B; reduce it if connection errors persist.",
                args.concurrency,
            )

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

    failed: list[tuple[str, str]] = []
    completed = 0
    total = len(remaining)

    async def run_one(ds: dict) -> None:
        nonlocal completed
        dataset_id = ds["dataset_id"]
        try:
            result = await evaluate_one(evaluator, ds, semaphore)
            save_result(result, output_dir, args.model)
            done.add(dataset_id)
            completed += 1
            if completed % 50 == 0 or completed == total:
                save_eval_progress(progress_path, done)
            if completed % 100 == 0 or completed == total:
                log.info(f"Progress: {completed}/{total}")
        except Exception as e:
            log.warning(f"Failed to evaluate {dataset_id}: {e}")
            failed.append((dataset_id, str(e)))

    await asyncio.gather(*[run_one(ds) for ds in remaining])

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
