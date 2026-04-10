import argparse
import json
import logging
from pathlib import Path

import openml
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from openml_downloader import OpenMLDownloader
from kaggle_downloader import KaggleDownloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DATA_DIR = Path("./data")


def progress_file(source: str) -> Path:
    return DATA_DIR / f"{source}_progress.json"


def load_progress(source: str) -> set[str]:
    """Load already-downloaded dataset IDs from progress file."""
    pf = progress_file(source)
    if pf.exists():
        return set(json.loads(pf.read_text()))
    return set()


def save_progress(source: str, done: set[str]) -> None:
    pf = progress_file(source)
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(json.dumps(sorted(done)))


def get_all_openml_ids() -> list[str]:
    """Fetch all dataset IDs from OpenML, sorted by ID."""
    log.info("Fetching dataset list from OpenML...")
    df = openml.datasets.list_datasets(output_format="dataframe")
    assert isinstance(df, pd.DataFrame)
    ids = sorted(df.index.tolist())
    log.info(f"Found {len(ids)} datasets on OpenML")
    return [str(i) for i in ids]


# 单个数据集大小上限（字节），超过则跳过。默认 500 MB
MAX_DATASET_BYTES = 500 * 1024 * 1024


def get_all_kaggle_ids() -> list[str]:
    """Fetch all CSV dataset IDs (owner/slug) from Kaggle, auto-paginating until exhausted."""
    log.info("Fetching dataset list from Kaggle...")
    api = KaggleApi()
    api.authenticate()
    all_refs: list[str] = []
    page = 1
    while True:
        results = api.dataset_list(page=page, file_type="csv")
        if not results:
            break
        for ds in results:
            ref = str(getattr(ds, "ref", ""))
            size = getattr(ds, "totalBytes", 0) or 0
            if not ref:
                continue
            if size > MAX_DATASET_BYTES:
                log.info(f"Skipping {ref} (size {size / 1024 / 1024:.0f} MB > limit)")
                continue
            all_refs.append(ref)
        page += 1
    log.info(f"Found {len(all_refs)} datasets on Kaggle (after size filter)")
    return all_refs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download datasets from OpenML or Kaggle")
    parser.add_argument(
        "--source",
        choices=["openml", "kaggle"],
        default="openml",
        help="Data source to download from (default: openml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source

    if source == "openml":
        downloader = OpenMLDownloader(data_dir=DATA_DIR)
        all_ids = get_all_openml_ids()
    elif source == "kaggle":
        downloader = KaggleDownloader(data_dir=DATA_DIR)
        all_ids = get_all_kaggle_ids()
    else:
        raise ValueError(f"Unknown source: {source!r}. Use 'openml' or 'kaggle'.")

    done = load_progress(source)
    remaining = [did for did in all_ids if did not in done]
    log.info(f"[{source}] Already downloaded: {len(done)}, remaining: {len(remaining)}")

    failed: list[tuple[str, str]] = []

    for i, dataset_id in enumerate(remaining, 1):
        log.info(f"[{i}/{len(remaining)}] Downloading {source} dataset {dataset_id} ...")
        try:
            downloader.download(dataset_id)
            done.add(dataset_id)
            save_progress(source, done)
        except Exception as e:
            log.warning(f"Failed to download dataset {dataset_id}: {e}")
            failed.append((dataset_id, str(e)))
            continue

    log.info(f"Done. Downloaded: {len(done)}, Failed: {len(failed)}")
    if failed:
        fail_path = DATA_DIR / f"{source}_failed.json"
        fail_path.write_text(
            json.dumps([{"id": fid, "error": err} for fid, err in failed],
                       indent=2, ensure_ascii=False)
        )
        log.info(f"Failed list saved to {fail_path}")


if __name__ == "__main__":
    main()
