import json
import logging
from pathlib import Path

import openml
import pandas as pd

from openml_downloader import OpenMLDownloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DATA_DIR = Path("./data")
PROGRESS_FILE = DATA_DIR / "openml_progress.json"


def load_progress() -> set[int]:
    """Load already-downloaded dataset IDs from progress file."""
    if PROGRESS_FILE.exists():
        return set(json.loads(PROGRESS_FILE.read_text()))
    return set()


def save_progress(done: set[int]) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(sorted(done)))


def get_all_dataset_ids() -> list[int]:
    """Fetch all dataset IDs from OpenML, sorted by ID."""
    log.info("Fetching dataset list from OpenML...")
    df = openml.datasets.list_datasets(output_format="dataframe")
    assert isinstance(df, pd.DataFrame)
    ids = sorted(df.index.tolist())
    log.info(f"Found {len(ids)} datasets on OpenML")
    return ids


def main():
    downloader = OpenMLDownloader(data_dir=DATA_DIR)
    all_ids = get_all_dataset_ids()

    done = load_progress()
    remaining = [did for did in all_ids if did not in done]
    log.info(f"Already downloaded: {len(done)}, remaining: {len(remaining)}")

    failed: list[tuple[int, str]] = []

    for i, dataset_id in enumerate(remaining, 1):
        log.info(f"[{i}/{len(remaining)}] Downloading dataset {dataset_id} ...")
        try:
            downloader.download(str(dataset_id))
            done.add(dataset_id)
            # Save progress after each successful download
            save_progress(done)
        except Exception as e:
            log.warning(f"Failed to download dataset {dataset_id}: {e}")
            failed.append((dataset_id, str(e)))
            continue

    log.info(f"Done. Downloaded: {len(done)}, Failed: {len(failed)}")
    if failed:
        fail_path = DATA_DIR / "openml_failed.json"
        fail_path.write_text(
            json.dumps([{"id": fid, "error": err} for fid, err in failed],
                       indent=2, ensure_ascii=False)
        )
        log.info(f"Failed list saved to {fail_path}")


if __name__ == "__main__":
    main()
