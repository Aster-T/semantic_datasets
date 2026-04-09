"""Kaggle dataset downloader."""

from __future__ import annotations

from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from base import BaseDownloader, DatasetInfo


class KaggleDownloader(BaseDownloader):
    """Download datasets from Kaggle (kaggle.com).

    Requires a valid Kaggle API token at ~/.kaggle/kaggle.json.
    See: https://github.com/Kaggle/kaggle-api#api-credentials
    """

    def __init__(self, data_dir: str | Path = "./data"):
        super().__init__(data_dir)
        self._api = KaggleApi()
        self._api.authenticate()

    @property
    def source_name(self) -> str:
        return "kaggle"

    def search(self, query: str, max_results: int = 20) -> list[DatasetInfo]:
        results = self._api.dataset_list(search=query, page=1)
        return [self._to_info(ds) for ds in results[:max_results]]

    def list_datasets(self, max_results: int = 50, **filters) -> list[DatasetInfo]:
        sort_by = filters.pop("sort_by", "hottest")
        file_type = filters.pop("file_type", None)
        tag_ids = filters.pop("tag_ids", None)

        results = self._api.dataset_list(
            sort_by=sort_by,
            file_type=file_type,
            tag_ids=tag_ids,
            page=1,
        )
        return [self._to_info(ds) for ds in results[:max_results]]

    def info(self, dataset_id: str) -> DatasetInfo:
        """Get metadata for a Kaggle dataset.

        Args:
            dataset_id: Format 'owner/dataset-slug', e.g. 'zillow/zecon'.
        """
        owner, slug = dataset_id.split("/", 1)
        meta = self._api.dataset_view(owner, slug)
        return DatasetInfo(
            name=slug,
            source=self.source_name,
            dataset_id=dataset_id,
            description=str(getattr(meta, "description", "") or "")[:500],
            size=str(getattr(meta, "totalBytes", "")),
            tags=[str(t) for t in getattr(meta, "tags", [])],
            url=f"https://www.kaggle.com/datasets/{dataset_id}",
            extra={
                "license": str(getattr(meta, "licenseName", "")),
                "last_updated": str(getattr(meta, "lastUpdated", "")),
            },
        )

    def download(self, dataset_id: str, dest_dir: Path | None = None) -> Path:
        """Download and extract a Kaggle dataset.

        Args:
            dataset_id: Format 'owner/dataset-slug'.
        """
        dest = self._resolve_dest(dataset_id, dest_dir)

        self._api.dataset_download_files(dataset_id, path=str(dest), unzip=True)

        # Clean up leftover zip files
        for zf in dest.glob("*.zip"):
            zf.unlink()

        return dest

    # ------------------------------------------------------------------
    def _to_info(self, ds) -> DatasetInfo:
        ref = str(getattr(ds, "ref", ""))
        return DatasetInfo(
            name=str(getattr(ds, "title", ref)),
            source=self.source_name,
            dataset_id=ref,
            description=str(getattr(ds, "subtitle", ""))[:200],
            size=str(getattr(ds, "totalBytes", "")),
            tags=[str(t) for t in getattr(ds, "tags", [])],
            url=f"https://www.kaggle.com/datasets/{ref}",
            extra={
                "last_updated": str(getattr(ds, "lastUpdated", "")),
                "download_count": getattr(ds, "downloadCount", 0),
                "vote_count": getattr(ds, "voteCount", 0),
            },
        )
