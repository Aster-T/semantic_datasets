"""Downloader registry — central access point for all data sources."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from src.base import BaseDownloader, DatasetInfo


class DownloaderRegistry:
    """Registry that aggregates multiple dataset downloaders.

    Usage:
        registry = DownloaderRegistry(data_dir="./data")
        registry.register_defaults()          # registers OpenML + Kaggle
        results = registry.search("iris")      # searches all sources
        registry.download("openml/61")         # downloads by full_id
    """

    def __init__(self, data_dir: str | Path = "./data"):
        self.data_dir = Path(data_dir)
        self._downloaders: dict[str, BaseDownloader] = {}

    def register(self, downloader: BaseDownloader) -> None:
        self._downloaders[downloader.source_name] = downloader

    def register_defaults(self) -> None:
        """Register all built-in downloaders."""
        from src.kaggle_downloader import KaggleDownloader
        from src.openml_downloader import OpenMLDownloader

        self.register(OpenMLDownloader(data_dir=self.data_dir))
        self.register(KaggleDownloader(data_dir=self.data_dir))

    def get(self, source: str) -> BaseDownloader:
        if source not in self._downloaders:
            available = ", ".join(self._downloaders)
            raise KeyError(f"Unknown source '{source}'. Available: {available}")
        return self._downloaders[source]

    @property
    def sources(self) -> list[str]:
        return list(self._downloaders)

    def search(
        self, query: str, sources: list[str] | None = None, max_results: int = 20
    ) -> list[DatasetInfo]:
        """Search across all (or specified) sources."""
        results: list[DatasetInfo] = []
        for dl in self._iter_downloaders(sources):
            try:
                results.extend(dl.search(query, max_results=max_results))
            except Exception as e:
                print(f"[warn] {dl.source_name} search failed: {e}")
        return results

    def download(self, full_id: str, dest_dir: Path | None = None) -> Path:
        """Download by full_id ('source/dataset_id').

        Examples:
            registry.download("openml/61")
            registry.download("kaggle/zillow/zecon")
        """
        source, dataset_id = self._parse_full_id(full_id)
        return self.get(source).download(dataset_id, dest_dir)

    def info(self, full_id: str) -> DatasetInfo:
        source, dataset_id = self._parse_full_id(full_id)
        return self.get(source).info(dataset_id)

    # ------------------------------------------------------------------
    def _parse_full_id(self, full_id: str) -> tuple[str, str]:
        parts = full_id.split("/", 1)
        if len(parts) != 2 or not parts[1]:
            raise ValueError(
                f"Invalid full_id '{full_id}'. Expected 'source/dataset_id'."
            )
        return parts[0], parts[1]

    def _iter_downloaders(self, sources: list[str] | None) -> Iterator[BaseDownloader]:
        if sources is None:
            yield from self._downloaders.values()
        else:
            for s in sources:
                yield self.get(s)
