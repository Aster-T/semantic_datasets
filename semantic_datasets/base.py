"""Abstract base class for dataset downloaders."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ColumnSemantic:
    """Semantic description of a single column/feature."""

    column_name: str  # original column name in the data (e.g. "V1")
    semantic_name: str = ""  # human-readable name (e.g. "Recency")
    description: str = ""  # detailed description
    data_type: str = ""  # e.g. "numeric", "nominal"
    role: str = ""  # e.g. "feature", "target"


@dataclass
class DatasetInfo:
    """Metadata about a dataset."""

    name: str
    source: str  # e.g. "openml", "kaggle"
    dataset_id: str  # source-specific identifier
    description: str = ""
    size: str = ""
    tags: list[str] = field(default_factory=list)
    url: str = ""
    extra: dict = field(default_factory=dict)
    column_semantics: list[ColumnSemantic] = field(default_factory=list)

    @property
    def full_id(self) -> str:
        """Unique identifier across all sources: 'source/dataset_id'."""
        return f"{self.source}/{self.dataset_id}"

    def semantics_dict(self) -> dict[str, dict]:
        """Return column semantics as {column_name: {semantic_name, description, ...}}."""
        return {
            cs.column_name: {
                "semantic_name": cs.semantic_name,
                "description": cs.description,
                "data_type": cs.data_type,
                "role": cs.role,
            }
            for cs in self.column_semantics
        }


class BaseDownloader(abc.ABC):
    """Abstract base for all dataset downloaders.

    Subclasses must implement:
      - source_name: property returning the source identifier
      - search: search datasets by keyword
      - download: download a dataset by its source-specific id
      - list_datasets: list available datasets (with optional filters)
    """

    def __init__(self, data_dir: str | Path = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abc.abstractmethod
    def source_name(self) -> str:
        """Return the source identifier, e.g. 'openml' or 'kaggle'."""

    @abc.abstractmethod
    def search(self, query: str, max_results: int = 20) -> list[DatasetInfo]:
        """Search for datasets matching the query string."""

    @abc.abstractmethod
    def download(self, dataset_id: str, dest_dir: Path | None = None) -> Path:
        """Download a dataset and return the path to the downloaded files.

        Args:
            dataset_id: Source-specific dataset identifier.
            dest_dir: Override destination directory. Defaults to self.data_dir / source / id.

        Returns:
            Path to the directory containing downloaded files.
        """

    @abc.abstractmethod
    def list_datasets(
        self, max_results: int = 50, **filters
    ) -> list[DatasetInfo]:
        """List available datasets with optional filters."""

    @abc.abstractmethod
    def info(self, dataset_id: str) -> DatasetInfo:
        """Get detailed metadata for a specific dataset."""

    def _resolve_dest(self, dataset_id: str, dest_dir: Path | None) -> Path:
        """Resolve the destination directory for a download."""
        if dest_dir is None:
            dest_dir = self.data_dir / self.source_name / dataset_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        return dest_dir
