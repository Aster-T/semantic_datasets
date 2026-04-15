"""OpenML dataset downloader."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import openml
import pandas as pd
from openml.datasets.functions import _get_dataset_arff

from base import BaseDownloader, DatasetInfo

log = logging.getLogger(__name__)

OPENML_STATUSES = ("active", "in_preparation", "deactivated")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _coerce_dataset_id(value: Any) -> str:
    if _is_missing(value):
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _normalize_str(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value)


def _normalize_list(value: Any) -> list[str]:
    if _is_missing(value):
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return [part for part in parts if part]
    if isinstance(value, (list, tuple, set)):
        return [_normalize_str(item) for item in value if not _is_missing(item)]
    return [_normalize_str(value)]


def _json_ready(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
            if not _is_missing(item)
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value if not _is_missing(item)]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _json_ready(value.item())
        except (AttributeError, TypeError, ValueError):
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class OpenMLDownloader(BaseDownloader):
    """Download datasets from OpenML (openml.org)."""

    def __init__(self, data_dir: str | Path = "./data"):
        super().__init__(data_dir)
        self._dataset_listing_cache: pd.DataFrame | None = None

    @property
    def source_name(self) -> str:
        return "openml"

    def search(self, query: str, max_results: int = 20) -> list[DatasetInfo]:
        datasets = self._list_dataset_frame(status="all")
        name_col = datasets["name"] if "name" in datasets.columns else pd.Series("", index=datasets.index)
        desc_col = (
            datasets["description"]
            if "description" in datasets.columns
            else pd.Series("", index=datasets.index)
        )
        mask = name_col.str.contains(query, case=False, na=False) | desc_col.str.contains(
            query, case=False, na=False
        )
        matched = datasets[mask].head(max_results)
        return [self._row_to_info(row) for _, row in matched.iterrows()]

    def list_datasets(self, max_results: int = 50, **filters) -> list[DatasetInfo]:
        datasets = self._list_dataset_frame(**filters)
        return [
            self._row_to_info(row) for _, row in datasets.head(max_results).iterrows()
        ]

    def info(self, dataset_id: str) -> DatasetInfo:
        ds = openml.datasets.get_dataset(
            int(dataset_id), download_data=False, download_qualities=False
        )
        listing_row = self._get_listing_row(dataset_id)
        extra = {
            "format": ds.format,
            "version": ds.version,
            "default_target": ds.default_target_attribute,
            "visibility": getattr(ds, "visibility", ""),
            "row_id_attribute": getattr(ds, "row_id_attribute", ""),
            "ignore_attribute": _normalize_list(getattr(ds, "ignore_attribute", [])),
        }
        if listing_row is not None:
            extra.update(
                {
                    "status": _normalize_str(listing_row.get("status", "")),
                    "num_instances": self._row_int(listing_row, "NumberOfInstances"),
                    "num_features": self._row_int(listing_row, "NumberOfFeatures"),
                }
            )
        return DatasetInfo(
            name=ds.name,
            source=self.source_name,
            dataset_id=str(ds.dataset_id),
            description=ds.description or "",
            tags=_normalize_list(ds.tag),
            url=self._dataset_url(dataset_id, ds),
            extra=extra,
        )

    @staticmethod
    def _infer_task_type(dataset_id: int) -> str:
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

    def download(self, dataset_id: str, dest_dir: Path | None = None) -> Path:
        dest = self._resolve_openml_dest(dataset_id, dest_dir)

        ds = openml.datasets.get_dataset(
            int(dataset_id), download_data=False, download_qualities=False
        )
        self._download_dataset_assets(ds)
        download_info = self._save_dataset_arff(ds, dest / "table.arff")

        task_type = self._infer_task_type(int(dataset_id))
        listing_row = self._get_listing_row(dataset_id)

        # Save metadata
        metadata = {
            "name": ds.name,
            "source": "openml",
            "dataset_id": str(ds.dataset_id),
            "description": ds.description or "",
            "url": self._dataset_url(dataset_id, ds),
            "default_target": _normalize_str(ds.default_target_attribute),
            "task_type": task_type,
            "tags": _normalize_list(ds.tag),
            "status": _normalize_str(
                listing_row.get("status", "") if listing_row is not None else ""
            ),
            "version": _json_ready(getattr(ds, "version", None)),
            "format": _normalize_str(getattr(ds, "format", "")),
            "visibility": _normalize_str(getattr(ds, "visibility", "")),
            "creator": _normalize_str(getattr(ds, "creator", "")),
            "contributor": _normalize_list(getattr(ds, "contributor", [])),
            "collection_date": _normalize_str(getattr(ds, "collection_date", "")),
            "upload_date": _normalize_str(getattr(ds, "upload_date", "")),
            "language": _normalize_str(getattr(ds, "language", "")),
            "licence": _normalize_str(getattr(ds, "licence", "")),
            "version_label": _normalize_str(getattr(ds, "version_label", "")),
            "citation": _normalize_str(getattr(ds, "citation", "")),
            "row_id_attribute": _json_ready(getattr(ds, "row_id_attribute", None)),
            "ignore_attribute": _normalize_list(getattr(ds, "ignore_attribute", [])),
            "original_data_url": _normalize_str(getattr(ds, "original_data_url", "")),
            "paper_url": _normalize_str(getattr(ds, "paper_url", "")),
            "update_comment": _normalize_str(getattr(ds, "update_comment", "")),
            "md5_checksum": _normalize_str(getattr(ds, "md5_checksum", "")),
            "local_table_file": download_info["local_file"],
            "storage_format": download_info["storage_format"],
            "source_preference": "arff",
            "download_strategy": download_info["strategy"],
            "openml_cached_arff": download_info["source_arff"],
            "num_instances": self._row_int(listing_row, "NumberOfInstances"),
            "num_features": self._row_int(listing_row, "NumberOfFeatures"),
            "num_missing_values": self._row_int(listing_row, "NumberOfMissingValues"),
        }
        meta_path = dest / "metadata.json"
        meta_path.write_text(
            json.dumps(_json_ready(metadata), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return dest

    def all_dataset_ids(self) -> list[str]:
        datasets = self._list_dataset_frame(status="all")
        return [_coerce_dataset_id(dataset_id) for dataset_id in datasets.index.tolist()]

    # ------------------------------------------------------------------

    def _row_to_info(self, row: pd.Series) -> DatasetInfo:
        dataset_id = _coerce_dataset_id(row.get("did", row.name))
        return DatasetInfo(
            name=str(row.get("name", "")),
            source=self.source_name,
            dataset_id=dataset_id,
            description=str(row.get("description", "")),
            tags=_normalize_list(row.get("tag", [])),
            url=self._dataset_url(dataset_id),
            extra={
                "status": _normalize_str(row.get("status", "")),
                "format": _normalize_str(row.get("format", "")),
                "version": _normalize_str(row.get("version", "")),
                "num_instances": self._row_int(row, "NumberOfInstances"),
                "num_features": self._row_int(row, "NumberOfFeatures"),
            },
        )

    def _list_dataset_frame(self, **filters) -> pd.DataFrame:
        status = filters.pop("status", "all")
        if status == "all":
            datasets = self._get_all_dataset_listing()
        else:
            datasets = openml.datasets.list_datasets(
                status=status,
                output_format="dataframe",
            )
            assert isinstance(datasets, pd.DataFrame)
            datasets = self._ensure_dataset_index(datasets)

        for key, value in filters.items():
            if value is None:
                continue
            if key in datasets.columns:
                datasets = datasets[datasets[key] == value]

        return datasets.sort_index()

    def _get_all_dataset_listing(self) -> pd.DataFrame:
        if self._dataset_listing_cache is not None:
            return self._dataset_listing_cache.copy()

        try:
            datasets = openml.datasets.list_datasets(
                status="all",
                output_format="dataframe",
            )
            assert isinstance(datasets, pd.DataFrame)
            datasets = self._ensure_dataset_index(datasets)
        except Exception as exc:
            log.info(
                "OpenML client failed on status='all' (%s); falling back to per-status listing.",
                exc,
            )
            frames: list[pd.DataFrame] = []
            for status in OPENML_STATUSES:
                frame = openml.datasets.list_datasets(
                    status=status,
                    output_format="dataframe",
                )
                assert isinstance(frame, pd.DataFrame)
                frame = self._ensure_dataset_index(frame)
                if "status" not in frame.columns:
                    frame = frame.copy()
                    frame["status"] = status
                frames.append(frame)

            datasets = pd.concat(frames, axis=0) if frames else pd.DataFrame()

        datasets = datasets[~datasets.index.duplicated(keep="first")].sort_index()
        self._dataset_listing_cache = datasets.copy()
        return datasets

    @staticmethod
    def _ensure_dataset_index(datasets: pd.DataFrame) -> pd.DataFrame:
        if "did" in datasets.columns:
            datasets = datasets.copy()
            datasets.index = pd.Index(datasets["did"].tolist(), name="did")
        return datasets

    def _get_listing_row(
        self,
        dataset_id: str,
        *,
        fetch_if_missing: bool = True,
    ) -> pd.Series | None:
        if self._dataset_listing_cache is None and not fetch_if_missing:
            return None

        datasets = self._get_all_dataset_listing() if fetch_if_missing else self._dataset_listing_cache
        if datasets is None or datasets.empty:
            return None

        lookup_keys: list[Any] = [dataset_id]
        if dataset_id.isdigit():
            lookup_keys.insert(0, int(dataset_id))

        for key in lookup_keys:
            try:
                row = datasets.loc[key]
            except KeyError:
                continue
            if isinstance(row, pd.DataFrame):
                return row.iloc[0]
            return row
        return None

    def _resolve_openml_dest(
        self,
        dataset_id: str,
        dest_dir: Path | None,
    ) -> Path:
        if dest_dir is None:
            dest_dir = self.data_dir / self.source_name / "arff" / dataset_id
        return self._resolve_dest(dataset_id, dest_dir)

    def _download_dataset_assets(self, ds: Any) -> None:
        if getattr(ds, "data_file", None):
            return

        arff_file = _get_dataset_arff(ds)
        ds.data_file = str(arff_file)

    def _save_dataset_arff(self, ds: Any, out_path: Path) -> dict[str, str]:
        source_arff = _normalize_str(getattr(ds, "data_file", ""))
        if not source_arff:
            raise ValueError(f"OpenML ARFF file is unavailable for dataset {ds.dataset_id}")

        source_path = Path(source_arff)
        if not source_path.exists():
            raise FileNotFoundError(f"OpenML ARFF cache file does not exist: {source_path}")

        shutil.copy2(source_path, out_path)
        return {
            "strategy": "copied_openml_arff",
            "source_arff": str(source_path),
            "local_file": out_path.name,
            "storage_format": "arff",
        }

    @staticmethod
    def _row_int(row: pd.Series | None, key: str) -> int | None:
        if row is None or key not in row:
            return None
        value = row.get(key)
        if _is_missing(value):
            return None
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _dataset_url(dataset_id: str, ds: Any | None = None) -> str:
        if ds is not None:
            openml_url = _normalize_str(getattr(ds, "openml_url", ""))
            if openml_url:
                return openml_url
        return f"https://www.openml.org/search?type=data&id={dataset_id}"
