"""OpenML dataset downloader."""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import openml
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse
from openml.datasets.functions import _get_cache_directory, _get_dataset_arff, _get_dataset_parquet

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


def _normalize_nested_value(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, dict):
        return {str(key): _normalize_nested_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_nested_value(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _normalize_nested_value(value.item())
        except (AttributeError, TypeError, ValueError):
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@contextmanager
def _openml_factorize_list_compat() -> Iterator[None]:
    """Patch pandas.factorize so OpenML 0.15 works with pandas 3 list inputs."""
    original_factorize = pd.factorize
    if getattr(original_factorize, "_semantic_datasets_openml_compat", False):
        yield
        return

    def compat_factorize(values: Any, *args: Any, **kwargs: Any):
        if isinstance(values, list):
            values = np.asarray(values, dtype=object)
        return original_factorize(values, *args, **kwargs)

    setattr(compat_factorize, "_semantic_datasets_openml_compat", True)
    pd.factorize = compat_factorize
    try:
        yield
    finally:
        pd.factorize = original_factorize


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
        dest = self._resolve_dest(dataset_id, dest_dir)

        ds = openml.datasets.get_dataset(
            int(dataset_id), download_data=False, download_qualities=False
        )
        self._download_dataset_assets(ds)
        out_path = dest / "table.parquet"
        download_info = self._save_dataset_parquet(ds, out_path)

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
            "local_table_file": out_path.name,
            "storage_format": "parquet",
            "download_strategy": download_info["strategy"],
            "openml_cached_parquet": download_info["source_parquet"],
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

    def _download_dataset_assets(self, ds: Any) -> None:
        if getattr(ds, "parquet_file", None) or getattr(ds, "data_file", None):
            return

        parquet_url = _normalize_str(getattr(ds, "_parquet_url", ""))
        if parquet_url:
            try:
                parquet_file = _get_dataset_parquet(ds)
                if parquet_file is not None:
                    self._validate_openml_parquet(parquet_file)
                    ds.parquet_file = str(parquet_file)
                    ds.data_file = None
                    return
                log.warning(
                    "OpenML parquet unavailable for dataset %s; falling back to ARFF.",
                    ds.dataset_id,
                )
            except Exception as exc:
                self._cleanup_partial_openml_parquet(ds)
                ds.parquet_file = None
                log.warning(
                    "OpenML parquet download failed for dataset %s (%s); falling back to ARFF.",
                    ds.dataset_id,
                    exc,
                )

        arff_file = _get_dataset_arff(ds)
        ds.data_file = str(arff_file)
        ds.parquet_file = None

    @staticmethod
    def _validate_openml_parquet(parquet_file: Path) -> None:
        try:
            pq.read_schema(parquet_file)
        except Exception as exc:
            raise ValueError(f"Invalid OpenML parquet cache file: {parquet_file}") from exc

    @staticmethod
    def _cleanup_partial_openml_parquet(ds: Any) -> None:
        try:
            cache_dir = _get_cache_directory(ds)
        except Exception:
            return

        candidates = [
            cache_dir / f"dataset_{ds.dataset_id}.pq",
            cache_dir / "dataset.pq",
        ]
        for candidate in candidates:
            if candidate.exists():
                candidate.unlink()

    def _save_dataset_parquet(self, ds, out_path: Path) -> dict[str, str]:
        source_parquet = _normalize_str(getattr(ds, "parquet_file", ""))
        if source_parquet:
            source_path = Path(source_parquet)
            if source_path.exists():
                shutil.copy2(source_path, out_path)
                return {
                    "strategy": "copied_openml_parquet",
                    "source_parquet": str(source_path),
                }

        df = self._load_dataset_frame(ds)
        df = self._normalize_dataframe_for_parquet(df, dataset_id=str(ds.dataset_id))
        df.to_parquet(out_path, index=False)
        return {
            "strategy": "rewritten_from_dataframe",
            "source_parquet": source_parquet,
        }

    def _load_dataset_frame(self, ds: Any) -> pd.DataFrame:
        with _openml_factorize_list_compat():
            data, _, attribute_names = ds._load_data()

        if scipy.sparse.issparse(data):
            dtype = getattr(data, "dtype", np.dtype("float32"))
            dense_bytes = data.shape[0] * data.shape[1] * dtype.itemsize
            log.info(
                "Dataset %s converting sparse OpenML matrix to dense dataframe for parquet export (~%.1f MB).",
                ds.dataset_id,
                dense_bytes / (1024 * 1024),
            )
            return pd.DataFrame(data.toarray(), columns=attribute_names)

        if isinstance(data, pd.DataFrame):
            return data

        return pd.DataFrame(data, columns=attribute_names)

    def _normalize_dataframe_for_parquet(
        self,
        df: pd.DataFrame,
        *,
        dataset_id: str,
    ) -> pd.DataFrame:
        converted = df.copy()
        densified_columns: list[str] = []
        rewritten_columns: list[str] = []

        for column in converted.columns:
            series = converted[column]
            if isinstance(series.dtype, pd.SparseDtype):
                converted[column] = series.sparse.to_dense()
                densified_columns.append(str(column))
                series = converted[column]
            if series.dtype != "object":
                continue
            if not series.map(self._needs_json_serialization).any():
                continue

            converted[column] = series.map(
                lambda value: None
                if _is_missing(value)
                else json.dumps(_normalize_nested_value(value), ensure_ascii=False)
            )
            rewritten_columns.append(str(column))

        if densified_columns:
            log.info(
                "Dataset %s densified sparse columns for parquet compatibility: %s",
                dataset_id,
                ", ".join(densified_columns),
            )
        if rewritten_columns:
            log.info(
                "Dataset %s rewrote nested columns to JSON strings for parquet compatibility: %s",
                dataset_id,
                ", ".join(rewritten_columns),
            )

        return converted

    @staticmethod
    def _needs_json_serialization(value: Any) -> bool:
        if _is_missing(value):
            return False
        return isinstance(value, (list, tuple, set, dict))

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
