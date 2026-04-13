"""OpenML dataset downloader."""

from __future__ import annotations

import json
from pathlib import Path

import openml
import pandas as pd

from base import BaseDownloader, DatasetInfo


class OpenMLDownloader(BaseDownloader):
    """Download datasets from OpenML (openml.org)."""

    @property
    def source_name(self) -> str:
        return "openml"

    def search(self, query: str, max_results: int = 20) -> list[DatasetInfo]:
        datasets = openml.datasets.list_datasets(output_format="dataframe")
        assert isinstance(datasets, pd.DataFrame)

        mask = datasets["name"].str.contains(query, case=False, na=False) | datasets[
            "description"
        ].str.contains(query, case=False, na=False)
        matched = datasets[mask].head(max_results)
        return [self._row_to_info(row) for _, row in matched.iterrows()]

    def list_datasets(self, max_results: int = 50, **filters) -> list[DatasetInfo]:
        datasets = openml.datasets.list_datasets(output_format="dataframe")
        assert isinstance(datasets, pd.DataFrame)

        for key, value in filters.items():
            if key in datasets.columns:
                datasets = datasets[datasets[key] == value]

        return [
            self._row_to_info(row) for _, row in datasets.head(max_results).iterrows()
        ]

    def info(self, dataset_id: str) -> DatasetInfo:
        ds = openml.datasets.get_dataset(
            int(dataset_id), download_data=False, download_qualities=False
        )
        return DatasetInfo(
            name=ds.name,
            source=self.source_name,
            dataset_id=str(ds.dataset_id),
            description=ds.description or "",
            tags=list(ds.tag) if ds.tag else [],
            url=ds.openml_url or "",
            extra={
                "format": ds.format,
                "version": ds.version,
                "default_target": ds.default_target_attribute,
            },
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
            int(dataset_id), download_data=True, download_qualities=False
        )
        df, *_ = ds.get_data()

        # Save data as parquet
        out_path = dest / f"{ds.name}.parquet"
        df.to_parquet(out_path, index=False)

        task_type = self._infer_task_type(int(dataset_id))

        # Save metadata
        metadata = {
            "name": ds.name,
            "source": "openml",
            "dataset_id": dataset_id,
            "description": ds.description or "",
            "url": ds.openml_url or "",
            "default_target": ds.default_target_attribute or "",
            "task_type": task_type,
            "tags": list(ds.tag) if ds.tag else [],
        }
        meta_path = dest / "metadata.json"
        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return dest

    # ------------------------------------------------------------------

    def _row_to_info(self, row: pd.Series) -> DatasetInfo:
        return DatasetInfo(
            name=str(row.get("name", "")),
            source=self.source_name,
            dataset_id=str(row.name),  # index is the dataset id
            description=str(row.get("description", ""))[:200],
            tags=str(row.get("tag", "")).split(",") if row.get("tag") else [],
            extra={
                "format": str(row.get("format", "")),
                "version": str(row.get("version", "")),
                "num_instances": int(row.get("NumberOfInstances", 0)),
                "num_features": int(row.get("NumberOfFeatures", 0)),
            },
        )
