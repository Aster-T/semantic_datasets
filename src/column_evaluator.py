"""Use LLM to evaluate column header semantic quality and extract mappings."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from openai import AsyncOpenAI, OpenAI

log = logging.getLogger(__name__)


def clean_description(text: str) -> str:
    """Remove HTML tags, URLs, and excessive whitespace from description text."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove markdown link syntax residuals [text](url)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Remove markdown image syntax ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    # Collapse multiple whitespace / newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


SYSTEM_PROMPT = """\
**Role:** A data engineering expert who evaluates tabular dataset header quality.

**Goal:** Given a list of column names, their types, and the dataset description, \
assess whether the column headers carry semantic meaning and produce structured metadata.

**Quality Criteria:**

| Quality | Condition |
|---------|-----------|
| high    | More than 50% of column headers are semantically meaningful (e.g., "age", "gender", "income", "review_text"). Even if the description is empty or uninformative, the headers alone are sufficient. |
| mid     | More than 50% of headers are placeholders (e.g., "V1", "V2", "C1", "Feature_1", "att1"), BUT the dataset description explicitly provides a mapping from placeholders to meaningful names. |
| low     | More than 50% of headers are placeholders AND the description does NOT provide any mapping to meaningful names. |

**Placeholder Detection:**
A column name is a placeholder if it matches patterns like:
- Single letter + number: V1, V2, C1, C2, X1, X2
- Generic prefix + number: Feature_1, Column1, att1, Attr_1, f1, col_1
- Pure index: 0, 1, 2, ...

A column name is semantically meaningful if it describes the feature's real-world meaning, \
such as "age", "gender", "price", "review_text", "match", "wave", "d_age".
Note: abbreviated but interpretable names (e.g., "d_age", "age_o", "has_null") are still meaningful.

**Rules:**
1. Judge quality based ONLY on the column names and description provided. Never fabricate information.
2. If headers are meaningful, output "high" regardless of whether the description is informative.
3. For "mid" quality, extract EVERY placeholder-to-name mapping found in the description into "columns_mapping".
4. For "low" quality, leave "columns_mapping" as an empty object {}.
5. The target column is provided — do not guess or change it.
6. "type" must be one of: "numeric", "nominal", "string", "ordinal".
7. "task_type" is either "classification" or "regression".

**Output — strict JSON, nothing else:**
{
    "datasetname": "",
    "description": "",
    "quality": "high" or "mid" or "low",
    "columns": {
        "<col_name>": {
            "type": "",
            "description": ""
        }
    },
    "columns_mapping": {},
    "task_type": "",
    "target_column": ""
}

Notes on the "columns" field:
- If quality is "high": describe each column based on its name.
- If quality is "mid": describe each column based on the mapping found in the description.
- If quality is "low": set description to "unknown" for placeholder columns.

**Example 1 (high quality):**

Input:
  Column names: match, has_null, wave, gender, age, age_o, d_age, d_d_age
  Types: nominal, nominal, numeric, nominal, numeric, numeric, numeric, nominal
  Description: "TEST"
  Task type: classification
  Target column: match

Output:
{
    "datasetname": "SpeedDating",
    "description": "TEST",
    "quality": "high",
    "columns": {
        "match": {"type": "nominal", "description": "whether the pair matched"},
        "has_null": {"type": "nominal", "description": "whether the record has null values"},
        "wave": {"type": "numeric", "description": "wave number of the speed dating event"},
        "gender": {"type": "nominal", "description": "gender of the participant"},
        "age": {"type": "numeric", "description": "age of the participant"},
        "age_o": {"type": "numeric", "description": "age of the partner"},
        "d_age": {"type": "numeric", "description": "age difference"},
        "d_d_age": {"type": "nominal", "description": "binned age difference"}
    },
    "columns_mapping": {},
    "task_type": "classification",
    "target_column": "match"
}

**Example 2 (mid quality):**

Input:
  Column names: Class, V1, V2, V3, V4
  Types: nominal, numeric, numeric, numeric, numeric
  Description: "Data taken from the Blood Transfusion Service Center. V1=Recency (months since last donation), V2=Frequency (total number of donations), V3=Monetary (total blood donated in c.c.), V4=Time (months since first donation)."
  Task type: classification
  Target column: Class

Output:
{
    "datasetname": "blood-transfusion-service-center",
    "description": "Data taken from the Blood Transfusion Service Center. V1=Recency...",
    "quality": "mid",
    "columns": {
        "Class": {"type": "nominal", "description": "whether the person donated blood in March 2007"},
        "V1": {"type": "numeric", "description": "months since last donation"},
        "V2": {"type": "numeric", "description": "total number of donations"},
        "V3": {"type": "numeric", "description": "total blood donated in c.c."},
        "V4": {"type": "numeric", "description": "months since first donation"}
    },
    "columns_mapping": {
        "V1": "Recency",
        "V2": "Frequency",
        "V3": "Monetary",
        "V4": "Time"
    },
    "task_type": "classification",
    "target_column": "Class"
}

**Example 3 (low quality):**

Input:
  Column names: Class, V2, V3, V4, V5, V6, V7, V8
  Types: nominal, nominal, nominal, numeric, nominal, nominal, nominal, nominal
  Description: "TEST"
  Task type: classification
  Target column: Class

Output:
{
    "datasetname": "dresses-sales",
    "description": "TEST",
    "quality": "low",
    "columns": {
        "Class": {"type": "nominal", "description": "target class label"},
        "V2": {"type": "nominal", "description": "unknown"},
        "V3": {"type": "nominal", "description": "unknown"},
        "V4": {"type": "numeric", "description": "unknown"},
        "V5": {"type": "nominal", "description": "unknown"},
        "V6": {"type": "nominal", "description": "unknown"},
        "V7": {"type": "nominal", "description": "unknown"},
        "V8": {"type": "nominal", "description": "unknown"}
    },
    "columns_mapping": {},
    "task_type": "classification",
    "target_column": "Class"
}"""


USER_PROMPT_TEMPLATE = """\
Dataset name: {dataset_name}

Column names: {columns}
Types: {column_types}

Dataset description:
{description}

Known task type: {task_type}
Known target column: {default_target}

Assess the header quality and output the JSON."""


def build_user_prompt(
    columns: list[str],
    column_types: list[str],
    description: str,
    dataset_name: str = "",
    task_type: str = "",
    default_target: str = "",
) -> str:
    """Render the user prompt with dataset metadata and normalized text."""
    return USER_PROMPT_TEMPLATE.format(
        dataset_name=dataset_name or "unknown",
        columns=json.dumps(columns, ensure_ascii=False),
        column_types=json.dumps(column_types, ensure_ascii=False),
        description=clean_description(description),
        task_type=task_type or "unknown",
        default_target=default_target or "unknown",
    )


@dataclass
class EvalResult:
    """Result of column quality evaluation for one dataset."""

    dataset_id: str
    source: str
    quality: str  # "high", "mid", or "low"
    description: str
    columns: dict[str, dict]  # col_name -> {"type": ..., "description": ...}
    columns_mapping: dict[str, str]  # original -> semantic name
    task_type: str  # "classification" or "regression"
    target_column: str


class ColumnEvaluator:
    """Evaluate column header semantic quality using an LLM."""

    LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
    LOCAL_API_KEY = "no-key-required"
    LOCAL_TIMEOUT_SECONDS = 300.0

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        local: bool = False,
    ):
        if local:
            base_url = base_url or self.LOCAL_BASE_URL
            api_key = api_key or self.LOCAL_API_KEY
        timeout = self.LOCAL_TIMEOUT_SECONDS if local else None
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self._model = model

    def evaluate(
        self,
        columns: list[str],
        column_types: list[str],
        description: str,
        dataset_id: str = "",
        dataset_name: str = "",
        source: str = "",
        task_type: str = "",
        default_target: str = "",
    ) -> EvalResult:
        """Evaluate column quality and extract semantic mappings if possible."""
        user_prompt = build_user_prompt(
            columns=columns,
            column_types=column_types,
            description=description,
            dataset_name=dataset_name or dataset_id,
            task_type=task_type,
            default_target=default_target,
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            log.warning(f"LLM returned invalid JSON for {dataset_id}: {content[:200]}")
            result = {"quality": "low", "columns_mapping": {}}

        return EvalResult(
            dataset_id=dataset_id,
            source=source,
            quality=result.get("quality", "low"),
            description=result.get("description", ""),
            columns=result.get("columns", {}),
            columns_mapping=result.get("columns_mapping", {}),
            task_type=result.get("task_type") or result.get("Task_type", ""),
            target_column=result.get("target_column", ""),
        )

    async def async_evaluate(
        self,
        columns: list[str],
        column_types: list[str],
        description: str,
        dataset_id: str = "",
        dataset_name: str = "",
        source: str = "",
        task_type: str = "",
        default_target: str = "",
    ) -> EvalResult:
        """Async version of evaluate for concurrent execution."""
        user_prompt = build_user_prompt(
            columns=columns,
            column_types=column_types,
            description=description,
            dataset_name=dataset_name or dataset_id,
            task_type=task_type,
            default_target=default_target,
        )

        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            log.warning(f"LLM returned invalid JSON for {dataset_id}: {content[:200]}")
            result = {"quality": "low", "columns_mapping": {}}

        return EvalResult(
            dataset_id=dataset_id,
            source=source,
            quality=result.get("quality", "low"),
            description=result.get("description", ""),
            columns=result.get("columns", {}),
            columns_mapping=result.get("columns_mapping", {}),
            task_type=result.get("task_type") or result.get("Task_type", ""),
            target_column=result.get("target_column", ""),
        )
