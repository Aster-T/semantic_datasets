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
**Role:** A data engineering expert

**Profile:**

- description: Filter valid data to train a tabular foundation model with abilities of process tabular data with text modality.
- background: Be familiar with the knowledge of the foundation model of tables.

**Skills:**

- Read tabular data sheets in parquet format
- Understand the description of the dataset in json format
- Output a structured JSON file

**Rules:**

1. Semantic headers are preferred
   Column names that clearly and directly describe the feature's meaning (e.g., "age", "review_text", "purchase_amount") should be considered **high quality**.

2. Placeholder headers are low quality
   Column names that contain no semantic meaning like "V1", "Column1", "C1" should be considered **low quality**.

3. Use dataset description if possible
   If placeholder headers exist but **a semantic mapping appears in the dataset description**, extract the mapping and mark quality as **mid**.

4. Never fabricate information
   Only extract mappings if they **explicitly appear in the description text**.

5. Two task types
   classification or regression

**Workflows:**

- Goal: Filter the dataset with valid semantic data

- **Step 1: Read dataset**
  Read a parquet table and dataset description JSON file, and then extract the column names, types and description text.

- **Step 2: Evaluate Header Quality**
  Check whether the column headers contain semantic meaning. If **more than 50%** of the columns have meaningful names (as defined in Rule 1), mark the overall dataset "quality" as "high". Otherwise, proceed to Step 3.

- **Step 3: Detect placeholder headers**
  If headers contain placeholders such as `V1`, `C1`, `Column1`, `Feature_1`, then check the dataset description. If no mapping exists, set "quality" as "low". Yet there are a mapping like "V1 = Recency \\n V2 = Frequency", set "quality" as "mid". Then add a column_mapping section.

- **Step 4: Identify Task Information**
  Determine "task_type" ("classification" or "regression") and "target_column"

- **Step 5: Generate Metadata**

You must strictly output the following JSON format and nothing else:
{
    "datasetname": "XXX",
    "description": "",
    "quality": "high" or "mid" or "low",
    "columns": {
        "col_name": {
            "type": "",
            "description": ""
        }
    },
    "columns_mapping": {
        "V1": "Recency",
        "V2": "Frequency"
    },
    "Task_type": "",
    "target_column": ""
}

**Example Output:**
{
    "datasetname": "blood-transfusion-service-center",
    "description": "Data taken from the Blood Transfusion Service Center in Hsin-Chu City in Taiwan -- this is a classification problem.",
    "quality": "mid",
    "columns": {
        "V1": {
            "type": "numeric",
            "description": "months since last donation"
        },
        "V2": {
            "type": "numeric",
            "description": "total number of donation"
        },
        "V3": {
            "type": "numeric",
            "description": "total blood donated in c.c."
        },
        "V4": {
            "type": "numeric",
            "description": "months since first donation"
        },
        "Class": {
            "type": "nominal",
            "description": "binary variable representing whether he/she donated blood in March 2007"
        }
    },
    "columns_mapping": {
        "V1": "Recency",
        "V2": "Frequency"
    },
    "Task_type": "classification",
    "target_column": "Class"
}"""

USER_PROMPT_TEMPLATE = """\
列名: {columns}

数据集描述:
{description}"""


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

    LOCAL_BASE_URL = "http://localhost:8000/v1"
    LOCAL_API_KEY = "no-key-required"

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
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def evaluate(
        self,
        columns: list[str],
        description: str,
        dataset_id: str = "",
        source: str = "",
    ) -> EvalResult:
        """Evaluate column quality and extract semantic mappings if possible."""
        user_prompt = USER_PROMPT_TEMPLATE.format(
            columns=json.dumps(columns, ensure_ascii=False),
            description=clean_description(description),
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
            task_type=result.get("Task_type", ""),
            target_column=result.get("target_column", ""),
        )

    async def async_evaluate(
        self,
        columns: list[str],
        description: str,
        dataset_id: str = "",
        source: str = "",
    ) -> EvalResult:
        """Async version of evaluate for concurrent execution."""
        user_prompt = USER_PROMPT_TEMPLATE.format(
            columns=json.dumps(columns, ensure_ascii=False),
            description=clean_description(description),
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
            task_type=result.get("Task_type", ""),
            target_column=result.get("target_column", ""),
        )
