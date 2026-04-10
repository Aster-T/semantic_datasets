"""Use LLM to evaluate column header semantic quality and extract mappings."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from openai import OpenAI

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是一个表格数据语义分析专家。你的任务是判断表格列名的语义质量，并在可能时从数据集描述中提取列名到语义名称的映射。

规则：
1. "quality" 只有两个值：
   - "high"：列名本身就具有清晰的语义含义，例如 ["Age", "Gender", "Income", "City"]
   - "low"：列名是占位符或编码，例如 ["V1", "V2", "C1", "X0", "col_1", "feature_0"]

2. "column_mapping"：
   - 如果 quality 为 "high"，column_mapping 为空对象 {}
   - 如果 quality 为 "low" 且描述中包含列名的语义解释，则提取映射，例如 {"V1": "Recency", "V2": "Frequency"}
   - 如果 quality 为 "low" 但描述中没有足够的语义信息，column_mapping 也为空对象 {}

你必须严格输出如下 JSON 格式，不要输出任何其他内容：
{
  "quality": "high" 或 "low",
  "column_mapping": {}
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
    quality: str  # "high" or "low"
    columns: list[str]
    column_mapping: dict[str, str]  # original -> semantic name


class ColumnEvaluator:
    """Evaluate column header semantic quality using an LLM."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
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
            description=description[:3000],  # truncate very long descriptions
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
            result = {"quality": "low", "column_mapping": {}}

        return EvalResult(
            dataset_id=dataset_id,
            source=source,
            quality=result.get("quality", "low"),
            columns=columns,
            column_mapping=result.get("column_mapping", {}),
        )
