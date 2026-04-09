"""Extract column-level semantic information from dataset descriptions."""

from __future__ import annotations

import re

from semantic_datasets.base import ColumnSemantic


def parse_attribute_semantics(
    description: str, column_names: list[str]
) -> list[ColumnSemantic]:
    """Parse a dataset description to extract semantic info for each column.

    Handles common patterns in OpenML / UCI descriptions:
      - "V1: Recency - months since last donation"
      - "1. age: age in years"
      - "- sepal_length: length of sepal in cm"
      - "Attribute Information" / "Features" sections
    """
    if not description:
        return [ColumnSemantic(column_name=c) for c in column_names]

    # Try to isolate the attribute/feature section if present
    attr_section = _extract_attribute_section(description)
    text = attr_section if attr_section else description

    # Build a mapping: column_name -> (semantic_name, description)
    mapping: dict[str, tuple[str, str]] = {}

    for col in column_names:
        sem_name, sem_desc = _find_column_semantic(col, text)
        if sem_name or sem_desc:
            mapping[col] = (sem_name, sem_desc)

    # If no per-column matches, try line-by-line ordered matching
    if not mapping:
        mapping = _try_ordered_matching(text, column_names)

    return [
        ColumnSemantic(
            column_name=col,
            semantic_name=mapping.get(col, ("", ""))[0],
            description=mapping.get(col, ("", ""))[1],
        )
        for col in column_names
    ]


def _extract_attribute_section(description: str) -> str:
    """Extract the 'Attribute Information' or similar section."""
    patterns = [
        r"(?i)(?:attribute|feature|variable|column)\s*(?:information|description|details?)s?\s*[:\-]?\s*\n(.*?)(?:\n\s*\n|\Z)",
        r"(?i)attributes?\s*[:\-]\s*\n(.*?)(?:\n\s*\n|\Z)",
    ]
    for pat in patterns:
        m = re.search(pat, description, re.DOTALL)
        if m:
            return m.group(1).strip()
    return ""


def _find_column_semantic(
    column_name: str, text: str
) -> tuple[str, str]:
    """Find semantic info for a specific column name in the text.

    Looks for patterns like:
      "V1: Recency - months since last donation"
      "V1 - Recency (months since last donation)"
      "V1. Recency: months since last donation"
    """
    escaped = re.escape(column_name)

    patterns = [
        # "V1: Recency - months since last donation"
        # "V1 - Recency - months since last donation"
        rf"(?:^|\n)\s*[\-\*\•]?\s*{escaped}\s*[:.\-\)]\s*(.+)",
        # "1) V1 = Recency ..."
        rf"(?:^|\n)\s*\d+[\).\]]\s*{escaped}\s*[:=\-]\s*(.+)",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            raw = m.group(1).strip().rstrip(".")
            sem_name, sem_desc = _split_name_desc(raw)
            return sem_name, sem_desc

    return "", ""


def _split_name_desc(raw: str) -> tuple[str, str]:
    """Split 'Recency - months since last donation' into (name, description).

    Heuristic: the first segment before ' - ', ':', or '(' is the semantic name,
    the rest is the description.
    """
    separators = [" - ", " -- ", ": ", " ("]
    for sep in separators:
        if sep in raw:
            parts = raw.split(sep, 1)
            name = parts[0].strip()
            desc = parts[1].strip().rstrip(")")
            return name, desc

    # No separator found — if short enough, treat as semantic name
    if len(raw) < 60:
        return raw, ""
    return "", raw


def _try_ordered_matching(
    text: str, column_names: list[str]
) -> dict[str, tuple[str, str]]:
    """Fallback: match bullet/numbered lines to columns in order."""
    # Extract lines that look like attribute descriptions
    lines = [
        line.strip()
        for line in text.split("\n")
        if re.match(r"^\s*[\-\*\•\d]", line) and len(line.strip()) > 5
    ]

    mapping: dict[str, tuple[str, str]] = {}
    for col, line in zip(column_names, lines):
        # Strip leading bullet/number
        cleaned = re.sub(r"^[\-\*\•\d\).\]\s]+", "", line).strip()
        if cleaned:
            sem_name, sem_desc = _split_name_desc(cleaned)
            mapping[col] = (sem_name, sem_desc)

    return mapping
