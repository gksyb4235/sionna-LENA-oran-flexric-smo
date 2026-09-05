"""MongoDB decomposition-pattern lookup for the Decomposition Agent."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = "decomposition"
DEFAULT_COLLECTION = "decomposition_patterns"
DEFAULT_TIMEOUT_MS = 1500
DEFAULT_LIMIT = 3
MAX_STRING_LENGTH = 1200
MAX_LIST_ITEMS = 20
MAX_DICT_ITEMS = 50
PATTERN_FIELDS = {
    "pattern_id",
    "status",
    "intent_class",
    "summary",
    "keywords",
    "required_semantics",
    "decomposition_rules",
    "validation",
    "examples",
    "version",
}


try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover - used only before dependencies are installed.

    def tool(func=None, **_kwargs):  # type: ignore[no-untyped-def]
        if func is None:
            return lambda wrapped: wrapped
        return func


@dataclass(frozen=True)
class DecompositionPatternContext:
    """Compact active decomposition patterns returned from MongoDB."""

    used: bool
    documents: list[dict[str, Any]]


def _log(message: str) -> None:
    print(f"decomposition-pattern-search: {message}", file=sys.stderr)


def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _timeout_ms() -> int:
    raw_value = os.getenv("DECOMPOSITION_KNOWLEDGE_TIMEOUT_MS", str(DEFAULT_TIMEOUT_MS))
    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_TIMEOUT_MS
    return max(100, min(value, 10000))


def _sanitize(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value if len(value) <= MAX_STRING_LENGTH else f"{value[:MAX_STRING_LENGTH]}..."
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return [_sanitize(item) for item in value[:MAX_LIST_ITEMS]]
    if isinstance(value, dict):
        return {
            str(key): _sanitize(item)
            for index, (key, item) in enumerate(value.items())
            if index < MAX_DICT_ITEMS
        }
    return str(value)


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) and item.strip() for item in value)


def _valid_pattern(document: Any) -> bool:
    if not isinstance(document, dict) or document.get("status") != "active":
        return False
    if not all(isinstance(document.get(key), str) and document[key].strip() for key in ("pattern_id", "intent_class", "summary", "version")):
        return False
    if not all(_is_string_list(document.get(key)) for key in ("keywords", "required_semantics", "decomposition_rules")):
        return False
    validation = document.get("validation")
    if not isinstance(validation, dict):
        return False
    concepts = validation.get("required_concepts")
    if not isinstance(concepts, dict) or not concepts or not all(isinstance(name, str) and _is_string_list(terms) for name, terms in concepts.items()):
        return False
    if not isinstance(validation.get("preserve_numbers"), bool) or not isinstance(validation.get("preserve_uppercase_tokens"), bool):
        return False
    examples = document.get("examples")
    return isinstance(examples, list) and bool(examples) and all(
        isinstance(example, dict)
        and isinstance(example.get("intent"), str)
        and _is_string_list(example.get("expected_subtasks"))
        for example in examples
    )


def _compact_pattern(document: dict[str, Any]) -> dict[str, Any] | None:
    compact = {key: _sanitize(document[key]) for key in PATTERN_FIELDS if key in document}
    return compact if _valid_pattern(compact) else None


def find_decomposition_pattern_context(intent: str, limit: int = DEFAULT_LIMIT) -> DecompositionPatternContext:
    """Return only active text-matched decomposition patterns; never arbitrary fallback documents."""

    _load_env_file()
    uri = os.getenv("DECOMPOSITION_KNOWLEDGE_MONGODB_URI", "").strip()
    if not uri:
        return DecompositionPatternContext(used=False, documents=[])

    try:
        from pymongo import MongoClient
    except ImportError:
        _log("pymongo is not installed; continuing without decomposition patterns")
        return DecompositionPatternContext(used=False, documents=[])

    database_name = os.getenv("DECOMPOSITION_KNOWLEDGE_DATABASE", DEFAULT_DATABASE).strip() or DEFAULT_DATABASE
    collection_name = os.getenv("DECOMPOSITION_KNOWLEDGE_COLLECTION", DEFAULT_COLLECTION).strip() or DEFAULT_COLLECTION
    try:
        safe_limit = min(max(int(limit), 1), 10)
    except (TypeError, ValueError):
        safe_limit = DEFAULT_LIMIT

    client = None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=_timeout_ms())
        client.admin.command("ping")
        collection = client[database_name][collection_name]
        projection = {field: 1 for field in PATTERN_FIELDS}
        projection.update({"_id": 0, "score": {"$meta": "textScore"}})
        documents = list(
            collection.find(
                {"status": "active", "$text": {"$search": intent}},
                projection,
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(safe_limit)
        )
    except Exception as exc:  # noqa: BLE001 - decomposition remains best effort when knowledge is unavailable.
        _log(f"lookup failed; continuing without decomposition patterns ({type(exc).__name__})")
        return DecompositionPatternContext(used=False, documents=[])
    finally:
        if client is not None:
            client.close()

    compact_documents = [compact for document in documents if (compact := _compact_pattern(document)) is not None]
    return DecompositionPatternContext(used=bool(compact_documents), documents=compact_documents)


def _normalized_text(values: list[str]) -> str:
    return " ".join(values).casefold()


def validate_decomposition_semantics(intent: str, subtasks: list[str], patterns: list[dict[str, Any]]) -> None:
    """Fail when a matched pattern's required meaning disappeared from model subtasks."""

    if not patterns:
        return
    output = _normalized_text(subtasks)
    violations: list[str] = []
    for pattern in patterns:
        validation = pattern.get("validation")
        if not isinstance(validation, dict):
            violations.append("invalid_pattern_validation")
            continue
        concepts = validation.get("required_concepts", {})
        for name, terms in concepts.items():
            if not any(str(term).casefold() in output for term in terms):
                violations.append(f"missing_concept:{name}")
        if validation.get("preserve_numbers"):
            for number in set(re.findall(r"\d+(?:\.\d+)?", intent)):
                if number not in output:
                    violations.append(f"missing_number:{number}")
        if validation.get("preserve_uppercase_tokens"):
            for token in set(re.findall(r"\b[A-Z][A-Z0-9_-]+\b", intent)):
                if token.casefold() not in output:
                    violations.append(f"missing_target:{token}")
    if violations:
        raise ValueError("decomposition_pattern_violation:" + ",".join(sorted(set(violations))))


@tool
def search_decomposition_patterns(intent: str, limit: int = DEFAULT_LIMIT) -> str:
    """Search active MongoDB decomposition patterns relevant to the original intent."""

    context = find_decomposition_pattern_context(intent=intent, limit=limit)
    return json.dumps({"used": context.used, "documents": context.documents}, ensure_ascii=False)
