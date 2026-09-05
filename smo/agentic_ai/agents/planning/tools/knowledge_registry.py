"""Knowledge DB lookup utilities for the Planning Agent."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = "knowledge"
DEFAULT_AGENTS_COLLECTION = "agents"
DEFAULT_TOOLS_COLLECTION = "tools"
DEFAULT_QUERY_LIMIT = 10
DEFAULT_TIMEOUT_MS = 1500
DEFAULT_SIMILARITY_THRESHOLD = 0.18
MAX_STRING_LENGTH = 1200
MAX_LIST_ITEMS = 20
MAX_DICT_ITEMS = 50
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "each", "need", "needs", "into",
    "using", "use", "all", "are", "task", "subtask", "what", "how", "then", "than", "should",
    "could", "would", "must", "have", "has", "been", "being", "was", "were", "will", "shall",
}
ACTIVE_STATUS_VALUES = ["active", "available"]
PLANNER_MANIFEST_FIELDS = (
    "contract_version",
    "purpose",
    "capabilities",
    "constraints",
    "planning_contract",
    "planning_schema_path",
)
PLANNER_MANIFEST_KEYS = {*PLANNER_MANIFEST_FIELDS, "contract_ref"}
FORBIDDEN_MANIFEST_SCHEMA_KEYS = {
    "properties",
    "oneOf",
    "allOf",
    "anyOf",
    "additionalProperties",
    "request_schema",
    "response_schema",
    "approval_request_schema",
    "approval_response_schema",
    "evaluation_request_schema",
}
PUBLIC_GRAPH_NODE_INPUT_FIELDS = {
    "id",
    "agent_id",
    "tool_id",
    "type",
    "role",
    "name",
    "label",
    "source",
    "description",
    "capabilities",
    "operations",
    "owner_agent",
}
SEARCH_FIELD_NAMES = (
    "name",
    "agent_id",
    "tool_id",
    "id",
    "description",
    "role",
    "type",
    "domain",
    "owner_agent",
    "capabilities",
    "tags",
    "keywords",
    "aliases",
    "can_do",
    "cannot_do",
    "operations",
    "tools",
    "inputs",
    "outputs",
)


@dataclass(frozen=True)
class KnowledgeRegistryResult:
    used: bool
    agents: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    available: bool = True
    error: str | None = None
    query_keywords: list[str] = field(default_factory=list)


class CommunicationContractError(ValueError):
    """Raised when a selected Knowledge DB agent contract cannot be honored."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ContractSchemaError(ValueError):
    """Raised when MongoDB contains a malformed or unsupported contract schema."""


SUPPORTED_SCHEMA_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}
SUPPORTED_SCHEMA_KEYS = {
    "type", "const", "enum", "allOf", "anyOf", "oneOf", "not", "required",
    "properties", "additionalProperties", "minItems", "items", "x-unique-by",
    "minLength", "minimum", "exclusiveMinimum", "description", "$comment",
}


def _contract_identifier(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _schema_error(path: str, message: str) -> ValueError:
    return ValueError(f"{path}: {message}")


def _schema_type_matches(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, int | float) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, False)


def _validate_contract_schema(schema: Any, *, path: str) -> None:
    if not isinstance(schema, dict):
        raise ContractSchemaError(f"{path}: contract schema must be an object")
    unknown = sorted(set(schema).difference(SUPPORTED_SCHEMA_KEYS))
    if unknown:
        raise ContractSchemaError(f"{path}: unsupported schema keywords {unknown}")

    expected_type = schema.get("type")
    if expected_type is not None:
        allowed_types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not allowed_types or not all(item in SUPPORTED_SCHEMA_TYPES for item in allowed_types):
            raise ContractSchemaError(f"{path}: unsupported schema type {expected_type!r}")
    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, list) or not enum):
        raise ContractSchemaError(f"{path}: enum must be a non-empty list")
    required = schema.get("required")
    if required is not None and (not isinstance(required, list) or not all(isinstance(item, str) for item in required)):
        raise ContractSchemaError(f"{path}: required must be a string list")
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            raise ContractSchemaError(f"{path}: properties must be an object")
        for key, child in properties.items():
            _validate_contract_schema(child, path=f"{path}.properties.{key}")
    for keyword in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(keyword)
        if branches is None:
            continue
        if not isinstance(branches, list) or not branches:
            raise ContractSchemaError(f"{path}: {keyword} must contain schemas")
        for index, branch in enumerate(branches):
            _validate_contract_schema(branch, path=f"{path}.{keyword}[{index}]")
    if "not" in schema:
        _validate_contract_schema(schema["not"], path=f"{path}.not")
    if "items" in schema:
        _validate_contract_schema(schema["items"], path=f"{path}.items")
    additional = schema.get("additionalProperties")
    if additional is not None and not isinstance(additional, bool | dict):
        raise ContractSchemaError(f"{path}: additionalProperties must be boolean or a schema")
    if isinstance(additional, dict):
        _validate_contract_schema(additional, path=f"{path}.additionalProperties")
    for keyword in ("minItems", "minLength"):
        limit = schema.get(keyword)
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 0):
            raise ContractSchemaError(f"{path}: {keyword} must be a non-negative integer")
    for keyword in ("minimum", "exclusiveMinimum"):
        limit = schema.get(keyword)
        if limit is not None and (not isinstance(limit, int | float) or isinstance(limit, bool)):
            raise ContractSchemaError(f"{path}: {keyword} must be numeric")
    unique_by = schema.get("x-unique-by")
    if unique_by is not None and (not isinstance(unique_by, str) or not unique_by):
        raise ContractSchemaError(f"{path}: x-unique-by must name one field")


def _validate_contract_payload(value: Any, schema: dict[str, Any], *, path: str) -> None:

    if not isinstance(schema, dict):
        raise _schema_error(path, "contract schema must be an object")

    expected_type = schema.get("type")
    if expected_type is not None:
        allowed_types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not allowed_types or not all(isinstance(item, str) for item in allowed_types):
            raise _schema_error(path, "contract schema type must be a string or string list")
        if not any(_schema_type_matches(value, item) for item in allowed_types):
            raise _schema_error(path, f"expected type {allowed_types}")

    if "const" in schema and value != schema["const"]:
        raise _schema_error(path, f"must equal {schema['const']!r}")
    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, list) or value not in enum:
            raise _schema_error(path, f"must be one of {enum!r}")

    for keyword in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(keyword)
        if branches is None:
            continue
        if not isinstance(branches, list) or not branches:
            raise _schema_error(path, f"{keyword} must contain schemas")
        matches = 0
        errors: list[str] = []
        for branch in branches:
            try:
                _validate_contract_payload(value, branch, path=path)
            except ValueError as exc:
                errors.append(str(exc))
            else:
                matches += 1
        if keyword == "allOf" and matches != len(branches):
            raise _schema_error(path, errors[0] if errors else "allOf did not match")
        if keyword == "anyOf" and matches == 0:
            raise _schema_error(path, errors[0] if errors else "anyOf did not match")
        if keyword == "oneOf" and matches != 1:
            detail = errors[0] if matches == 0 and errors else f"matched {matches} branches"
            raise _schema_error(path, f"oneOf requires exactly one match ({detail})")

    excluded = schema.get("not")
    if excluded is not None:
        try:
            _validate_contract_payload(value, excluded, path=path)
        except ValueError:
            pass
        else:
            raise _schema_error(path, "matched a forbidden schema")

    if isinstance(value, dict):
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise _schema_error(path, "required must be a string list")
        missing = [item for item in required if item not in value]
        if missing:
            raise _schema_error(path, f"missing required fields {missing}")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise _schema_error(path, "properties must be an object")
        for key, child_schema in properties.items():
            if key in value:
                _validate_contract_payload(value[key], child_schema, path=f"{path}.{key}")
        extras = [key for key in value if key not in properties]
        additional = schema.get("additionalProperties", True)
        if additional is False and extras:
            raise _schema_error(path, f"unexpected fields {extras}")
        if isinstance(additional, dict):
            for key in extras:
                _validate_contract_payload(value[key], additional, path=f"{path}.{key}")

    if isinstance(value, list):
        minimum_items = schema.get("minItems")
        if isinstance(minimum_items, int) and len(value) < minimum_items:
            raise _schema_error(path, f"must contain at least {minimum_items} items")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _validate_contract_payload(item, item_schema, path=f"{path}[{index}]")
        unique_by = schema.get("x-unique-by")
        if unique_by is not None:
            if not isinstance(unique_by, str) or not unique_by:
                raise _schema_error(path, "x-unique-by must name one field")
            seen: set[str] = set()
            for index, item in enumerate(value):
                if not isinstance(item, dict) or unique_by not in item:
                    raise _schema_error(f"{path}[{index}]", f"missing uniqueness field {unique_by!r}")
                marker = repr(item[unique_by])
                if marker in seen:
                    raise _schema_error(path, f"duplicate {unique_by!r} value")
                seen.add(marker)

    if isinstance(value, str):
        minimum_length = schema.get("minLength")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            raise _schema_error(path, f"must contain at least {minimum_length} characters")

    if isinstance(value, int | float) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, int | float) and value < minimum:
            raise _schema_error(path, f"must be at least {minimum}")
        exclusive_minimum = schema.get("exclusiveMinimum")
        if isinstance(exclusive_minimum, int | float) and value <= exclusive_minimum:
            raise _schema_error(path, f"must be greater than {exclusive_minimum}")


def validate_contract_payload(value: Any, schema: dict[str, Any], *, path: str = "$") -> None:
    """Validate data against the fail-closed JSON Schema subset used by Knowledge DB."""

    _validate_contract_schema(schema, path=f"{path}#schema")
    _validate_contract_payload(value, schema, path=path)


def validate_contract_schema(schema: dict[str, Any], *, path: str = "$schema") -> None:
    _validate_contract_schema(schema, path=path)


def _agent_document_for_node(node: dict[str, Any], knowledge: KnowledgeRegistryResult) -> dict[str, Any] | None:
    exact_values = {
        str(node.get(key)).strip()
        for key in ("id", "agent_id")
        if isinstance(node.get(key), str) and str(node.get(key)).strip()
    }
    exact_matches: list[dict[str, Any]] = []
    for document in knowledge.agents:
        document_ids = {
            str(document.get(key)).strip()
            for key in ("id", "agent_id")
            if isinstance(document.get(key), str) and str(document.get(key)).strip()
        }
        if exact_values.intersection(document_ids):
            exact_matches.append(document)

    name_value = _contract_identifier(node.get("name"))
    name_matches: list[dict[str, Any]] = []
    if name_value:
        for document in knowledge.agents:
            names = [document.get("name")]
            aliases = document.get("aliases")
            if isinstance(aliases, list):
                names.extend(aliases)
            if name_value in {_contract_identifier(value) for value in names if _contract_identifier(value)}:
                name_matches.append(document)

    def canonical_id(document: dict[str, Any]) -> str:
        return str(document.get("agent_id") or document.get("id") or "").strip()

    exact_ids = {canonical_id(document) for document in exact_matches}
    name_ids = {canonical_id(document) for document in name_matches}
    if len(exact_ids) > 1 or len(name_ids) > 1 or (exact_ids and name_ids and exact_ids != name_ids):
        raise ValueError("conflicting_knowledge_agent_identity")
    if exact_matches:
        return exact_matches[0]
    return name_matches[0] if name_matches else None


def communication_contract_for_node(
    node: dict[str, Any],
    knowledge: KnowledgeRegistryResult,
) -> dict[str, Any] | None:
    document = _agent_document_for_node(node, knowledge)
    contract = document.get("communication_contract") if isinstance(document, dict) else None
    return contract if isinstance(contract, dict) else None


def communication_contract_ref(agent_id: str, contract: dict[str, Any]) -> dict[str, str]:
    """Return a stable, non-authoritative reference to a full MongoDB contract."""

    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    version = contract.get("version")
    return {
        "agent_id": agent_id,
        "version": version if isinstance(version, str) and version else "unversioned",
        "hash": f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}",
    }


def _nested_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _nested_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_keys(child)


def planner_manifest_is_valid(document: dict[str, Any]) -> bool:
    """Return whether an agent's compact planning manifest matches its full contract."""

    agent_id = document.get("agent_id") or document.get("id")
    contract = document.get("communication_contract")
    manifest = document.get("planner_manifest")
    if not isinstance(agent_id, str) or not agent_id.strip() or not isinstance(contract, dict):
        return False
    if not isinstance(manifest, dict) or set(manifest) != PLANNER_MANIFEST_KEYS:
        return False
    if manifest.get("contract_ref") != communication_contract_ref(agent_id.strip(), contract):
        return False
    if manifest.get("contract_version") != contract.get("version"):
        return False
    if manifest.get("planning_schema_path") != "planning.evaluation_request_schema":
        return False
    if not all(isinstance(manifest.get(key), str) and manifest[key].strip() for key in ("contract_version", "purpose")):
        return False
    for key in ("capabilities", "constraints"):
        values = manifest.get(key)
        if not isinstance(values, list) or not values or not all(isinstance(value, str) and value.strip() for value in values):
            return False
    if not isinstance(manifest.get("planning_contract"), dict) or not manifest["planning_contract"]:
        return False
    if FORBIDDEN_MANIFEST_SCHEMA_KEYS.intersection(_nested_keys(manifest)):
        return False
    return len(json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))) <= 2500


def canonicalize_knowledge_agent_identities(
    langgraph_spec: dict[str, Any],
    knowledge: KnowledgeRegistryResult,
) -> dict[str, Any]:
    """Resolve agent name/ID aliases to the authoritative MongoDB agent_id."""

    nodes: list[Any] = []
    id_map: dict[str, str] = {}
    canonical_ids: set[str] = set()
    terminal_ids = {str(node_id) for node_id in langgraph_spec.get("terminal_nodes", [])}
    for raw_node in langgraph_spec.get("nodes", []):
        if not isinstance(raw_node, dict):
            nodes.append(raw_node)
            continue
        node = {key: value for key, value in raw_node.items() if key in PUBLIC_GRAPH_NODE_INPUT_FIELDS}
        old_id = str(node.get("id") or "")
        document = _agent_document_for_node(node, knowledge)
        canonical_id = str(document.get("agent_id") or document.get("id") or "").strip() if isinstance(document, dict) else ""
        if canonical_id:
            if canonical_id in canonical_ids:
                raise ValueError(f"duplicate_knowledge_agent_node:{canonical_id}")
            canonical_ids.add(canonical_id)
            node["id"] = canonical_id
            node["agent_id"] = canonical_id
            contract = document.get("communication_contract")
            if isinstance(contract, dict):
                node["contract_ref"] = communication_contract_ref(canonical_id, contract)
            if old_id:
                id_map[old_id] = canonical_id
        elif old_id in terminal_ids:
            node.update({
                "type": "agent",
                "role": "completion_checker",
                "name": "planning-completion-check",
                "source": "planning_agent",
            })
        nodes.append(node)

    def mapped(value: Any) -> Any:
        return id_map.get(value, value) if isinstance(value, str) else value

    edges = [
        {**edge, "source": mapped(edge.get("source")), "target": mapped(edge.get("target"))}
        if isinstance(edge, dict) else edge
        for edge in langgraph_spec.get("edges", [])
    ]
    return {
        **langgraph_spec,
        "nodes": nodes,
        "edges": edges,
        "entrypoint": mapped(langgraph_spec.get("entrypoint")),
        "representative_agent": mapped(langgraph_spec.get("representative_agent")),
        "terminal_nodes": [mapped(node_id) for node_id in langgraph_spec.get("terminal_nodes", [])],
    }


def validate_subgraph_communication_contracts(
    langgraph_spec: dict[str, Any],
    knowledge: KnowledgeRegistryResult,
) -> None:
    """Require selected agents and their Planning payload to match MongoDB contracts."""

    selected_contracts: dict[str, dict[str, Any]] = {}
    for node in langgraph_spec.get("nodes", []):
        if not isinstance(node, dict):
            continue
        if node.get("role") == "completion_checker" or node.get("name") == "planning-completion-check":
            if _agent_document_for_node(node, knowledge) is not None:
                raise CommunicationContractError(
                    "knowledge_agent_communication_contract_violation",
                    f"Knowledge DB agent {node.get('id')!r} cannot be used as a Planning completion checker.",
                )
            continue
        if node.get("type") != "agent":
            continue
        document = _agent_document_for_node(node, knowledge)
        contract = document.get("communication_contract") if isinstance(document, dict) else None
        if not isinstance(contract, dict):
            raise CommunicationContractError(
                "knowledge_agent_communication_contract_missing",
                f"Knowledge DB agent {node.get('id')!r} has no communication_contract.",
            )
        if contract.get("schema_dialect") != "json-schema-subset-v1":
            raise CommunicationContractError(
                "knowledge_agent_communication_contract_invalid",
                f"Knowledge DB agent {node.get('id')!r} uses an unsupported communication contract dialect.",
            )
        request_schema = contract.get("request_schema")
        response_schema = contract.get("response_schema")
        if not isinstance(request_schema, dict) or not isinstance(response_schema, dict):
            raise CommunicationContractError(
                "knowledge_agent_communication_contract_invalid",
                f"Knowledge DB agent {node.get('id')!r} must define request_schema and response_schema.",
            )
        try:
            validate_contract_schema(request_schema, path=f"{node.get('id')}.request_schema")
            validate_contract_schema(response_schema, path=f"{node.get('id')}.response_schema")
            for schema_name in ("approval_request_schema", "approval_response_schema"):
                optional_schema = contract.get(schema_name)
                if optional_schema is not None:
                    if not isinstance(optional_schema, dict):
                        raise ContractSchemaError(f"{node.get('id')}.{schema_name}: schema must be an object")
                    validate_contract_schema(optional_schema, path=f"{node.get('id')}.{schema_name}")
        except ContractSchemaError as exc:
            raise CommunicationContractError(
                "knowledge_agent_communication_contract_invalid",
                f"Knowledge DB agent {node.get('id')!r} has an invalid communication_contract: {exc}",
            ) from exc
        planning = contract.get("planning")
        evaluation_schema = planning.get("evaluation_request_schema") if isinstance(planning, dict) else None
        if not isinstance(evaluation_schema, dict):
            raise CommunicationContractError(
                "knowledge_agent_communication_contract_invalid",
                f"Knowledge DB agent {node.get('id')!r} has no Planning evaluation schema.",
            )
        try:
            validate_contract_schema(evaluation_schema, path=f"{node.get('id')}.planning.evaluation_request_schema")
        except ContractSchemaError as exc:
            raise CommunicationContractError(
                "knowledge_agent_communication_contract_invalid",
                f"Knowledge DB agent {node.get('id')!r} has an invalid communication_contract: {exc}",
            ) from exc
        if not planner_manifest_is_valid(document):
            raise CommunicationContractError(
                "knowledge_agent_planner_manifest_invalid",
                f"Knowledge DB agent {node.get('id')!r} has no valid planner_manifest.",
            )
        selected_contracts[_contract_identifier(node.get("id"))] = contract

    consumer_id = _contract_identifier(langgraph_spec.get("entrypoint"))
    contract = selected_contracts.get(consumer_id)
    if contract is None:
        raise CommunicationContractError(
            "knowledge_agent_communication_contract_missing",
            "The entrypoint agent has no Knowledge DB communication_contract.",
        )
    planning = contract.get("planning")
    schema = planning.get("evaluation_request_schema") if isinstance(planning, dict) else None
    if not isinstance(schema, dict):
        raise CommunicationContractError(
            "knowledge_agent_communication_contract_invalid",
            "The entrypoint agent contract has no planning.evaluation_request_schema.",
        )
    try:
        validate_contract_payload(
            langgraph_spec.get("evaluation_request"),
            schema,
            path="$.langgraph_spec.evaluation_request",
        )
    except ContractSchemaError as exc:
        raise CommunicationContractError(
            "knowledge_agent_communication_contract_invalid",
            f"The entrypoint agent has an invalid MongoDB communication_contract: {exc}",
        ) from exc
    except ValueError as exc:
        raise CommunicationContractError(
            "knowledge_agent_communication_contract_violation",
            f"Planning output violates the entrypoint agent's MongoDB communication_contract: {exc}",
        ) from exc


def _log(message: str) -> None:
    print(f"knowledge-registry: {message}", file=sys.stderr)


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


def _int_env(name: str, default: int, *, minimum: int = 1, maximum: int = 20) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return min(max(value, minimum), maximum)


def _float_env(name: str, default: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    raw_value = os.getenv(name, str(default))
    try:
        value = float(raw_value)
    except ValueError:
        return default
    return min(max(value, minimum), maximum)


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
        return {str(k): _sanitize(v) for index, (k, v) in enumerate(value.items()) if index < MAX_DICT_ITEMS}
    return str(value)


def _sanitize_contract(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return [_sanitize_contract(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_contract(item) for key, item in value.items()}
    return str(value)


def _sanitize_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = [_sanitize(document) for document in documents]
    for source, document in zip(documents, sanitized):
        if isinstance(document, dict):
            document.pop("_match_score", None)
            if isinstance(source.get("communication_contract"), dict):
                document["communication_contract"] = _sanitize_contract(source["communication_contract"])
    return sanitized


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in TOKEN_PATTERN.findall(text):
        token = raw_token.lower()
        if token in STOPWORDS:
            continue
        if len(token) >= 3 or raw_token.isupper():
            tokens.append(token)
    return tokens


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _query_keywords(subtask: str, intent: str | None = None) -> list[str]:
    return _unique(_tokens(" ".join(part for part in [subtask, intent or ""] if part)))


def _candidate_text(document: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in SEARCH_FIELD_NAMES:
        value = document.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif isinstance(value, dict):
            parts.extend(str(item) for item in value.values())
    runtime = document.get("runtime")
    if isinstance(runtime, dict):
        parts.extend(str(value) for value in runtime.values() if isinstance(value, str))
    return " ".join(parts)


def _matched_keywords(query_keywords: list[str], document: dict[str, Any]) -> list[str]:
    candidate = _candidate_text(document).lower()
    candidate_tokens = set(_tokens(candidate))
    matches: list[str] = []
    for keyword in query_keywords:
        key = keyword.lower()
        if key in candidate_tokens or key in candidate:
            matches.append(key)
    return _unique(matches)


def _similarity_score(query: str, document: dict[str, Any], query_keywords: list[str] | None = None) -> float:
    query_tokens = set(query_keywords or _tokens(query))
    candidate_tokens = set(_tokens(_candidate_text(document)))
    if not query_tokens or not candidate_tokens:
        return 0.0
    common = query_tokens.intersection(candidate_tokens)
    overlap = len(common) / len(query_tokens)
    coverage = len(common) / len(candidate_tokens)
    sequence = SequenceMatcher(None, " ".join(sorted(query_tokens)), " ".join(sorted(candidate_tokens))).ratio()
    score = (overlap * 0.68) + (coverage * 0.17) + (sequence * 0.15)
    text_score = document.get("score")
    if isinstance(text_score, int | float):
        score = min(1.0, score + min(float(text_score), 10.0) / 100.0)
    return round(score, 4)


def _annotate_and_filter(
    documents: list[dict[str, Any]],
    query: str,
    limit: int,
    threshold: float,
    query_keywords: list[str] | None = None,
) -> list[dict[str, Any]]:
    keywords = query_keywords or _query_keywords(query)
    scored: list[tuple[float, dict[str, Any]]] = []
    seen: set[str] = set()
    for document in documents:
        doc_id = str(document.get("_id") or document.get("id") or document.get("agent_id") or document.get("tool_id") or id(document))
        if doc_id in seen:
            continue
        seen.add(doc_id)
        matches = _matched_keywords(keywords, document)
        score = _similarity_score(query, document, keywords)
        if matches:
            score = min(1.0, score + min(len(matches), 6) * 0.03)
        if score < threshold:
            continue
        annotated = dict(document)
        annotated["_match_score"] = round(score, 4)
        annotated["_matched_keywords"] = matches
        annotated["_match_strategy"] = "text_keyword_regex_fuzzy"
        scored.append((score, annotated))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [document for _, document in scored[:limit]]


def _regex_query(query_keywords: list[str]) -> dict[str, Any]:
    words = [word for word in query_keywords if word]
    if not words:
        return {}
    pattern = "|".join(re.escape(word) for word in words[:20])
    return {"$or": [{field: {"$regex": pattern, "$options": "i"}} for field in SEARCH_FIELD_NAMES]}


def _active_registry_query(query: dict[str, Any]) -> dict[str, Any]:
    active_filter = {"$or": [{"status": {"$in": ACTIVE_STATUS_VALUES}}, {"status": {"$exists": False}}]}
    if not query:
        return active_filter
    return {"$and": [query, active_filter]}


def _find_text(collection: Any, query: str, limit: int) -> list[dict[str, Any]]:
    cursor = collection.find(
        _active_registry_query({"$text": {"$search": query}}),
        {"score": {"$meta": "textScore"}},
    ).sort([("score", {"$meta": "textScore"})])
    return list(cursor.limit(limit * 4))


def _find_fallback(collection: Any, query_keywords: list[str], limit: int) -> list[dict[str, Any]]:
    regex_query = _regex_query(query_keywords)
    if not regex_query:
        return []
    return list(collection.find(_active_registry_query(regex_query)).limit(limit * 4))


def _find_candidates(collection: Any, query: str, query_keywords: list[str], limit: int, threshold: float) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    try:
        documents.extend(_find_text(collection, query, limit))
    except Exception as exc:  # noqa: BLE001 - missing text index should fall back.
        _log(f"text search unavailable; using keyword fallback lookup ({type(exc).__name__})")
    try:
        documents.extend(_find_fallback(collection, query_keywords, limit))
    except Exception as exc:  # noqa: BLE001 - regex lookup should not break planning.
        _log(f"keyword fallback lookup failed ({type(exc).__name__})")
    return _annotate_and_filter(documents, query, limit, threshold, query_keywords)


def _find_active_catalog(collection: Any, limit: int) -> list[dict[str, Any]]:
    documents = list(collection.find(_active_registry_query({})).limit(limit))
    return [
        {
            **document,
            "_matched_keywords": [],
            "_match_strategy": "active_catalog_fallback",
        }
        for document in documents
    ]


def find_knowledge_for_subtask(subtask: str, *, intent: str | None = None) -> KnowledgeRegistryResult:
    """Find lower-level agents and tools for one subtask from MongoDB Knowledge DB."""

    _load_env_file()
    query_keywords = _query_keywords(subtask, intent)
    query = " ".join(query_keywords) or " ".join(part for part in [subtask, intent or ""] if part)
    uri = os.getenv("KNOWLEDGE_MONGODB_URI", "").strip()
    if not uri:
        return KnowledgeRegistryResult(used=False, agents=[], tools=[], available=False, error="knowledge_mongodb_uri_unset", query_keywords=query_keywords)

    try:
        from pymongo import MongoClient
    except ImportError:
        _log("pymongo is not installed; Knowledge DB is unavailable")
        return KnowledgeRegistryResult(used=False, agents=[], tools=[], available=False, error="pymongo_not_installed", query_keywords=query_keywords)

    database_name = os.getenv("KNOWLEDGE_DATABASE", DEFAULT_DATABASE).strip() or DEFAULT_DATABASE
    agents_collection = os.getenv("KNOWLEDGE_AGENTS_COLLECTION", DEFAULT_AGENTS_COLLECTION).strip() or DEFAULT_AGENTS_COLLECTION
    tools_collection = os.getenv("KNOWLEDGE_TOOLS_COLLECTION", DEFAULT_TOOLS_COLLECTION).strip() or DEFAULT_TOOLS_COLLECTION
    limit = _int_env("KNOWLEDGE_QUERY_LIMIT", DEFAULT_QUERY_LIMIT)
    timeout_ms = _int_env("KNOWLEDGE_TIMEOUT_MS", DEFAULT_TIMEOUT_MS, minimum=100, maximum=10000)
    threshold = _float_env("KNOWLEDGE_SIMILARITY_THRESHOLD", DEFAULT_SIMILARITY_THRESHOLD)

    client = None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        client.admin.command("ping")
        database = client[database_name]
        agents = _find_candidates(database[agents_collection], query, query_keywords, limit, threshold)
        tools = _find_candidates(database[tools_collection], query, query_keywords, limit, threshold)
        if not agents:
            agents = _find_active_catalog(database[agents_collection], limit)
        if not tools:
            tools = _find_active_catalog(database[tools_collection], limit)
    except Exception as exc:  # noqa: BLE001 - caller must block without local fallback.
        error = f"{type(exc).__name__}: {exc}"
        _log(f"Knowledge DB lookup failed ({error})")
        return KnowledgeRegistryResult(used=False, agents=[], tools=[], available=False, error=error, query_keywords=query_keywords)
    finally:
        if client is not None:
            client.close()

    sanitized_agents = _sanitize_documents(agents)
    sanitized_tools = _sanitize_documents(tools)
    return KnowledgeRegistryResult(
        used=bool(sanitized_agents or sanitized_tools),
        agents=sanitized_agents,
        tools=sanitized_tools,
        available=True,
        error=None,
        query_keywords=query_keywords,
    )


def find_agent_communication_contract(agent_id: str) -> dict[str, Any] | None:
    """Read one active agent's exact communication contract from Knowledge DB."""

    _load_env_file()
    uri = os.getenv("KNOWLEDGE_MONGODB_URI", "").strip()
    if not uri or not isinstance(agent_id, str) or not agent_id.strip():
        return None
    try:
        from pymongo import MongoClient
    except ImportError:
        return None
    database_name = os.getenv("KNOWLEDGE_DATABASE", DEFAULT_DATABASE).strip() or DEFAULT_DATABASE
    collection_name = os.getenv("KNOWLEDGE_AGENTS_COLLECTION", DEFAULT_AGENTS_COLLECTION).strip() or DEFAULT_AGENTS_COLLECTION
    timeout_ms = _int_env("KNOWLEDGE_TIMEOUT_MS", DEFAULT_TIMEOUT_MS, minimum=100, maximum=10000)
    client = None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        document = client[database_name][collection_name].find_one(
            {"agent_id": agent_id.strip(), "$or": [{"status": {"$in": ACTIVE_STATUS_VALUES}}, {"status": {"$exists": False}}]},
            {"_id": 0, "communication_contract": 1},
        )
    except Exception as exc:  # noqa: BLE001 - approval must fail closed when registry is unavailable.
        _log(f"Agent communication contract lookup failed ({type(exc).__name__})")
        return None
    finally:
        if client is not None:
            client.close()
    contract = document.get("communication_contract") if isinstance(document, dict) else None
    return _sanitize_contract(contract) if isinstance(contract, dict) else None


def resolve_agent_communication_contract(agent_id: str, contract_ref: dict[str, Any]) -> dict[str, Any]:
    """Resolve a live MongoDB contract only when its authoritative reference matches exactly."""

    if not isinstance(agent_id, str) or not agent_id.strip() or not isinstance(contract_ref, dict):
        raise ValueError("invalid_agent_communication_contract_ref")
    canonical_id = agent_id.strip()
    contract = find_agent_communication_contract(canonical_id)
    if not isinstance(contract, dict):
        raise ValueError(f"agent_communication_contract_missing:{canonical_id}")
    if contract_ref != communication_contract_ref(canonical_id, contract):
        raise ValueError(f"agent_communication_contract_ref_mismatch:{canonical_id}")
    return contract
