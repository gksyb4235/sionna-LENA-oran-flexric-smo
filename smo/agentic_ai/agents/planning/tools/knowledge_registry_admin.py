"""Seed and maintain the MongoDB Knowledge Registry for Planning Agent."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from registry_catalog import registered_agent_documents, registered_tool_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = "knowledge"
DEFAULT_AGENTS_COLLECTION = "agents"
DEFAULT_TOOLS_COLLECTION = "tools"


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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _upsert_documents(collection: Any, documents: list[dict[str, Any]], id_key: str) -> list[str]:
    updated_ids: list[str] = []
    timestamp = _now()
    for raw_document in documents:
        document = dict(raw_document)
        document.setdefault("schema_version", "registry-v1")
        document["updated_at"] = timestamp
        doc_id = str(document.get(id_key) or document.get("id") or "").strip()
        if not doc_id:
            raise ValueError(f"Registry document missing {id_key}/id: {document}")
        document[id_key] = doc_id
        document.setdefault("id", doc_id)
        collection.update_one(
            {id_key: doc_id},
            {"$set": document, "$setOnInsert": {"created_at": timestamp}},
            upsert=True,
        )
        updated_ids.append(doc_id)
    return updated_ids


def _create_indexes(agents: Any, tools: Any) -> None:
    for collection, index_name in ((agents, "agent_registry_text"), (tools, "tool_registry_text")):
        try:
            collection.drop_index(index_name)
        except Exception:
            pass

    agents.create_index("agent_id", unique=True)
    agents.create_index("id")
    agents.create_index("status")
    agents.create_index([
        ("name", "text"),
        ("agent_id", "text"),
        ("description", "text"),
        ("capabilities", "text"),
        ("can_do", "text"),
        ("tags", "text"),
        ("keywords", "text"),
        ("aliases", "text"),
        ("domain", "text"),
    ], name="agent_registry_text")
    tools.create_index("tool_id", unique=True)
    tools.create_index("id")
    tools.create_index("status")
    tools.create_index([
        ("name", "text"),
        ("tool_id", "text"),
        ("description", "text"),
        ("capabilities", "text"),
        ("can_do", "text"),
        ("operations", "text"),
        ("tags", "text"),
        ("keywords", "text"),
        ("aliases", "text"),
        ("domain", "text"),
    ], name="tool_registry_text")


def seed_registry(*, uri: str, database_name: str, agents_collection: str, tools_collection: str, include_planned: bool) -> dict[str, Any]:
    from pymongo import MongoClient

    agent_docs = registered_agent_documents(include_planned=include_planned)
    tool_docs = registered_tool_documents(include_planned=include_planned)
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        database = client[database_name]
        agents = database[agents_collection]
        tools = database[tools_collection]
        _create_indexes(agents, tools)
        agent_ids = _upsert_documents(agents, agent_docs, "agent_id")
        tool_ids = _upsert_documents(tools, tool_docs, "tool_id")
        return {
            "database": database_name,
            "agents_collection": agents_collection,
            "tools_collection": tools_collection,
            "agent_count": len(agent_ids),
            "tool_count": len(tool_ids),
            "agent_ids": agent_ids,
            "tool_ids": tool_ids,
        }
    finally:
        client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed the Planning Agent Knowledge Registry MongoDB collections.")
    parser.add_argument("--uri", default=None, help="MongoDB URI. Defaults to KNOWLEDGE_MONGODB_URI from .env.")
    parser.add_argument("--database", default=None, help="MongoDB database name.")
    parser.add_argument("--agents-collection", default=None, help="Agents collection name.")
    parser.add_argument("--tools-collection", default=None, help="Tools collection name.")
    parser.add_argument("--include-planned", action="store_true", help="Also seed planned/inactive registry documents.")
    parser.add_argument("--dry-run", action="store_true", help="Print documents that would be seeded without connecting to MongoDB.")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Print machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_env_file()
    args = build_parser().parse_args(argv)
    database_name = args.database or os.getenv("KNOWLEDGE_DATABASE", DEFAULT_DATABASE)
    agents_collection = args.agents_collection or os.getenv("KNOWLEDGE_AGENTS_COLLECTION", DEFAULT_AGENTS_COLLECTION)
    tools_collection = args.tools_collection or os.getenv("KNOWLEDGE_TOOLS_COLLECTION", DEFAULT_TOOLS_COLLECTION)
    if args.dry_run:
        payload = {
            "database": database_name,
            "agents_collection": agents_collection,
            "tools_collection": tools_collection,
            "agents": registered_agent_documents(include_planned=args.include_planned),
            "tools": registered_tool_documents(include_planned=args.include_planned),
        }
    else:
        uri = args.uri or os.getenv("KNOWLEDGE_MONGODB_URI", "").strip()
        if not uri:
            raise SystemExit("KNOWLEDGE_MONGODB_URI is not set. Pass --uri or configure planning/.env.")
        payload = seed_registry(
            uri=uri,
            database_name=database_name,
            agents_collection=agents_collection,
            tools_collection=tools_collection,
            include_planned=args.include_planned,
        )
    if args.json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Knowledge registry ready: {payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
