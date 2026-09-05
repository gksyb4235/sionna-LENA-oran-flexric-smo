"""Tool package for the Planning Agent."""

from tools.knowledge_registry import KnowledgeRegistryResult, find_knowledge_for_subtask


def mock_execute_subgraph(*args, **kwargs):
    """Load the optional LangGraph runtime only when execution is requested."""
    from tools.subgraph_runtime import mock_execute_subgraph as execute

    return execute(*args, **kwargs)

__all__ = ["KnowledgeRegistryResult", "find_knowledge_for_subtask", "mock_execute_subgraph"]
