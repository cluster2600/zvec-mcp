"""FastMCP server — exposes RAG knowledge base + memory tools for Claude Code."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import zvec
from mcp.server.fastmcp import FastMCP

from zvec_mcp.config import Config
from zvec_mcp.embeddings import get_embedder
from zvec_mcp.knowledge import KnowledgeBase
from zvec_mcp.memory import MemoryStore

# ---------------------------------------------------------------------------
# Logging to stderr (stdout is JSON-RPC for MCP)
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [zvec-mcp] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals (initialised in main / on first tool call)
# ---------------------------------------------------------------------------
cfg = Config()
_zvec_init = False
_kb: KnowledgeBase | None = None
_mem: MemoryStore | None = None

mcp = FastMCP(
    "zvec-mcp",
    instructions=(
        "You have access to a local vector database powered by zvec. "
        "Use the knowledge_* tools to manage a RAG knowledge base (ingest "
        "documents, search for relevant context). Use the memory_* tools to "
        "store and recall facts, preferences, and observations across "
        "conversations."
    ),
)


def _ensure_init() -> tuple[KnowledgeBase, MemoryStore]:
    """Lazy-init zvec engine, knowledge base, and memory store."""
    global _zvec_init, _kb, _mem
    if not _zvec_init:
        zvec.init()
        _zvec_init = True
    if _kb is None:
        emb = get_embedder(cfg)
        _kb = KnowledgeBase(cfg, emb)
    if _mem is None:
        emb = get_embedder(cfg)
        _mem = MemoryStore(cfg, emb)
    return _kb, _mem


# ===================================================================
# Knowledge Base (RAG) tools
# ===================================================================


@mcp.tool()
def knowledge_ingest(text: str, source: str = "manual") -> str:
    """Ingest text into the knowledge base for later RAG retrieval.

    The text is split into overlapping chunks, embedded, and stored in
    a local zvec vector collection. Use this to add documentation,
    articles, notes, or any reference material.

    Args:
        text: The full text content to ingest.
        source: A label identifying where this text came from
                (e.g. a file path, URL, or descriptive name).
    """
    kb, _ = _ensure_init()
    n = kb.ingest(text, source=source)
    return json.dumps({"status": "ok", "chunks_stored": n, "source": source})


@mcp.tool()
def knowledge_ingest_file(path: str) -> str:
    """Read a file from disk and ingest its contents into the knowledge base.

    The file is read as UTF-8 text, chunked, embedded, and stored for
    later semantic search.

    Args:
        path: Absolute or relative path to the file to ingest.
    """
    kb, _ = _ensure_init()
    try:
        n = kb.ingest_file(path)
        return json.dumps({"status": "ok", "chunks_stored": n, "source": path})
    except FileNotFoundError as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def knowledge_search(query: str, topk: int = 5) -> str:
    """Search the knowledge base for text relevant to a query.

    Returns the most semantically similar chunks. Use this to retrieve
    context before answering questions about ingested documents.

    Args:
        query: Natural language search query.
        topk: Maximum number of results to return (default 5).
    """
    kb, _ = _ensure_init()
    results = kb.search(query, topk=topk)
    return json.dumps({"results": results, "count": len(results)})


@mcp.tool()
def knowledge_delete_source(source: str) -> str:
    """Delete all chunks from a specific source in the knowledge base.

    Args:
        source: The source label used during ingestion.
    """
    kb, _ = _ensure_init()
    kb.delete_source(source)
    return json.dumps({"status": "ok", "deleted_source": source})


@mcp.tool()
def knowledge_stats() -> str:
    """Return statistics about the knowledge base collection."""
    kb, _ = _ensure_init()
    return json.dumps(kb.stats())


# ===================================================================
# Memory tools
# ===================================================================


@mcp.tool()
def memory_remember(text: str, category: str = "general") -> str:
    """Store a fact, observation, or preference in long-term memory.

    Memories are embedded and stored in a vector collection so they can
    be recalled later by semantic similarity. Duplicate text is
    deduplicated automatically.

    Args:
        text: The fact or observation to remember.
        category: An optional category tag (e.g. "preference",
                  "project", "person", "decision").
    """
    _, mem = _ensure_init()
    mid = mem.remember(text, category=category)
    return json.dumps({"status": "ok", "memory_id": mid, "category": category})


@mcp.tool()
def memory_recall(query: str, topk: int = 5, category: str | None = None) -> str:
    """Recall memories that are semantically similar to a query.

    Use this to retrieve relevant context from past conversations,
    stored facts, or user preferences.

    Args:
        query: Natural language query to match against memories.
        topk: Maximum number of results (default 5).
        category: Optional category filter.
    """
    _, mem = _ensure_init()
    results = mem.recall(query, topk=topk, category=category)
    return json.dumps({"results": results, "count": len(results)})


@mcp.tool()
def memory_forget(memory_id: str) -> str:
    """Delete a specific memory by its ID.

    Args:
        memory_id: The ID returned by memory_remember.
    """
    _, mem = _ensure_init()
    ok = mem.forget(memory_id)
    return json.dumps({"status": "ok" if ok else "not_found", "memory_id": memory_id})


@mcp.tool()
def memory_forget_category(category: str) -> str:
    """Delete all memories in a category.

    Args:
        category: The category to clear.
    """
    _, mem = _ensure_init()
    mem.forget_category(category)
    return json.dumps({"status": "ok", "deleted_category": category})


@mcp.tool()
def memory_stats() -> str:
    """Return statistics about the memory store."""
    _, mem = _ensure_init()
    return json.dumps(mem.stats())


# ===================================================================
# Combined status
# ===================================================================


@mcp.tool()
def zvec_status() -> str:
    """Return overall status of the zvec-mcp server.

    Shows configuration, embedding backend, and collection stats.
    """
    kb, mem = _ensure_init()
    return json.dumps({
        "data_dir": str(cfg.data_dir),
        "embedding_backend": cfg.embedding_backend,
        "embedding_dim": cfg.embedding_dim,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "knowledge": kb.stats(),
        "memory": mem.stats(),
    })


# ===================================================================
# Entry point
# ===================================================================


def main() -> None:
    """Run the MCP server over stdio."""
    logger.info("Starting zvec-mcp server (data_dir=%s)", cfg.data_dir)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
