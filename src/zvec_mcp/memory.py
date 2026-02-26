"""Memory manager — semantic storage and recall of facts and observations."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

import zvec
from zvec import (
    CollectionSchema,
    DataType,
    Doc,
    FieldSchema,
    VectorQuery,
    VectorSchema,
)
from zvec.model.param import FlatIndexParam, InvertIndexParam

from zvec_mcp.config import Config
from zvec_mcp.embeddings import _Embedder

logger = logging.getLogger(__name__)


def _memory_id(text: str) -> str:
    """Content-addressed ID so duplicate memories are deduplicated."""
    h = hashlib.sha256(text.encode()).hexdigest()[:16]
    return f"mem_{h}"


class MemoryStore:
    """Manages a zvec collection for conversational / long-term memory."""

    def __init__(self, cfg: Config, embedder: _Embedder) -> None:
        self._cfg = cfg
        self._emb = embedder
        self._col: zvec.Collection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _schema(self) -> CollectionSchema:
        return CollectionSchema(
            name=self._cfg.memory_collection,
            fields=[
                FieldSchema("text", DataType.STRING, nullable=False),
                FieldSchema("category", DataType.STRING, nullable=True, index_param=InvertIndexParam()),
                FieldSchema("created_at", DataType.INT64, nullable=False),
            ],
            vectors=[
                VectorSchema(
                    "embedding",
                    DataType.VECTOR_FP32,
                    dimension=self._emb.dim,
                    index_param=FlatIndexParam(),
                ),
            ],
        )

    def open(self) -> None:
        self._cfg.ensure_dirs()
        path = str(self._cfg.memory_path)
        if self._cfg.memory_path.exists():
            self._col = zvec.open(path)
            logger.info("Opened memory store at %s", path)
        else:
            self._col = zvec.create_and_open(path, self._schema())
            logger.info("Created memory store at %s", path)

    @property
    def col(self) -> zvec.Collection:
        if self._col is None:
            self.open()
        assert self._col is not None
        return self._col

    # ------------------------------------------------------------------
    # Store / Recall / Forget
    # ------------------------------------------------------------------

    def remember(
        self,
        text: str,
        *,
        category: str = "general",
    ) -> str:
        """Store a fact or observation. Returns the memory ID."""
        mid = _memory_id(text)
        vec = self._emb.embed(text)
        doc = Doc(
            id=mid,
            fields={
                "text": text,
                "category": category,
                "created_at": int(time.time()),
            },
            vectors={"embedding": vec},
        )
        self.col.upsert([doc])
        logger.info("Remembered [%s] category=%s: %s", mid, category, text[:80])
        return mid

    def recall(
        self,
        query: str,
        *,
        topk: int = 5,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over memories.

        Returns list of dicts: id, score, text, category, created_at.
        """
        vec = self._emb.embed(query)
        filt = f"category = '{category}'" if category else None
        results = self.col.query(
            vectors=VectorQuery(field_name="embedding", vector=vec),
            topk=topk,
            filter=filt,
            output_fields=["text", "category", "created_at"],
        )
        out: list[dict[str, Any]] = []
        for doc in results:
            out.append({
                "id": doc.id,
                "score": doc.score,
                "text": doc.field("text") if doc.has_field("text") else None,
                "category": doc.field("category") if doc.has_field("category") else None,
                "created_at": doc.field("created_at") if doc.has_field("created_at") else None,
            })
        return out

    def forget(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        status = self.col.delete(memory_id)
        ok = status.ok() if hasattr(status, "ok") else True
        if ok:
            logger.info("Forgot memory %s", memory_id)
        return ok

    def forget_category(self, category: str) -> None:
        """Delete all memories in a category."""
        self.col.delete_by_filter(f"category = '{category}'")
        logger.info("Forgot all memories in category=%s", category)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        s = self.col.stats
        return {
            "collection": self._cfg.memory_collection,
            "path": str(self._cfg.memory_path),
            "doc_count": s.doc_count,
            "embedding_dim": self._emb.dim,
        }
