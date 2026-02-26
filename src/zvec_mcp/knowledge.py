"""Knowledge base manager — chunked document storage and retrieval (RAG)."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
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


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split *text* into overlapping chunks by character count.

    Tries to break on sentence boundaries (`.`, `!`, `?`, `\\n`) when
    possible so that chunks are more coherent.
    """
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to find a sentence boundary near the end of the window
        best = -1
        for sep in (".\n", "\n\n", "\n", ". ", "! ", "? "):
            idx = text.rfind(sep, start + size // 2, end)
            if idx != -1:
                best = idx + len(sep)
                break
        if best == -1:
            # Fall back to a space
            idx = text.rfind(" ", start + size // 2, end)
            best = idx + 1 if idx != -1 else end

        chunks.append(text[start:best])
        start = best - overlap
        if start < 0:
            start = 0

    return chunks


def _make_chunk_id(source: str, idx: int) -> str:
    """Deterministic chunk ID from source name + index."""
    h = hashlib.sha256(f"{source}:{idx}".encode()).hexdigest()[:16]
    return f"kb_{h}"


class KnowledgeBase:
    """Manages a zvec collection for chunked document retrieval."""

    def __init__(self, cfg: Config, embedder: _Embedder) -> None:
        self._cfg = cfg
        self._emb = embedder
        self._col: zvec.Collection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _schema(self) -> CollectionSchema:
        return CollectionSchema(
            name=self._cfg.knowledge_collection,
            fields=[
                FieldSchema("source", DataType.STRING, nullable=False, index_param=InvertIndexParam()),
                FieldSchema("chunk_idx", DataType.INT32, nullable=False),
                FieldSchema("text", DataType.STRING, nullable=False),
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
        """Open or create the knowledge collection."""
        self._cfg.ensure_dirs()
        path = str(self._cfg.knowledge_path)
        if self._cfg.knowledge_path.exists():
            self._col = zvec.open(path)
            logger.info("Opened knowledge base at %s", path)
        else:
            self._col = zvec.create_and_open(path, self._schema())
            logger.info("Created knowledge base at %s", path)

    @property
    def col(self) -> zvec.Collection:
        if self._col is None:
            self.open()
        assert self._col is not None
        return self._col

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(
        self,
        text: str,
        source: str = "manual",
        *,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Chunk, embed, and store *text*.

        Returns the number of chunks stored.
        """
        chunks = _chunk_text(text, self._cfg.chunk_size, self._cfg.chunk_overlap)
        now = int(time.time())

        docs: list[Doc] = []
        for i, chunk in enumerate(chunks):
            vec = self._emb.embed(chunk)
            docs.append(
                Doc(
                    id=_make_chunk_id(source, i),
                    fields={
                        "source": source,
                        "chunk_idx": i,
                        "text": chunk,
                        "created_at": now,
                    },
                    vectors={"embedding": vec},
                )
            )

        self.col.upsert(docs)
        logger.info("Ingested %d chunks from source=%s", len(docs), source)
        return len(docs)

    def ingest_file(self, path: str) -> int:
        """Read a file and ingest its contents."""
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {p}")
        text = p.read_text(encoding="utf-8", errors="replace")
        return self.ingest(text, source=str(p))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, *, topk: int = 5) -> list[dict[str, Any]]:
        """Semantic search over the knowledge base.

        Returns list of dicts with keys: id, score, source, chunk_idx, text.
        """
        vec = self._emb.embed(query)
        results = self.col.query(
            vectors=VectorQuery(field_name="embedding", vector=vec),
            topk=topk,
            output_fields=["source", "chunk_idx", "text", "created_at"],
        )
        out: list[dict[str, Any]] = []
        for doc in results:
            out.append({
                "id": doc.id,
                "score": doc.score,
                "source": doc.field("source") if doc.has_field("source") else None,
                "chunk_idx": doc.field("chunk_idx") if doc.has_field("chunk_idx") else None,
                "text": doc.field("text") if doc.has_field("text") else None,
            })
        return out

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def delete_source(self, source: str) -> None:
        """Delete all chunks from a given source."""
        self.col.delete_by_filter(f"source == '{source}'")
        logger.info("Deleted chunks for source=%s", source)

    def stats(self) -> dict[str, Any]:
        s = self.col.stats
        return {
            "collection": self._cfg.knowledge_collection,
            "path": str(self._cfg.knowledge_path),
            "doc_count": s.doc_count,
            "embedding_dim": self._emb.dim,
        }
