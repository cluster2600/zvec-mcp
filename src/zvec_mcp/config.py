"""Configuration for zvec-mcp server."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Server configuration, driven by environment variables."""

    # Root data directory for all zvec collections
    data_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("ZVEC_MCP_DATA_DIR", os.path.expanduser("~/.zvec-mcp"))
    ))

    # Embedding model
    # "local" => sentence-transformers all-MiniLM-L6-v2 (384-dim, free, offline)
    # "openai" => OpenAI text-embedding-3-small (1536-dim, requires API key)
    embedding_backend: str = field(default_factory=lambda: os.environ.get(
        "ZVEC_MCP_EMBEDDING", "local"
    ))

    # OpenAI settings (only when embedding_backend == "openai")
    openai_api_key: str = field(default_factory=lambda: os.environ.get(
        "OPENAI_API_KEY", ""
    ))
    openai_model: str = field(default_factory=lambda: os.environ.get(
        "ZVEC_MCP_OPENAI_MODEL", "text-embedding-3-small"
    ))
    openai_dimension: int = field(default_factory=lambda: int(os.environ.get(
        "ZVEC_MCP_OPENAI_DIM", "1536"
    )))

    # Chunking
    chunk_size: int = field(default_factory=lambda: int(os.environ.get(
        "ZVEC_MCP_CHUNK_SIZE", "512"
    )))
    chunk_overlap: int = field(default_factory=lambda: int(os.environ.get(
        "ZVEC_MCP_CHUNK_OVERLAP", "64"
    )))

    # Collection names
    knowledge_collection: str = "knowledge"
    memory_collection: str = "memory"

    @property
    def embedding_dim(self) -> int:
        if self.embedding_backend == "openai":
            return self.openai_dimension
        return 384  # all-MiniLM-L6-v2

    @property
    def knowledge_path(self) -> Path:
        return self.data_dir / self.knowledge_collection

    @property
    def memory_path(self) -> Path:
        return self.data_dir / self.memory_collection

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
