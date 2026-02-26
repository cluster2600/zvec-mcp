"""Embedding wrapper — lazy-loads the chosen backend on first use."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from zvec_mcp.config import Config

logger = logging.getLogger(__name__)

# Singleton so we only load the model once per process
_embedder: _Embedder | None = None


class _Embedder:
    """Thin wrapper that dispatches to zvec's embedding functions."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._fn: object | None = None  # lazy

    def _load(self) -> None:
        if self._fn is not None:
            return

        if self._cfg.embedding_backend == "openai":
            from zvec.extension import OpenAIDenseEmbedding

            self._fn = OpenAIDenseEmbedding(
                model=self._cfg.openai_model,
                dimension=self._cfg.openai_dimension,
                api_key=self._cfg.openai_api_key or None,
            )
            logger.info(
                "Loaded OpenAI embedding: model=%s dim=%d",
                self._cfg.openai_model,
                self._cfg.openai_dimension,
            )
        else:
            from zvec.extension import DefaultLocalDenseEmbedding

            self._fn = DefaultLocalDenseEmbedding()
            logger.info("Loaded local sentence-transformers embedding (dim=384)")

    def embed(self, text: str) -> list[float]:
        """Embed a single string → list[float]."""
        self._load()
        vec = self._fn.embed(text)  # type: ignore[union-attr]
        if isinstance(vec, np.ndarray):
            return vec.tolist()
        return list(vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings."""
        return [self.embed(t) for t in texts]

    @property
    def dim(self) -> int:
        return self._cfg.embedding_dim


def get_embedder(cfg: Config) -> _Embedder:
    """Return (or create) the process-global embedder."""
    global _embedder
    if _embedder is None:
        _embedder = _Embedder(cfg)
    return _embedder
