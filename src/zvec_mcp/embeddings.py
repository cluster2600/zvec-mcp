"""Embedding wrapper — lazy-loads the chosen backend on first use."""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from zvec_mcp.config import Config

logger = logging.getLogger(__name__)

# Singleton so we only load the model once per process
_embedder: _Embedder | None = None


class _HttpEmbedder:
    """Calls any OpenAI-compatible /v1/embeddings endpoint (LM Studio, Ollama, vLLM, …)."""

    def __init__(self, url: str, model: str, api_key: str = "") -> None:
        self._url = url
        self._model = model
        self._api_key = api_key

    def embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self._model, "input": text}).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(self._url, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            body: dict[str, Any] = json.loads(resp.read())
        return body["data"][0]["embedding"]


class _Embedder:
    """Thin wrapper that dispatches to the configured embedding backend."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._fn: object | None = None  # lazy

    def _load(self) -> None:
        if self._fn is not None:
            return

        backend = self._cfg.embedding_backend

        if backend == "openai":
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
        elif backend == "http":
            self._fn = _HttpEmbedder(
                url=self._cfg.http_url,
                model=self._cfg.http_model,
                api_key=self._cfg.http_api_key,
            )
            logger.info(
                "Loaded HTTP embedding: url=%s model=%s dim=%d",
                self._cfg.http_url,
                self._cfg.http_model,
                self._cfg.http_dimension,
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
