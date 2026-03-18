# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Gemini Embedding 2 provider using the official google-genai SDK."""

import math
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

import logging

try:
    import anyio
    _ANYIO_AVAILABLE = True
except ImportError:
    _ANYIO_AVAILABLE = False

from openviking.models.embedder.base import (
    DenseEmbedderBase,
    EmbedResult,
    truncate_and_normalize,
)

logger = logging.getLogger("gemini_embedders")

_TEXT_BATCH_SIZE = 100

# Keep for backward-compat with existing unit tests that import it
_GEMINI_INPUT_TOKEN_LIMIT = 8192  # gemini-embedding-2-preview hard limit

# Per-model token limits (Google API hard limits, from official docs)
_MODEL_TOKEN_LIMITS: Dict[str, int] = {
    "gemini-embedding-2-preview": 8192,
    "gemini-embedding-001": 2048,
}
_DEFAULT_TOKEN_LIMIT = 2048  # conservative fallback for unknown future models

_VALID_TASK_TYPES: frozenset = frozenset({
    "RETRIEVAL_QUERY",
    "RETRIEVAL_DOCUMENT",
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
    "CODE_RETRIEVAL_QUERY",
})

_ERROR_HINTS: Dict[int, str] = {
    400: "Invalid request — check model name and task_type value.",
    401: "Invalid API key. Verify your GOOGLE_API_KEY or api_key in config.",
    403: "Permission denied. API key may lack access to this model.",
    404: "Model not found: '{model}'. Check spelling (e.g. 'gemini-embedding-2-preview').",
    429: "Quota exceeded. Wait and retry, or increase your Google API quota.",
    500: "Gemini service error (Google-side). Retry after a delay.",
    503: "Gemini service unavailable. Retry after a delay.",
}


def _l2_normalize(vector: List[float]) -> List[float]:
    """Ensure vector has unit L2 norm. No-op for zero vectors."""
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm > 0 else vector


def _raise_api_error(e: APIError, model: str) -> None:
    hint = _ERROR_HINTS.get(e.code, "")
    # Gemini returns HTTP 400 (not 401) when the API key is invalid
    if e.code == 400 and "api key" in str(e).lower():
        hint = "Invalid API key. Verify your GOOGLE_API_KEY or api_key in config."
    msg = f"Gemini embedding failed (HTTP {e.code})"
    if hint:
        msg += f": {hint.format(model=model)}"
    raise RuntimeError(msg) from e


class GeminiDenseEmbedder(DenseEmbedderBase):
    """Dense embedder backed by Google's Gemini Embedding models.

    REST endpoint: /v1beta/models/{model}:embedContent (SDK handles Parts format internally).
    Input token limit: per-model (8192 for gemini-embedding-2-preview, 2048 for gemini-embedding-001).
    Output dimension: 1–3072 (MRL; recommended 768, 1536, 3072; default 3072).
    Task types: RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, SEMANTIC_SIMILARITY, CLASSIFICATION,
                CLUSTERING, CODE_RETRIEVAL_QUERY, QUESTION_ANSWERING, FACT_VERIFICATION.
    Non-symmetric: use query_param/document_param in EmbeddingModelConfig.
    """

    # Default output dimensions per model (used when user does not specify `dimension`).
    # gemini-embedding-2-preview: 3072 MRL model — supports 1–3072 via output_dimensionality
    # gemini-embedding-001:       3072 (native 768-dim vectors; 3072 shown as default for MRL compat)
    # text-embedding-004:         768  fixed-dim legacy model, does not support MRL truncation
    # Future gemini-embedding-*:  default 3072 via _default_dimension() fallback
    # Future text-embedding-*:    default 768  via _default_dimension() prefix rule
    KNOWN_DIMENSIONS: Dict[str, int] = {
        "gemini-embedding-2-preview": 3072,
        "gemini-embedding-001": 3072,
        "text-embedding-004": 768,
    }

    @classmethod
    def _default_dimension(cls, model: str) -> int:
        """Return default output dimension for a Gemini model.

        Lookup order:
        1. Exact match in KNOWN_DIMENSIONS
        2. Prefix rule: text-embedding-* → 768 (legacy fixed-dim series)
        3. Fallback: 3072 (gemini-embedding-* MRL models)

        Examples:
            gemini-embedding-2-preview → 3072 (exact match)
            gemini-embedding-2         → 3072 (fallback — future model)
            text-embedding-004         → 768  (exact match)
            text-embedding-005         → 768  (prefix rule — future model)
        """
        if model in cls.KNOWN_DIMENSIONS:
            return cls.KNOWN_DIMENSIONS[model]
        if model.startswith("text-embedding-"):
            return 768
        return 3072

    def __init__(
        self,
        model_name: str = "gemini-embedding-2-preview",
        api_key: Optional[str] = None,
        dimension: Optional[int] = None,
        task_type: Optional[str] = None,
        max_concurrent_batches: int = 10,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_name, config)
        if not api_key:
            raise ValueError("Gemini provider requires api_key")
        if task_type and task_type not in _VALID_TASK_TYPES:
            raise ValueError(
                f"Invalid task_type '{task_type}'. "
                f"Valid values: {', '.join(sorted(_VALID_TASK_TYPES))}"
            )
        if dimension is not None and not (1 <= dimension <= 3072):
            raise ValueError(f"dimension must be between 1 and 3072, got {dimension}")
        self.client = genai.Client(api_key=api_key)
        self.task_type = task_type
        self._dimension = dimension or self._default_dimension(model_name)
        self._token_limit = _MODEL_TOKEN_LIMITS.get(model_name, _DEFAULT_TOKEN_LIMIT)
        self._max_concurrent_batches = max_concurrent_batches
        config_kwargs: Dict[str, Any] = {"output_dimensionality": self._dimension}
        if self.task_type:
            config_kwargs["task_type"] = self.task_type
        self._embed_config = types.EmbedContentConfig(**config_kwargs)

    def embed(self, text: str) -> EmbedResult:
        # SDK accepts plain str; converts to REST Parts format internally.
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=self._embed_config,
            )
            vector = _l2_normalize(
                truncate_and_normalize(list(result.embeddings[0].values), self._dimension)
            )
            return EmbedResult(dense_vector=vector)
        except (APIError, ClientError) as e:
            _raise_api_error(e, self.model_name)

    def embed_batch(self, texts: List[str]) -> List[EmbedResult]:
        if not texts:
            return []
        results: List[EmbedResult] = []
        for i in range(0, len(texts), _TEXT_BATCH_SIZE):
            batch = texts[i : i + _TEXT_BATCH_SIZE]
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                    config=self._embed_config,
                )
                for emb in response.embeddings:
                    vector = _l2_normalize(
                        truncate_and_normalize(list(emb.values), self._dimension)
                    )
                    results.append(EmbedResult(dense_vector=vector))
            except (APIError, ClientError) as e:
                logger.warning(
                    "Gemini batch embed failed (HTTP %d) for batch of %d, falling back to individual",
                    e.code, len(batch),
                )
                for text in batch:
                    results.append(self.embed(text))
        return results

    async def async_embed_batch(self, texts: List[str]) -> List[EmbedResult]:
        """Concurrent batch embedding via client.aio — requires anyio to be installed.

        Dispatches all 100-text chunks in parallel, bounded by max_concurrent_batches.
        Per-batch APIError falls back to individual embed() calls via thread pool.
        Raises ImportError if anyio is not installed.
        """
        if not _ANYIO_AVAILABLE:
            raise ImportError(
                "anyio is required for async_embed_batch: pip install 'openviking[gemini-async]'"
            )
        if not texts:
            return []
        batches = [texts[i : i + _TEXT_BATCH_SIZE] for i in range(0, len(texts), _TEXT_BATCH_SIZE)]
        results: List[Optional[List[EmbedResult]]] = [None] * len(batches)
        sem = anyio.Semaphore(self._max_concurrent_batches)

        async def _embed_one(idx: int, batch: List[str]) -> None:
            async with sem:
                try:
                    response = await self.client.aio.models.embed_content(
                        model=self.model_name, contents=batch, config=self._embed_config
                    )
                    results[idx] = [
                        EmbedResult(
                            dense_vector=_l2_normalize(
                                truncate_and_normalize(list(emb.values), self._dimension)
                            )
                        )
                        for emb in response.embeddings
                    ]
                except (APIError, ClientError) as e:
                    logger.warning(
                        "Gemini async batch embed failed (HTTP %d) for batch of %d, falling back",
                        e.code, len(batch),
                    )
                    results[idx] = [
                        await anyio.to_thread.run_sync(self.embed, text) for text in batch
                    ]

        async with anyio.create_task_group() as tg:
            for idx, batch in enumerate(batches):
                tg.start_soon(_embed_one, idx, batch)

        return [r for batch_results in results for r in (batch_results or [])]

    def get_dimension(self) -> int:
        return self._dimension

    def close(self):
        if hasattr(self.client, "_http_client"):
            try:
                self.client._http_client.close()
            except Exception:
                pass
