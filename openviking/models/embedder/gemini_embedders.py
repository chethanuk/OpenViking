# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Gemini Embedding 2 provider using the official google-genai SDK."""

from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError

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

_SUPPORTED_MULTIMODAL_MIMES = frozenset({
    # Images
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    # Audio
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/ogg",
    "audio/flac",
    # Video
    "video/mp4",
    "video/mpeg",
    "video/mov",
    "video/avi",
    "video/webm",
    "video/wmv",
    "video/3gpp",
    # Documents
    "application/pdf",
})

_TEXT_BATCH_SIZE = 100

# Maximum input tokens per Gemini embedding request (model hard limit).
_GEMINI_INPUT_TOKEN_LIMIT = 8192

_GEMINI_PDF_MAX_PAGES = 6


def _count_pdf_pages(data: bytes) -> int:
    """Count pages in a PDF using pdfminer-six (core dependency).

    Returns 0 if the data is not a valid PDF or pdfminer cannot parse it.
    Caller treats 0 as 'unknown' — proceeds with a warning.
    """
    try:
        import io
        from pdfminer.pdfpage import PDFPage
        return sum(1 for _ in PDFPage.get_pages(io.BytesIO(data), check_extractable=False))
    except Exception:
        return 0


class GeminiDenseEmbedder(DenseEmbedderBase):
    """Dense embedder backed by Google's Gemini Embedding 2 model.

    Input token limit: 8,192 tokens per request.
    Output dimension: 128–3072 (recommended: 768, 1536, 3072; default: 3072).
    """

    KNOWN_DIMENSIONS: Dict[str, int] = {
        "gemini-embedding-2-preview": 3072,
        "gemini-embedding-001": 3072,
        "text-embedding-004": 768,
    }

    def __init__(
        self,
        model_name: str = "gemini-embedding-2-preview",
        api_key: Optional[str] = None,
        dimension: Optional[int] = None,
        task_type: Optional[str] = None,
        max_concurrent_batches: int = 10,
        enable_multimodal: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_name, config)
        if not api_key:
            raise ValueError("Gemini provider requires api_key")
        self.client = genai.Client(api_key=api_key)
        self.task_type = task_type
        self._dimension = dimension or self.KNOWN_DIMENSIONS.get(model_name, 3072)
        self._max_concurrent_batches = max_concurrent_batches
        self._enable_multimodal = enable_multimodal
        # Index config: RETRIEVAL_DOCUMENT by default, or user's task_type override
        index_task = self.task_type or "RETRIEVAL_DOCUMENT"
        self._index_config = types.EmbedContentConfig(
            output_dimensionality=self._dimension,
            task_type=index_task,
        )
        # Query config: always RETRIEVAL_QUERY — never overridden by config
        self._query_config = types.EmbedContentConfig(
            output_dimensionality=self._dimension,
            task_type="RETRIEVAL_QUERY",
        )
        # Backward-compat alias: embed(), embed_batch(), embed_multimodal() use this
        self._embed_config = self._index_config

    @property
    def supports_multimodal(self) -> bool:
        return self._enable_multimodal

    def embed(self, text: str) -> EmbedResult:
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=self._embed_config,
            )
            vector = truncate_and_normalize(list(result.embeddings[0].values), self._dimension)
            return EmbedResult(dense_vector=vector)
        except APIError as e:
            raise RuntimeError(f"Gemini embedding failed (code={e.code}): {e}") from e

    def embed_query(self, text: str) -> EmbedResult:
        """Embed a query string using RETRIEVAL_QUERY task type."""
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=self._query_config,
            )
            vector = truncate_and_normalize(list(result.embeddings[0].values), self._dimension)
            return EmbedResult(dense_vector=vector)
        except APIError as e:
            raise RuntimeError(f"Gemini embedding failed (code={e.code}): {e}") from e

    def embed_multimodal(self, vectorize: "Vectorize") -> EmbedResult:  # type: ignore[name-defined]
        """Embed a Vectorize object that may contain text, image, audio, video, or PDF parts.

        When enable_multimodal=False (default), behaves like embed(): text only.
        When enable_multimodal=True: uses vectorize.get_parts() to build the Gemini parts list.
        PDF parts are checked against the 6-page limit before sending. If no supported media
        parts remain after filtering, falls back to embed(text) for efficiency.
        """
        if not self._enable_multimodal:
            return self.embed(getattr(vectorize, "text", ""))

        from openviking.core.context import ModalContent
        parts_list = vectorize.get_parts()

        api_parts: List[Any] = []
        has_media_part = False

        for part in parts_list:
            if isinstance(part, str):
                if part:
                    api_parts.append(types.Part.from_text(text=part))
            else:
                # ModalContent
                if part.data is None or part.mime_type not in _SUPPORTED_MULTIMODAL_MIMES:
                    continue
                if part.mime_type == "application/pdf":
                    pages = _count_pdf_pages(part.data)
                    if pages > _GEMINI_PDF_MAX_PAGES:
                        logger.warning(
                            f"PDF {part.uri!r} has {pages} pages "
                            f"(Gemini limit={_GEMINI_PDF_MAX_PAGES}). "
                            "Falling back to text-only embedding. "
                            "Pre-chunk PDFs to ≤6 pages upstream."
                        )
                        return self.embed(vectorize.text)
                    if pages == 0:
                        logger.warning(
                            f"PDF {part.uri!r}: page count unknown "
                            "(pdfminer could not parse). "
                            "Sending to API — may fail if >6 pages."
                        )
                api_parts.append(
                    types.Part.from_bytes(data=part.data, mime_type=part.mime_type)
                )
                has_media_part = True

        # No supported media parts — use plain text embed for efficiency
        if not has_media_part:
            return self.embed(vectorize.text)

        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=[types.Content(parts=api_parts)],
                config=self._embed_config,
            )
            vector = truncate_and_normalize(list(result.embeddings[0].values), self._dimension)
            return EmbedResult(dense_vector=vector)
        except APIError as e:
            if e.code in (429, 502, 503, 504):
                raise RuntimeError(f"Gemini transient error (code={e.code}), caller should retry") from e
            logger.warning(
                f"Gemini multimodal embed failed (code={e.code}) — "
                f"falling back to text. [multimodal_fallback=True]"
            )
            return self.embed(vectorize.text)

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
                    vector = truncate_and_normalize(list(emb.values), self._dimension)
                    results.append(EmbedResult(dense_vector=vector))
            except APIError as e:
                logger.warning(
                    f"Gemini batch embed failed (code={e.code}) for batch of {len(batch)}, "
                    "falling back to individual calls"
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
                            dense_vector=truncate_and_normalize(list(emb.values), self._dimension)
                        )
                        for emb in response.embeddings
                    ]
                except APIError as e:
                    logger.warning(
                        f"Gemini batch embed failed (code={e.code}) for batch of {len(batch)}, "
                        "falling back to individual calls"
                    )
                    results[idx] = [
                        await anyio.to_thread.run_sync(self.embed, text) for text in batch
                    ]

        async with anyio.create_task_group() as tg:
            for idx, batch in enumerate(batches):
                tg.start_soon(_embed_one, idx, batch)

        return [r for batch_results in results for r in (batch_results or [])]

    async def async_embed_multimodal_batch(
        self, vectorizes: List["Vectorize"]
    ) -> List[EmbedResult]:
        """Concurrent multimodal batch via anyio Semaphore — requires anyio.

        Bounded by max_concurrent_batches. Transient errors propagate; others fall back
        to text embed via embed_multimodal's own error handling.
        Falls back to base class (asyncio.gather) if anyio is unavailable.
        """
        if not _ANYIO_AVAILABLE:
            return await super().async_embed_multimodal_batch(vectorizes)
        if not vectorizes:
            return []

        results: List[Optional[EmbedResult]] = [None] * len(vectorizes)
        sem = anyio.Semaphore(self._max_concurrent_batches)

        async def _embed_one(idx: int, v: "Vectorize") -> None:
            async with sem:
                try:
                    results[idx] = await anyio.to_thread.run_sync(
                        self.embed_multimodal, v
                    )
                except RuntimeError as e:
                    if "transient" in str(e).lower():
                        raise
                    logger.warning(
                        f"async_embed_multimodal_batch item {idx} failed: {e}. "
                        "Falling back to text embed."
                    )
                    text = getattr(v, "text", "")
                    results[idx] = await anyio.to_thread.run_sync(self.embed, text)

        async with anyio.create_task_group() as tg:
            for idx, v in enumerate(vectorizes):
                tg.start_soon(_embed_one, idx, v)

        return [r for r in results]  # type: ignore[return-value]

    def get_dimension(self) -> int:
        return self._dimension

    def close(self):
        if hasattr(self.client, "_http_client"):
            try:
                self.client._http_client.close()
            except Exception:
                pass
