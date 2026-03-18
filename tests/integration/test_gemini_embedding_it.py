# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for GeminiDenseEmbedder — require real GOOGLE_API_KEY.

Run: GOOGLE_API_KEY=<key> pytest tests/integration/test_gemini_embedding_it.py -v

Auto-skipped when GOOGLE_API_KEY is not set. No mocking — real API calls.
"""
import os

import pytest

API_KEY = os.environ.get("GOOGLE_API_KEY")
pytestmark = pytest.mark.skipif(not API_KEY, reason="GOOGLE_API_KEY not set")
MODEL = "gemini-embedding-2-preview"


@pytest.fixture(scope="module")
def embedder():
    from openviking.models.embedder.gemini_embedders import GeminiDenseEmbedder
    return GeminiDenseEmbedder(MODEL, api_key=API_KEY, dimension=768)


def test_embed_returns_correct_dimension(embedder):
    r = embedder.embed("What is machine learning?")
    assert r.dense_vector and len(r.dense_vector) == 768
    norm = sum(v**2 for v in r.dense_vector) ** 0.5
    assert 0.99 < norm < 1.01, f"vector not L2-normalized, norm={norm}"


def test_embed_batch_count(embedder):
    texts = ["apple", "banana", "cherry", "date", "elderberry"]
    results = embedder.embed_batch(texts)
    assert len(results) == len(texts)
    for r in results:
        assert r.dense_vector and len(r.dense_vector) == 768


def test_large_text_chunking(embedder):
    """Text > 8192 tokens is auto-chunked by base class and returns merged L2-normalized vector."""
    large = "Machine learning is a subset of artificial intelligence. " * 600  # ~12k tokens
    r = embedder.embed(large)
    assert r.dense_vector and len(r.dense_vector) == 768
    norm = sum(v**2 for v in r.dense_vector) ** 0.5
    assert 0.99 < norm < 1.01, f"chunked vector not L2-normalized, norm={norm}"


def test_task_type_query_vs_document_differ():
    """RETRIEVAL_QUERY and RETRIEVAL_DOCUMENT embeddings for same text must differ."""
    from openviking.models.embedder.gemini_embedders import GeminiDenseEmbedder
    qe = GeminiDenseEmbedder(MODEL, api_key=API_KEY, task_type="RETRIEVAL_QUERY", dimension=768)
    de = GeminiDenseEmbedder(MODEL, api_key=API_KEY, task_type="RETRIEVAL_DOCUMENT", dimension=768)
    text = "What is the capital of France?"
    q = qe.embed(text).dense_vector
    d = de.embed(text).dense_vector
    dot = sum(a * b for a, b in zip(q, d))
    assert dot < 0.999, f"RETRIEVAL_QUERY and RETRIEVAL_DOCUMENT should produce different vectors, dot={dot}"


def test_config_nonsymmetric_routing():
    """EmbeddingConfig query/document embedders wire task_type via query_param/document_param."""
    from openviking_cli.utils.config.embedding_config import EmbeddingConfig, EmbeddingModelConfig
    cfg = EmbeddingConfig(
        dense=EmbeddingModelConfig(
            model=MODEL, provider="gemini", api_key=API_KEY, dimension=768,
            query_param="RETRIEVAL_QUERY", document_param="RETRIEVAL_DOCUMENT",
        )
    )
    q = cfg.get_query_embedder()
    d = cfg.get_document_embedder()
    assert q.task_type == "RETRIEVAL_QUERY"
    assert d.task_type == "RETRIEVAL_DOCUMENT"
    assert q.embed("search query").dense_vector is not None
    assert d.embed("document text").dense_vector is not None


def test_invalid_api_key_error_message():
    """Wrong API key must raise RuntimeError with 'Invalid API key' hint."""
    from openviking.models.embedder.gemini_embedders import GeminiDenseEmbedder
    bad = GeminiDenseEmbedder(MODEL, api_key="INVALID_KEY_XYZZY_123")
    with pytest.raises(RuntimeError, match="Invalid API key"):
        bad.embed("hello")


def test_batch_over_100(embedder):
    """150 texts auto-split into 2 batches (100 + 50)."""
    texts = [f"sentence number {i}" for i in range(150)]
    results = embedder.embed_batch(texts)
    assert len(results) == 150
    for r in results:
        assert r.dense_vector and len(r.dense_vector) == 768
