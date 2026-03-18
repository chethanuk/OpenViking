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

# Used for fixture parametrize — each param is a single (name, dim, limit) tuple
_MODELS_FIXTURE = [
    pytest.param(("gemini-embedding-2-preview", 3072, 8192), id="gemini-embedding-2-preview"),
    pytest.param(("gemini-embedding-001", 3072, 2048), id="gemini-embedding-001"),
]

# Used for @pytest.mark.parametrize("model_name,_dim,token_limit", ...)
_MODELS = [
    pytest.param("gemini-embedding-2-preview", 3072, 8192, id="gemini-embedding-2-preview"),
    pytest.param("gemini-embedding-001", 3072, 2048, id="gemini-embedding-001"),
]


@pytest.fixture(scope="module", params=_MODELS_FIXTURE)
def embedder(request):
    from openviking.models.embedder.gemini_embedders import GeminiDenseEmbedder
    model_name, _, _ = request.param
    return GeminiDenseEmbedder(model_name, api_key=API_KEY, dimension=768)


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


def test_batch_over_100(embedder):
    """150 texts auto-split into 2 batches (100 + 50)."""
    texts = [f"sentence number {i}" for i in range(150)]
    results = embedder.embed_batch(texts)
    assert len(results) == 150
    for r in results:
        assert r.dense_vector and len(r.dense_vector) == 768


@pytest.mark.parametrize("model_name,_dim,token_limit", _MODELS)
def test_large_text_chunking(model_name, _dim, token_limit):
    """Text exceeding the model's token limit is auto-chunked by base class."""
    from openviking.models.embedder.gemini_embedders import GeminiDenseEmbedder
    # ~2× the token limit to force chunking
    phrase = "Machine learning is a subset of artificial intelligence. "
    large = phrase * ((token_limit * 2) // len(phrase.split()) + 10)
    e = GeminiDenseEmbedder(model_name, api_key=API_KEY, dimension=768)
    r = e.embed(large)
    assert r.dense_vector and len(r.dense_vector) == 768
    norm = sum(v**2 for v in r.dense_vector) ** 0.5
    assert 0.99 < norm < 1.01, f"chunked vector not L2-normalized, norm={norm}"


@pytest.mark.parametrize("task_type", [
    "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY",
    "CLASSIFICATION", "CLUSTERING", "CODE_RETRIEVAL_QUERY",
    "QUESTION_ANSWERING", "FACT_VERIFICATION",
])
def test_all_task_types_accepted(task_type):
    """All 8 Gemini task types must be accepted by the API without error."""
    from openviking.models.embedder.gemini_embedders import GeminiDenseEmbedder
    e = GeminiDenseEmbedder(
        "gemini-embedding-2-preview", api_key=API_KEY,
        task_type=task_type, dimension=768
    )
    r = e.embed("test input for task type validation")
    assert r.dense_vector and len(r.dense_vector) == 768



def test_config_nonsymmetric_routing():
    """EmbeddingConfig query/document embedders wire task_type via query_param/document_param."""
    from openviking_cli.utils.config.embedding_config import EmbeddingConfig, EmbeddingModelConfig
    cfg = EmbeddingConfig(
        dense=EmbeddingModelConfig(
            model="gemini-embedding-2-preview", provider="gemini",
            api_key=API_KEY, dimension=768,
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
    bad = GeminiDenseEmbedder("gemini-embedding-2-preview", api_key="INVALID_KEY_XYZZY_123")
    with pytest.raises(RuntimeError, match="Invalid API key"):
        bad.embed("hello")


def test_invalid_model_error_message():
    """Unknown model name must raise RuntimeError with model-not-found hint."""
    from openviking.models.embedder.gemini_embedders import GeminiDenseEmbedder
    bad = GeminiDenseEmbedder("gemini-embedding-does-not-exist-xyz", api_key=API_KEY)
    with pytest.raises(RuntimeError, match="Model not found"):
        bad.embed("hello")
