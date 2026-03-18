# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Gemini-specific EmbeddingModelConfig and EmbeddingConfig behavior."""

from unittest.mock import patch

import pytest

from openviking_cli.utils.config.embedding_config import EmbeddingConfig, EmbeddingModelConfig


def _gcfg(**kw) -> EmbeddingModelConfig:
    """Helper: build a Gemini EmbeddingModelConfig with defaults."""
    return EmbeddingModelConfig(
        model="gemini-embedding-2-preview", provider="gemini", api_key="test-key", **kw
    )


class TestGeminiDimension:
    def test_preview_defaults_3072(self):
        assert _gcfg().get_effective_dimension() == 3072

    def test_001_defaults_3072(self):
        cfg = EmbeddingModelConfig(model="gemini-embedding-001", provider="gemini", api_key="k")
        assert cfg.get_effective_dimension() == 3072

    def test_004_defaults_768(self):
        cfg = EmbeddingModelConfig(model="text-embedding-004", provider="gemini", api_key="k")
        assert cfg.get_effective_dimension() == 768

    def test_unknown_model_defaults_3072(self):
        cfg = EmbeddingModelConfig(model="gemini-embedding-future", provider="gemini", api_key="k")
        assert cfg.get_effective_dimension() == 3072

    def test_explicit_dimension_overrides_default(self):
        assert _gcfg(dimension=1536).get_effective_dimension() == 1536

    def test_text_embedding_prefix_defaults_768(self):
        """text-embedding-* future models default to 768 via prefix rule."""
        cfg = EmbeddingModelConfig(model="text-embedding-005", provider="gemini", api_key="k")
        assert cfg.get_effective_dimension() == 768

    def test_future_gemini_model_defaults_3072(self):
        """Future gemini-embedding-* models default to 3072 via fallback."""
        for model in ["gemini-embedding-2", "gemini-embedding-2.1", "gemini-embedding-3-preview"]:
            cfg = EmbeddingModelConfig(model=model, provider="gemini", api_key="k")
            assert cfg.get_effective_dimension() == 3072


class TestGeminiContextRouting:
    @patch("openviking.models.embedder.gemini_embedders.genai.Client")
    def test_get_query_embedder_uses_query_param(self, _mock):
        cfg = EmbeddingConfig(
            dense=_gcfg(query_param="RETRIEVAL_QUERY", document_param="RETRIEVAL_DOCUMENT")
        )
        assert cfg.get_query_embedder().task_type == "RETRIEVAL_QUERY"

    @patch("openviking.models.embedder.gemini_embedders.genai.Client")
    def test_get_document_embedder_uses_document_param(self, _mock):
        cfg = EmbeddingConfig(
            dense=_gcfg(query_param="RETRIEVAL_QUERY", document_param="RETRIEVAL_DOCUMENT")
        )
        assert cfg.get_document_embedder().task_type == "RETRIEVAL_DOCUMENT"

    @patch("openviking.models.embedder.gemini_embedders.genai.Client")
    def test_symmetric_uses_task_type_field(self, _mock):
        cfg = EmbeddingConfig(dense=_gcfg(task_type="SEMANTIC_SIMILARITY"))
        assert cfg.get_embedder().task_type == "SEMANTIC_SIMILARITY"

    @patch("openviking.models.embedder.gemini_embedders.genai.Client")
    def test_symmetric_no_task_type_is_none(self, _mock):
        cfg = EmbeddingConfig(dense=_gcfg())
        assert cfg.get_embedder().task_type is None

    @patch("openviking.models.embedder.gemini_embedders.genai.Client")
    def test_only_query_param_set_routes_correctly(self, _mock):
        """When only query_param is set, document embedder falls back to static task_type."""
        cfg = EmbeddingConfig(dense=_gcfg(query_param="RETRIEVAL_QUERY"))
        assert cfg.get_query_embedder().task_type == "RETRIEVAL_QUERY"
        assert cfg.get_document_embedder().task_type is None


class TestGeminiConfigValidation:
    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key"):
            EmbeddingModelConfig(model="gemini-embedding-2-preview", provider="gemini")

    def test_invalid_query_param_raises(self):
        with pytest.raises(ValueError, match="Invalid query_param"):
            _gcfg(query_param="NOT_A_VALID_TYPE")

    def test_invalid_document_param_raises(self):
        with pytest.raises(ValueError, match="Invalid document_param"):
            _gcfg(document_param="ALSO_INVALID")

    def test_invalid_task_type_raises(self):
        with pytest.raises(ValueError, match="Invalid task_type"):
            _gcfg(task_type="BAD_TYPE")

    def test_valid_task_types_accepted(self):
        for t in [
            "RETRIEVAL_QUERY",
            "RETRIEVAL_DOCUMENT",
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
            "QUESTION_ANSWERING",
            "FACT_VERIFICATION",
            "CODE_RETRIEVAL_QUERY",
        ]:
            _gcfg(task_type=t)  # must not raise
