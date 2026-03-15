# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tests for DenseEmbedderBase default methods."""
from openviking.models.embedder.base import DenseEmbedderBase, EmbedResult


class _StubEmbedder(DenseEmbedderBase):
    def embed(self, text: str) -> EmbedResult:
        return EmbedResult(dense_vector=[0.1, 0.2])

    def get_dimension(self) -> int:
        return 2


def test_dense_embedder_base_embed_query_defaults_to_embed():
    """DenseEmbedderBase.embed_query() must fall back to embed() for non-Gemini embedders."""
    stub = _StubEmbedder("stub-model")
    result = stub.embed_query("test query")
    assert result.dense_vector == [0.1, 0.2]
