# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, patch

import pytest

from openviking.storage.queuefs.embedding_queue import EmbeddingQueue


@pytest.mark.asyncio
async def test_dequeue_logs_on_parse_failure():
    """EmbeddingQueue.dequeue() must log an error when EmbeddingMsg.from_dict fails."""
    queue = EmbeddingQueue.__new__(EmbeddingQueue)

    raw_dict = {"id": "test-id", "message": "hello", "context_data": {}}

    with patch(
        "openviking.storage.queuefs.embedding_queue.NamedQueue.dequeue",
        new_callable=AsyncMock,
        return_value=raw_dict,
    ):
        with patch(
            "openviking.storage.queuefs.embedding_queue.EmbeddingMsg.from_dict",
            side_effect=ValueError("bad message"),
        ):
            with patch("openviking.storage.queuefs.embedding_queue.logger") as mock_logger:
                result = await queue.dequeue()

    assert result is None
    mock_logger.error.assert_called_once()
    args, kwargs = mock_logger.error.call_args
    assert "EmbeddingQueue" in args[0]
    assert kwargs.get("exc_info") is True
