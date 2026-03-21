# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.storage.queuefs.queue_manager import QueueManager


@pytest.mark.asyncio
async def test_concurrent_worker_increments_in_progress():
    """_worker_async_concurrent must call _on_dequeue_start() so in_progress > 0 while processing."""
    manager = QueueManager.__new__(QueueManager)
    manager._poll_interval = 0.05

    processing_started = asyncio.Event()
    stop_event = threading.Event()

    raw_msg = {"id": "msg-1", "message": "test", "context_data": {}}
    call_count = 0

    async def fake_dequeue_raw():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return raw_msg
        return None

    async def fake_process_dequeued(data):
        processing_started.set()
        await asyncio.sleep(0.01)

    async def fake_size():
        return 1 if call_count == 0 else 0

    queue = MagicMock()
    queue.name = "test-queue"
    queue.has_dequeue_handler.return_value = True
    queue.dequeue_raw = fake_dequeue_raw
    queue.size = fake_size
    queue.process_dequeued = fake_process_dequeued
    queue.ack = AsyncMock()
    queue._on_dequeue_start = MagicMock(wraps=lambda: None)
    queue._on_process_error = MagicMock()

    task = asyncio.create_task(
        manager._worker_async_concurrent(queue, stop_event, max_concurrent=5)
    )

    await asyncio.wait_for(processing_started.wait(), timeout=2.0)
    stop_event.set()

    await asyncio.gather(task, return_exceptions=True)

    # _on_dequeue_start must have been called once (for the one message)
    queue._on_dequeue_start.assert_called_once()
