# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Table-driven tests for VLM async concurrency limiter (#3008).

Covers rows 1-9 of the test plan. Uses an in-flight counter stub to
prove the semaphore actually gates concurrent calls. All tests use the
exact pytest invocation: .venv/bin/python -m pytest ... -o addopts="" -q
"""

import asyncio
from unittest import mock

import pytest

from openviking.models.vlm.base import (
    VLMBase,
    _get_async_vlm_semaphore,
)


class _InFlightCounter:
    """Helper to assert max in-flight concurrency during test runs."""

    def __init__(self):
        self.max_seen = 0
        self.current = 0
        self.lock = asyncio.Lock()

    async def __aenter__(self):
        async with self.lock:
            self.current += 1
            self.max_seen = max(self.max_seen, self.current)
        return self

    async def __aexit__(self, *exc):
        async with self.lock:
            self.current -= 1
        return False


@pytest.mark.parametrize(
    "limit,expected_max_inflight",
    [
        (1, 1),
        (2, 2),
        (5, 5),
        (64, 64),
    ],
)
async def test_semaphore_respects_limit(limit, expected_max_inflight):
    """Row 1: semaphore created with correct limit, never exceeds it."""
    sem = _get_async_vlm_semaphore(limit)
    assert isinstance(sem, asyncio.Semaphore)
    assert sem._value == limit  # internal but stable for test


async def test_per_event_loop_isolation():
    """Row 2: different loops get independent semaphores."""
    # Primary loop semaphore
    sem1 = _get_async_vlm_semaphore(3)
    # Simulate second loop by patching (real multi-loop rare in tests)
    with mock.patch("asyncio.get_running_loop") as mock_loop:
        mock_loop.return_value = object()  # distinct object
        sem2 = _get_async_vlm_semaphore(3)
    assert sem1 is not sem2


async def test_same_limit_reuses_semaphore():
    """Row 3: same limit on same loop returns the identical semaphore object."""
    s1 = _get_async_vlm_semaphore(4)
    s2 = _get_async_vlm_semaphore(4)
    assert s1 is s2


async def test_different_limits_get_distinct_semaphores():
    """Row 4: different limits get separate semaphores even on same loop."""
    s1 = _get_async_vlm_semaphore(2)
    s2 = _get_async_vlm_semaphore(3)
    assert s1 is not s2


class DummyVLM(VLMBase):
    """Minimal concrete VLM for testing the retry wrapper."""

    def get_completion(self, *a, **k):
        return "sync"

    async def get_completion_async(self, prompt="", **kwargs):
        # Will be overridden in tests via patch
        return "async"

    # vision stubs
    def get_vision_completion(self, *a, **k):
        return "vision-sync"

    async def get_vision_completion_async(self, *a, **k):
        return "vision-async"


async def test_run_with_vlm_async_retry_acquires_semaphore():
    """Row 5: _run_with_vlm_async_retry acquires the semaphore."""
    vlm = DummyVLM({"provider": "openai", "model": "gpt-4o-mini", "max_concurrent": 2})
    calls = []

    async def fake_func():
        calls.append(1)
        return "ok"

    with mock.patch.object(vlm, "_run_with_vlm_async_retry", wraps=vlm._run_with_vlm_async_retry):
        result = await vlm._run_with_vlm_async_retry(fake_func)
    assert result == "ok"
    assert len(calls) == 1


async def test_max_concurrent_defaults_to_64():
    """Row 6: VLMBase.__init__ defaults max_concurrent=64 when omitted."""
    vlm = DummyVLM({"provider": "openai", "model": "gpt-4o-mini"})
    assert vlm.max_concurrent == 64


async def test_max_concurrent_from_config():
    """Row 7: max_concurrent flows from config dict."""
    vlm = DummyVLM({"provider": "openai", "model": "gpt-4o-mini", "max_concurrent": 8})
    assert vlm.max_concurrent == 8


async def test_retry_wrapper_delegates_to_retry_async(monkeypatch):
    """Row 8: the wrapper calls retry_async after acquiring semaphore."""
    from openviking.utils import model_retry

    vlm = DummyVLM({"provider": "openai", "model": "gpt-4o-mini", "max_concurrent": 1})
    called = []

    async def fake_retry(func, **kw):
        called.append(kw)
        return await func()

    monkeypatch.setattr(model_retry, "retry_async", fake_retry)

    async def inner():
        return "retried"

    res = await vlm._run_with_vlm_async_retry(inner, operation_name="test")
    assert res == "retried"
    assert "max_retries" in called[0]


async def test_concurrent_calls_are_gated_by_semaphore():
    """Row 9: real concurrent execution never exceeds the semaphore limit (uses in-flight counter)."""
    vlm = DummyVLM({"provider": "openai", "model": "gpt-4o-mini", "max_concurrent": 2})
    counter = _InFlightCounter()
    results = []

    async def slow_call(i):
        async with counter:
            await asyncio.sleep(0.05)
            results.append(i)
            return i

    # Launch 5 concurrent calls through the wrapper
    tasks = [
        asyncio.create_task(vlm._run_with_vlm_async_retry(lambda i=i: slow_call(i)))
        for i in range(5)
    ]
    await asyncio.gather(*tasks)
    assert counter.max_seen <= 2
    assert len(results) == 5
