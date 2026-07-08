# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Repro-style integration tests for the multi-session extraction burst (#3008).

These mirror the plan's rows 10-12. The plan's original target was
``tests/session/test_memory_extraction_concurrency.py`` driving 4 real
``commit_async`` calls and asserting zero ``.failed.json``. That variant cannot
run in this environment because the RAGFS/AGFS native binding (``ragfs_python``)
is unavailable, so ``AsyncOpenViking`` init (and therefore the whole session
extraction stack + ``.failed.json`` machinery) errors on collection.

Instead we exercise the SAME fix code at the VLM client layer — the one place
every async extraction call routes through — using the plan's exact stub:
real ``OpenAIVLM`` instances with ``client.chat.completions.create`` replaced by
an async callable that (a) tracks a live in-flight counter and (b) raises a
synthetic 429 whenever more than the provider's threshold are in flight. The
extraction fan-out (N sessions * 3 gathered VLM calls) is modelled as N*3
concurrent tasks. A raised-after-exhaustion error is the exact condition that
drives the session layer to write ``.failed.json``; zero raised errors == zero
``.failed.json``.
"""

import asyncio
from types import SimpleNamespace

from openviking.models.vlm.backends.openai_vlm import OpenAIVLM


def _make_response(content: str = "ok"):
    """Minimal object shaped like an OpenAI chat completion response."""
    message = SimpleNamespace(content=content, reasoning_content=None, tool_calls=None)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=None)


class _ProviderStub:
    """Stands in for the provider's async ``chat.completions.create``.

    Raises a synthetic 429 whenever concurrent in-flight calls exceed
    ``threshold`` — modelling a provider whose real rate limit is ``threshold``.
    """

    def __init__(self, threshold: int, call_latency: float = 0.02):
        self.threshold = threshold
        self.call_latency = call_latency
        self.in_flight = 0
        self.max_seen = 0
        self.error_count = 0

    async def create(self, **kwargs):
        # Single-threaded event loop: no await between bump and check, so the
        # counter reading is exact.
        self.in_flight += 1
        self.max_seen = max(self.max_seen, self.in_flight)
        try:
            if self.in_flight > self.threshold:
                self.error_count += 1
                raise Exception("Error code: 429 - Too Many Requests")
            await asyncio.sleep(self.call_latency)
            return _make_response("ok")
        finally:
            self.in_flight -= 1


def _make_vlm(max_concurrent: int, stub: _ProviderStub, max_retries: int = 0) -> OpenAIVLM:
    vlm = OpenAIVLM(
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test-key",
            "api_base": "http://localhost:0",
            "max_concurrent": max_concurrent,
            "max_retries": max_retries,
        }
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=stub.create))
    )
    # Route every async call through the stub instead of a real HTTP client.
    vlm.get_async_client = lambda: fake_client  # type: ignore[method-assign]
    return vlm


# 4 sessions * 3 gathered extraction steps (summary + long-term + execution).
N_SESSIONS = 4
STEPS_PER_SESSION = 3
TOTAL_CALLS = N_SESSIONS * STEPS_PER_SESSION  # 12


async def _fan_out(vlm: OpenAIVLM, n: int):
    """Fire ``n`` concurrent extraction-style calls, mimicking gather fan-out."""
    tasks = [
        asyncio.create_task(vlm.get_completion_async(prompt=f"extract {i}"))
        for i in range(n)
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def test_multi_session_extraction_no_429_silent_loss():
    """Row 10 (repro): limiter sized to the provider's real limit prevents the
    burst, so the 12 concurrent extraction calls all succeed — no error is
    raised, i.e. zero ``.failed.json`` and no silent memory loss."""
    provider_limit = 2
    stub = _ProviderStub(threshold=provider_limit)
    vlm = _make_vlm(max_concurrent=provider_limit, stub=stub)

    results = await _fan_out(vlm, TOTAL_CALLS)

    failures = [r for r in results if isinstance(r, Exception)]
    assert failures == [], f"expected zero failures, got: {failures}"
    assert stub.error_count == 0, "provider should never have seen a burst"
    assert stub.max_seen <= provider_limit, f"peak in-flight {stub.max_seen} > {provider_limit}"
    assert all(r == "ok" for r in results)


async def test_multi_session_429_without_limiter_control():
    """Row 11 (control): with the limiter effectively disabled (budget >> load),
    all 12 calls fire at once, exceed the provider's limit, and 429s propagate as
    failures — this is the path that writes ``.failed.json``. Proves the repro is
    real and that the semaphore (not retry) is what fixes it."""
    provider_limit = 2
    stub = _ProviderStub(threshold=provider_limit)
    # Budget far above the load == no effective concurrency cap.
    vlm = _make_vlm(max_concurrent=1000, stub=stub)

    results = await _fan_out(vlm, TOTAL_CALLS)

    failures = [r for r in results if isinstance(r, Exception)]
    assert failures, "control run must produce 429 failures (the .failed.json path)"
    assert stub.max_seen > provider_limit, "burst should have formed without the limiter"


async def test_semantic_queue_plus_extraction_no_deadlock():
    """Row 12: a semantic-queue-style workload and an extraction-style workload
    sharing the ONE VLM semaphore both complete within the timeout — the shared
    budget yields backpressure, not a starvation deadlock (no task holds the
    semaphore while awaiting a second slot)."""
    provider_limit = 2
    stub = _ProviderStub(threshold=provider_limit)
    # Two distinct VLM instances (semantic queue vs extraction) share the budget.
    vlm_semantic = _make_vlm(max_concurrent=provider_limit, stub=stub)
    vlm_extraction = _make_vlm(max_concurrent=provider_limit, stub=stub)

    async def both():
        return await asyncio.gather(
            _fan_out(vlm_semantic, 6),
            _fan_out(vlm_extraction, 6),
        )

    semantic_res, extraction_res = await asyncio.wait_for(both(), timeout=10.0)

    all_res = list(semantic_res) + list(extraction_res)
    assert [r for r in all_res if isinstance(r, Exception)] == []
    assert stub.max_seen <= provider_limit, f"peak in-flight {stub.max_seen} > {provider_limit}"
