"""Tests for per-namespace monotonic write counter."""

import pytest

from openviking.server.namespace_versions import (
    NamespaceVersionService,
    get_namespace_version_service,
)


@pytest.mark.asyncio
async def test_increment_and_get():
    """Counter increments monotonically per namespace."""
    svc = NamespaceVersionService()
    await svc.increment("ns-1")
    await svc.increment("ns-1")
    assert await svc.get("ns-1") == 2


@pytest.mark.asyncio
async def test_unknown_namespace_returns_zero():
    """Unknown namespace returns 0 — not an error."""
    svc = NamespaceVersionService()
    assert await svc.get("never-seen") == 0


@pytest.mark.asyncio
async def test_namespaces_are_independent():
    """Writes to ns-1 do not affect ns-2."""
    svc = NamespaceVersionService()
    await svc.increment("ns-1")
    await svc.increment("ns-1")
    await svc.increment("ns-2")
    assert await svc.get("ns-1") == 2
    assert await svc.get("ns-2") == 1


@pytest.mark.asyncio
async def test_uses_asyncio_lock():
    """Concurrent increments are safe (asyncio.Lock, not threading.Lock)."""
    import asyncio as aio

    svc = NamespaceVersionService()
    await aio.gather(*[svc.increment("concurrent-ns") for _ in range(50)])
    assert await svc.get("concurrent-ns") == 50


def test_module_singleton_exists():
    """get_namespace_version_service() returns a singleton."""
    svc1 = get_namespace_version_service()
    svc2 = get_namespace_version_service()
    assert svc1 is svc2
