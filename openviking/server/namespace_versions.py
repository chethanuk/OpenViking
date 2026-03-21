"""Per-namespace monotonic write counter.

In-memory singleton. Counters reset to 0 on server restart.
Client-side mitigation: version cache is empty on reconnect, so the first
recall after restart always fetches (version mismatch triggers re-fetch).

SINGLE-WORKER ASSUMPTION: asyncio.Lock provides only intra-process
synchronization. Multi-worker deployments (uvicorn --workers N > 1) have
independent per-process counter dicts. A write handled by Worker A increments
only Worker A's counter; Worker B's counter remains unchanged. This can cause
a false cache hit when the client checks Worker B's stale counter. For
distributed correctness, replace this dict with Redis INCR (see follow-up).
The default deployment (single worker, local server) is unaffected.

TODO (follow-up): replace with Redis-backed counter for multi-worker safety.
TODO (follow-up): persist counter across restarts via startup epoch prefix.
See: https://github.com/volcengine/OpenViking/issues/817
"""

import asyncio
from typing import Dict


class NamespaceVersionService:
    """Thread-safe (asyncio-safe) monotonic write counter per namespace."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._versions: Dict[str, int] = {}

    async def increment(self, namespace: str) -> int:
        """Increment and return the counter for namespace."""
        async with self._lock:
            self._versions[namespace] = self._versions.get(namespace, 0) + 1
            return self._versions[namespace]

    async def get(self, namespace: str) -> int:
        """Return current counter for namespace (0 if never written)."""
        async with self._lock:
            return self._versions.get(namespace, 0)


# Module-level singleton — same pattern as `_service` in dependencies.py
_namespace_version_service: "NamespaceVersionService | None" = None


def get_namespace_version_service() -> NamespaceVersionService:
    """Return the singleton NamespaceVersionService, creating it on first call."""
    global _namespace_version_service
    if _namespace_version_service is None:
        _namespace_version_service = NamespaceVersionService()
    return _namespace_version_service
