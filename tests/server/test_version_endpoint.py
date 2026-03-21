# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for memory_version field in FindResult and GET /version route (issue #817).

These tests are unit-level (no AGFS binding required) and verify:
  1. FindResult has an optional memory_version field defaulting to None.
  2. to_dict() includes memory_version when set.
  3. to_dict() omits memory_version when None (backward-compat).
  4. The GET /version route exists and is registered on the search router.
"""

from openviking_cli.retrieve.types import FindResult


def _make_empty_find_result(**kwargs) -> FindResult:
    return FindResult(memories=[], resources=[], skills=[], **kwargs)


# ---------------------------------------------------------------------------
# FindResult.memory_version field contract
# ---------------------------------------------------------------------------


def test_find_result_has_memory_version_attribute():
    """FindResult must have memory_version attribute (defaults to None)."""
    result = _make_empty_find_result()
    assert hasattr(result, "memory_version"), "FindResult missing memory_version field"
    assert result.memory_version is None


def test_find_result_memory_version_can_be_set():
    """FindResult accepts memory_version as a keyword argument."""
    result = _make_empty_find_result(memory_version=42)
    assert result.memory_version == 42


def test_find_result_memory_version_is_int_or_none():
    """memory_version must be int when set."""
    result = _make_empty_find_result(memory_version=7)
    assert isinstance(result.memory_version, int)
    assert result.memory_version >= 0


# ---------------------------------------------------------------------------
# FindResult.to_dict() — memory_version serialization
# ---------------------------------------------------------------------------


def test_to_dict_includes_memory_version_when_set():
    """to_dict() must include memory_version key when value is not None."""
    result = _make_empty_find_result(memory_version=3)
    d = result.to_dict()
    assert "memory_version" in d, "to_dict() must include memory_version when set"
    assert d["memory_version"] == 3


def test_to_dict_omits_memory_version_when_none():
    """to_dict() must omit memory_version when None (backward compatibility)."""
    result = _make_empty_find_result()
    d = result.to_dict()
    assert "memory_version" not in d, "to_dict() must NOT include memory_version when None"


def test_to_dict_memory_version_zero_is_included():
    """to_dict() must include memory_version=0 (0 is a valid version, not falsy-skipped)."""
    result = _make_empty_find_result(memory_version=0)
    d = result.to_dict()
    assert "memory_version" in d
    assert d["memory_version"] == 0


# ---------------------------------------------------------------------------
# GET /version route registration
# ---------------------------------------------------------------------------


def test_version_route_registered_on_search_router():
    """GET /version must be registered as a route on the search router."""
    from openviking.server.routers.search import router

    version_routes = [r for r in router.routes if r.path.endswith("/version")]
    assert version_routes, "GET /version route not registered on search router"
    assert any("GET" in r.methods for r in version_routes), (
        "Route /version exists but does not accept GET"
    )
