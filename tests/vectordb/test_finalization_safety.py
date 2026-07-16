# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Finalization-safety guards for vector index cleanup (issue #3101).

During StreamableHTTP / MCP session teardown the interpreter can enter
finalization (sys.meta_path is None) while Collection.__del__ and the
APScheduler index-manage job still try to flush/close native indexes.
That path SEGVs/ABRTs. These tests pin the sys.is_finalizing() guards
without touching the native C++ engine.
"""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock

import pytest

from openviking.storage.vectordb.collection.collection import Collection
from openviking.storage.vectordb.collection.local_collection import LocalCollection
from openviking.storage.vectordb.index.local_index import PersistentIndex


def _make_persistent_index(tmp_path, engine_proxy: MagicMock) -> PersistentIndex:
    # Bypass __init__ deliberately: stay native-free and avoid disk/engine setup.
    idx = object.__new__(PersistentIndex)
    idx.engine_proxy = engine_proxy
    idx.version_dir = str(tmp_path)
    idx.now_version = "1"
    # LocalIndex.close() reads dense_search — omit → AttributeError masks the bug.
    idx.dense_search = None
    return idx


def _make_local_collection(
    scheduler: MagicMock,
    rebuild: MagicMock,
    persist_all: MagicMock,
) -> LocalCollection:
    col = object.__new__(LocalCollection)
    col.scheduler = scheduler
    col.index_manage_job_id = "job-1"
    col.index_maintenance_seconds = 60
    col.collection_name = "default"
    col._rebuild_indexes_if_needed = rebuild
    col._persist_all_indexes = persist_all
    return col


def _make_collection_wrapper(inner: MagicMock) -> Collection:
    # Bypass __init__ assert isinstance(collection, ICollection).
    wrapper = object.__new__(Collection)
    wrapper._Collection__collection = inner
    return wrapper


# Shutdown triggers that reach the same finalization-unsafe cleanup path.
# StreamableHTTP client disconnect and SSE reconnect both end in session
# manager shutdown → GC/APScheduler → persist/close during finalization.
SHUTDOWN_TRIGGERS = (
    "streamable_http_disconnect",
    "sse_reconnect",
    "gc_finalization",
)


@pytest.mark.parametrize("finalizing", [True, False], ids=["finalizing", "not_finalizing"])
@pytest.mark.parametrize("shutdown_trigger", SHUTDOWN_TRIGGERS)
def test_persistent_index_persist_skips_native_dump_when_finalizing(
    tmp_path, monkeypatch, finalizing, shutdown_trigger
):
    """persist() must not call engine_proxy.dump during interpreter finalization."""
    del shutdown_trigger  # documents the entry path; behavior is identical
    monkeypatch.setattr(sys, "is_finalizing", lambda: finalizing)

    engine = MagicMock()
    engine.get_update_ts.return_value = 2
    engine.dump.return_value = 2
    idx = _make_persistent_index(tmp_path, engine)
    # get_newest_version is a method on PersistentIndex — stub so not_finalizing
    # path reaches dump without scanning disk for versions.
    monkeypatch.setattr(idx, "get_newest_version", lambda: 1)

    result = idx.persist()

    if finalizing:
        assert result == 0
        engine.dump.assert_not_called()
        engine.get_update_ts.assert_not_called()
    else:
        engine.get_update_ts.assert_called()
        engine.dump.assert_called_once()
        assert result == 2


@pytest.mark.parametrize("finalizing", [True, False], ids=["finalizing", "not_finalizing"])
@pytest.mark.parametrize("shutdown_trigger", SHUTDOWN_TRIGGERS)
def test_persistent_index_close_skips_native_drop_when_finalizing(
    tmp_path, monkeypatch, finalizing, shutdown_trigger
):
    """close() must not call engine_proxy.drop during interpreter finalization."""
    del shutdown_trigger
    monkeypatch.setattr(sys, "is_finalizing", lambda: finalizing)

    engine = MagicMock()
    engine.get_update_ts.return_value = 1
    engine.dump.return_value = 0
    idx = _make_persistent_index(tmp_path, engine)
    monkeypatch.setattr(idx, "get_newest_version", lambda: 1)
    monkeypatch.setattr(idx, "_clean_index", MagicMock())

    idx.close()

    if finalizing:
        engine.drop.assert_not_called()
        engine.dump.assert_not_called()
    else:
        engine.drop.assert_called_once()


@pytest.mark.parametrize("finalizing", [True, False], ids=["finalizing", "not_finalizing"])
@pytest.mark.parametrize("shutdown_trigger", SHUTDOWN_TRIGGERS)
def test_register_index_manage_job_skips_work_when_finalizing(
    monkeypatch, finalizing, shutdown_trigger
):
    """APScheduler self-reschedule must not rebuild/persist/add_job mid-finalization."""
    del shutdown_trigger
    monkeypatch.setattr(sys, "is_finalizing", lambda: finalizing)

    scheduler = MagicMock()
    rebuild = MagicMock()
    persist_all = MagicMock()
    col = _make_local_collection(scheduler, rebuild, persist_all)

    col._register_index_manage_job()

    if finalizing:
        rebuild.assert_not_called()
        persist_all.assert_not_called()
        scheduler.add_job.assert_not_called()
    else:
        rebuild.assert_called_once()
        persist_all.assert_called_once()
        scheduler.add_job.assert_called_once()


@pytest.mark.parametrize("finalizing", [True, False], ids=["finalizing", "not_finalizing"])
@pytest.mark.parametrize("shutdown_trigger", SHUTDOWN_TRIGGERS)
def test_collection_del_skips_close_when_finalizing(
    monkeypatch, caplog, finalizing, shutdown_trigger
):
    """Collection.__del__ must not close underlying ICollection during finalization."""
    del shutdown_trigger
    monkeypatch.setattr(sys, "is_finalizing", lambda: finalizing)

    inner = MagicMock()
    wrapper = _make_collection_wrapper(inner)

    with caplog.at_level(logging.WARNING):
        wrapper.__del__()

    if finalizing:
        inner.close.assert_not_called()
    else:
        inner.close.assert_called_once()
        assert any(
            "not closed explicitly" in rec.message.lower()
            or "closing in __del__" in rec.message.lower()
            for rec in caplog.records
        )
