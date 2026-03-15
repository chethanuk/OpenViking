# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Vectorize.get_parts() multi-part content support."""
import pytest
from openviking.core.context import ModalContent, Vectorize


def test_get_parts_legacy_text_only():
    """Legacy Vectorize(text=...) returns [text]."""
    v = Vectorize(text="hello world")
    parts = v.get_parts()
    assert parts == ["hello world"]


def test_get_parts_legacy_text_and_media():
    """Legacy Vectorize(text=..., media=...) returns [text, media]."""
    media = ModalContent(mime_type="image/png", uri="img.png", data=b"\x89PNG")
    v = Vectorize(text="caption", media=media)
    parts = v.get_parts()
    assert parts == ["caption", media]


def test_get_parts_explicit_parts_overrides_legacy():
    """Explicit parts= takes precedence over text/media fields."""
    media = ModalContent(mime_type="image/jpeg", uri="photo.jpg", data=b"jpg")
    v = Vectorize(text="ignored", media=media, parts=["Section A", media, "Section B"])
    parts = v.get_parts()
    assert len(parts) == 3
    assert parts[0] == "Section A"
    assert parts[1] is media
    assert parts[2] == "Section B"


def test_get_parts_empty_text_omitted():
    """Empty text string is omitted from parts list."""
    v = Vectorize(text="")
    parts = v.get_parts()
    assert parts == []


def test_get_parts_explicit_empty_list():
    """Explicit parts=[] returns empty list (not legacy fallback)."""
    v = Vectorize(text="some text", parts=[])
    parts = v.get_parts()
    assert parts == []
