# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tests for _infer_image_mime helper."""
import pytest


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("photo.jpg", "image/jpeg"),
        ("photo.jpeg", "image/jpeg"),
        ("photo.JPG", "image/jpeg"),
        ("screenshot.png", "image/png"),
        ("anim.gif", "image/gif"),
        ("hero.webp", "image/webp"),
        ("icon.bmp", "image/bmp"),
        ("logo.svg", "image/svg+xml"),
        ("unknown.tiff", None),
        ("noext", None),
    ],
)
def test_infer_image_mime(filename, expected):
    from openviking.utils.embedding_utils import _infer_image_mime
    assert _infer_image_mime(filename) == expected
