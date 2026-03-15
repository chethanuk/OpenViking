# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tenant-field backfill tests for EmbeddingMsgConverter."""

import pytest

from openviking.core.context import Context, Vectorize
from openviking.storage.queuefs.embedding_msg_converter import EmbeddingMsgConverter
from openviking_cli.session.user_id import UserIdentifier


@pytest.mark.parametrize(
    ("uri", "expected_space"),
    [
        (
            "viking://user/memories/preferences/me.md",
            lambda user: user.user_space_name(),
        ),
        (
            "viking://agent/memories/cases/me.md",
            lambda user: user.agent_space_name(),
        ),
        (
            "viking://resources/doc.md",
            lambda _user: "",
        ),
    ],
)
def test_embedding_msg_converter_backfills_account_and_owner_space(uri, expected_space):
    user = UserIdentifier("acme", "alice", "helper")
    context = Context(uri=uri, abstract="hello", user=user)

    # Simulate legacy producer that forgot tenant fields.
    context.account_id = ""
    context.owner_space = ""

    msg = EmbeddingMsgConverter.from_context(context)

    assert msg is not None
    assert msg.context_data["account_id"] == "acme"
    assert msg.context_data["owner_space"] == expected_space(user)


def test_converter_text_only_vectorize():
    """Converter produces a valid message for text-only content."""
    context = Context(uri="viking://agent/memories/notes.md", abstract="some note")
    context.set_vectorize(Vectorize(text="some note text"))

    msg = EmbeddingMsgConverter.from_context(context)

    assert msg is not None
    assert msg.message == "some note text"


def test_converter_returns_none_for_empty_text():
    """Converter returns None when there is no vectorizable text."""
    context = Context(uri="viking://agent/resources/img.png", abstract="")
    context.set_vectorize(Vectorize(text=""))

    msg = EmbeddingMsgConverter.from_context(context)

    assert msg is None
