# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for custom prompt template override in SemanticProcessor (#578)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openviking_cli.utils.config.parser_config import SemanticConfig


class TestGenerateTextSummaryCustomPrompt:
    """Tests that _generate_text_summary uses custom templates when set."""

    def _make_semantic_processor(self):
        from openviking.storage.queuefs.semantic_processor import SemanticProcessor

        return SemanticProcessor()

    @pytest.mark.asyncio
    async def test_document_custom_prompt_bypasses_render_prompt(self):
        """Custom document_summary_prompt renders via Jinja2, not render_prompt."""
        custom_tpl = "DOMAIN: {{ file_name }} :: {{ content }}"
        semantic_cfg = SemanticConfig(document_summary_prompt=custom_tpl)

        fake_vlm = MagicMock()
        fake_vlm.is_available.return_value = True
        fake_vlm.get_completion_async = AsyncMock(return_value="custom summary")

        fake_config = MagicMock()
        fake_config.vlm = fake_vlm
        fake_config.semantic = semantic_cfg
        fake_config.code.code_summary_mode = "llm"

        render_prompt_calls = []

        def fake_render_prompt(prompt_id, variables=None):
            render_prompt_calls.append(prompt_id)
            return "built-in prompt"

        processor = self._make_semantic_processor()

        with (
            patch(
                "openviking.storage.queuefs.semantic_processor.get_openviking_config",
                return_value=fake_config,
            ),
            patch(
                "openviking.storage.queuefs.semantic_processor.render_prompt",
                side_effect=fake_render_prompt,
            ),
            patch("openviking.storage.queuefs.semantic_processor.get_viking_fs") as mock_fs,
        ):
            mock_fs.return_value.read_file = AsyncMock(return_value="file content here")
            sem = asyncio.Semaphore(1)
            result = await processor._generate_text_summary("/docs/readme.md", "readme.md", sem)

        # render_prompt must NOT have been called for document_summary
        assert "semantic.document_summary" not in render_prompt_calls
        # VLM received the rendered custom template (not built-in)
        call_arg = fake_vlm.get_completion_async.call_args[0][0]
        assert "DOMAIN:" in call_arg
        assert "readme.md" in call_arg
        assert result == {"name": "readme.md", "summary": "custom summary"}

    @pytest.mark.asyncio
    async def test_no_custom_prompt_uses_render_prompt(self):
        """When custom fields are None, render_prompt IS called with built-in ID."""
        semantic_cfg = SemanticConfig()  # all None

        fake_vlm = MagicMock()
        fake_vlm.is_available.return_value = True
        fake_vlm.get_completion_async = AsyncMock(return_value="built-in summary")

        fake_config = MagicMock()
        fake_config.vlm = fake_vlm
        fake_config.semantic = semantic_cfg
        fake_config.code.code_summary_mode = "llm"

        render_prompt_calls = []

        def fake_render_prompt(prompt_id, variables=None):
            render_prompt_calls.append(prompt_id)
            return "built-in prompt text"

        processor = self._make_semantic_processor()

        with (
            patch(
                "openviking.storage.queuefs.semantic_processor.get_openviking_config",
                return_value=fake_config,
            ),
            patch(
                "openviking.storage.queuefs.semantic_processor.render_prompt",
                side_effect=fake_render_prompt,
            ),
            patch("openviking.storage.queuefs.semantic_processor.get_viking_fs") as mock_fs,
        ):
            mock_fs.return_value.read_file = AsyncMock(return_value="file content")
            sem = asyncio.Semaphore(1)
            # readme.md → documentation type
            await processor._generate_text_summary("/docs/readme.md", "readme.md", sem)

        assert "semantic.document_summary" in render_prompt_calls
