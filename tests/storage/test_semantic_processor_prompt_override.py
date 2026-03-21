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


class TestGenerateOverviewCustomPrompt:
    """Tests that _generate_overview uses custom template when set."""

    def _make_semantic_processor(self):
        from openviking.storage.queuefs.semantic_processor import SemanticProcessor

        return SemanticProcessor()

    @pytest.mark.asyncio
    async def test_overview_custom_prompt_bypasses_render_prompt(self):
        """Custom overview_prompt renders via Jinja2, not render_prompt."""
        custom_tpl = "OVERVIEW: {{ dir_name }} FILES: {{ file_summaries }} CHILDREN: {{ children_abstracts }}"
        semantic_cfg = SemanticConfig(overview_prompt=custom_tpl)
        semantic_cfg.max_overview_prompt_chars = 999999
        semantic_cfg.overview_batch_size = 999

        fake_vlm = MagicMock()
        fake_vlm.is_available.return_value = True
        fake_vlm.get_completion_async = AsyncMock(return_value="# MyDir\n\ncustom overview text")

        fake_config = MagicMock()
        fake_config.vlm = fake_vlm
        fake_config.semantic = semantic_cfg

        render_prompt_calls = []

        def fake_render_prompt(prompt_id, variables=None):
            render_prompt_calls.append(prompt_id)
            return "built-in overview"

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
        ):
            await processor._generate_overview(
                "project/mydir",
                [{"name": "a.py", "summary": "does things"}],
                [],
            )

        assert "semantic.overview_generation" not in render_prompt_calls
        call_arg = fake_vlm.get_completion_async.call_args[0][0]
        assert "OVERVIEW:" in call_arg
        assert "mydir" in call_arg

    @pytest.mark.asyncio
    async def test_empty_abstract_warning_logged(self):
        """_generate_overview logs WARNING when abstract would be empty."""
        import logging

        semantic_cfg = SemanticConfig()
        semantic_cfg.max_overview_prompt_chars = 999999
        semantic_cfg.overview_batch_size = 999

        fake_vlm = MagicMock()
        fake_vlm.is_available.return_value = True
        # Only an H1 title with no following paragraph → _extract_abstract returns ""
        fake_vlm.get_completion_async = AsyncMock(return_value="# MyDir\n")

        fake_config = MagicMock()
        fake_config.vlm = fake_vlm
        fake_config.semantic = semantic_cfg

        def fake_render_prompt(prompt_id, variables=None):
            return "prompt text"

        processor = self._make_semantic_processor()

        module_logger = logging.getLogger("openviking.storage.queuefs.semantic_processor")
        warning_messages = []

        class CapturingHandler(logging.Handler):
            def emit(self, record):
                if record.levelno == logging.WARNING:
                    warning_messages.append(record.getMessage())

        handler = CapturingHandler()
        module_logger.addHandler(handler)
        try:
            with (
                patch(
                    "openviking.storage.queuefs.semantic_processor.get_openviking_config",
                    return_value=fake_config,
                ),
                patch(
                    "openviking.storage.queuefs.semantic_processor.render_prompt",
                    side_effect=fake_render_prompt,
                ),
            ):
                await processor._generate_overview(
                    "project/emptydir",
                    [{"name": "x.py", "summary": "something"}],
                    [],
                )
        finally:
            module_logger.removeHandler(handler)

        assert any("abstract" in m.lower() or "emptydir" in m for m in warning_messages), (
            f"Expected warning about empty abstract, got: {warning_messages}"
        )
