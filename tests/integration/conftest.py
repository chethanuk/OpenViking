# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for integration tests.

Automatically starts an OpenViking server in a background thread so that
AsyncHTTPClient integration tests can run without a manually started server process.
"""

import math
import os
import shutil
import socket
import threading
import time
from pathlib import Path
from typing import Optional

import httpx
import pytest
import pytest_asyncio
import uvicorn

from openviking.server.app import create_app
from openviking.server.config import ServerConfig
from openviking.service.core import OpenVikingService
from openviking_cli.session.user_id import UserIdentifier

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_TMP_DIR = PROJECT_ROOT / "test_data" / "tmp_integration"


@pytest.fixture(scope="session")
def temp_dir():
    """Create temp directory for the whole test session."""
    shutil.rmtree(TEST_TMP_DIR, ignore_errors=True)
    TEST_TMP_DIR.mkdir(parents=True, exist_ok=True)
    yield TEST_TMP_DIR


@pytest.fixture(scope="session")
def server_url(temp_dir):
    """Start a real uvicorn server in a background thread.

    Returns the base URL (e.g. ``http://127.0.0.1:<port>``).
    The server is automatically shut down after the test session.
    """
    import asyncio

    loop = asyncio.new_event_loop()

    svc = OpenVikingService(
        path=str(temp_dir / "data"), user=UserIdentifier.the_default_user("test_user")
    )
    loop.run_until_complete(svc.initialize())

    config = ServerConfig()
    fastapi_app = create_app(config=config, service=svc)

    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    uvi_config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(uvi_config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server ready
    url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            r = httpx.get(f"{url}/health", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)

    yield url

    server.should_exit = True
    thread.join(timeout=5)
    loop.run_until_complete(svc.close())
    loop.close()


# ── Gemini shared fixtures and helpers ────────────────────────────────────────

GOOGLE_API_KEY: Optional[str] = os.environ.get("GOOGLE_API_KEY")

# (model_name, default_dim, token_limit) — for @pytest.mark.parametrize("model,dim,limit", ...)
GEMINI_MODELS = [
    pytest.param("gemini-embedding-2-preview", 3072, 8192, id="g2p"),
    pytest.param("gemini-embedding-001",       3072, 2048, id="g001"),
]

# Wrapped single-value tuples — for fixture params (request.param is the whole tuple)
GEMINI_MODELS_FIXTURE = [
    pytest.param(("gemini-embedding-2-preview", 3072, 8192), id="g2p"),
    pytest.param(("gemini-embedding-001",       3072, 2048), id="g001"),
]

# (model_name, dimension) pairs for OpenViking client fixtures
EMBED_PARAMS = [
    pytest.param(("gemini-embedding-2-preview", 512),  id="g2p-512"),
    pytest.param(("gemini-embedding-2-preview", 768),  id="g2p-768"),
    pytest.param(("gemini-embedding-2-preview", 1536), id="g2p-1536"),
    pytest.param(("gemini-embedding-2-preview", 3072), id="g2p-3072"),
    pytest.param(("gemini-embedding-001",       768),  id="g001-768"),
]


def l2_norm(vec) -> float:
    return math.sqrt(sum(v * v for v in vec))


def vectordb_engine_available() -> bool:
    try:
        from openviking.storage.vectordb.engine import PersistStore, VolatileStore
        return isinstance(PersistStore, type) and isinstance(VolatileStore, type)
    except Exception:
        return False


def sample_markdown(tmp_dir: Path, slug: str, content: str) -> Path:
    p = tmp_dir / f"{slug}.md"
    p.write_text(content, encoding="utf-8")
    return p


def gemini_config_dict(
    model: str, dim: int,
    query_param: Optional[str] = None,
    doc_param: Optional[str] = None,
    task_type: Optional[str] = None,
) -> dict:
    dense: dict = {"provider": "gemini", "model": model, "api_key": GOOGLE_API_KEY, "dimension": dim}
    if query_param:
        dense["query_param"] = query_param
    if doc_param:
        dense["document_param"] = doc_param
    if task_type:
        dense["task_type"] = task_type
    return {"embedding": {"dense": dense}, "storage": {"agfs": {"mode": "binding-client"}}}


async def make_ov_client(config_dict: dict, data_path: str):
    from openviking.async_client import AsyncOpenViking
    from openviking_cli.utils.config.open_viking_config import OpenVikingConfigSingleton
    await AsyncOpenViking.reset()
    OpenVikingConfigSingleton.reset_instance()
    OpenVikingConfigSingleton.initialize(config_dict=config_dict)
    client = AsyncOpenViking(path=data_path)
    await client.initialize()
    return client


async def teardown_ov_client():
    from openviking.async_client import AsyncOpenViking
    from openviking_cli.utils.config.open_viking_config import OpenVikingConfigSingleton
    await AsyncOpenViking.reset()
    OpenVikingConfigSingleton.reset_instance()


requires_api_key = pytest.mark.skipif(not GOOGLE_API_KEY, reason="GOOGLE_API_KEY not set")
requires_engine = pytest.mark.skipif(
    not vectordb_engine_available(),
    reason="VectorDB native engine not compiled — run: pip install -e . --no-build-isolation",
)


@pytest.fixture(scope="module", params=GEMINI_MODELS_FIXTURE)
def gemini_embedder(request):
    """Module-scoped GeminiDenseEmbedder at dim=768, parametrized over known models."""
    from openviking.models.embedder.gemini_embedders import GeminiDenseEmbedder
    model_name, _, _ = request.param
    return GeminiDenseEmbedder(model_name, api_key=GOOGLE_API_KEY, dimension=768)


@pytest_asyncio.fixture(params=EMBED_PARAMS)
async def gemini_ov_client(request, tmp_path):
    """AsyncOpenViking client backed by Gemini; yields (client, model, dim)."""
    model, dim = request.param
    data_path = str(tmp_path / "ov_data")
    Path(data_path).mkdir(parents=True, exist_ok=True)

    client = await make_ov_client(gemini_config_dict(model, dim), data_path)
    yield client, model, dim
    await teardown_ov_client()
