# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import hashlib
import json
import traceback
from typing import Any, Dict, Optional

from openviking.models.embedder.base import EmbedResult
from openviking.storage.errors import CollectionNotFoundError
from openviking.storage.queuefs.embedding_msg import EmbeddingMsg
from openviking.storage.queuefs.named_queue import DequeueHandlerBase
from openviking.storage.viking_vector_index_backend import VikingVectorIndexBackend
from openviking_cli.utils import get_logger
from openviking_cli.utils.config import get_openviking_config
from openviking_cli.utils.config.open_viking_config import OpenVikingConfig

logger = get_logger(__name__)


class EmbeddingHandler(DequeueHandlerBase):
    """
    Embedding handler that converts messages to embedding vectors
    and writes results to the vector database.

    Handles both text-only and multimodal (image/audio/video/PDF) embedding
    based on the configured embedder's capabilities. Reads media bytes from
    viking_fs at dequeue time for multimodal paths.

    Supports both dense and sparse embeddings based on configuration.
    """

    def __init__(self, vikingdb: VikingVectorIndexBackend):
        """Initialize the embedding handler.

        Args:
            vikingdb: VikingVectorIndexBackend instance for writing to vector database
        """
        self._vikingdb = vikingdb
        self._embedder = None
        config = get_openviking_config()
        self._collection_name = config.storage.vectordb.name
        self._vector_dim = config.embedding.dimension
        self._initialize_embedder(config)

    def _initialize_embedder(self, config: "OpenVikingConfig"):
        """Initialize the embedder instance from config."""
        self._embedder = config.embedding.get_embedder()

    @staticmethod
    def _seed_uri_for_id(uri: str, level: Any) -> str:
        """Build deterministic id seed URI from canonical uri + hierarchy level."""
        try:
            level_int = int(level)
        except (TypeError, ValueError):
            level_int = 2

        if level_int == 0:
            return uri if uri.endswith("/.abstract.md") else f"{uri}/.abstract.md"
        if level_int == 1:
            return uri if uri.endswith("/.overview.md") else f"{uri}/.overview.md"
        return uri

    async def on_dequeue(self, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Process dequeued message and add embedding vector(s)."""
        if not data:
            return None

        try:
            queue_data = json.loads(data["data"])
            # Parse EmbeddingMsg from data
            embedding_msg = EmbeddingMsg.from_dict(queue_data)
            inserted_data = embedding_msg.context_data

            if self._vikingdb.is_closing:
                logger.debug("Skip embedding dequeue during shutdown")
                self.report_success()
                return None

            # Only process string messages
            if not isinstance(embedding_msg.message, str):
                logger.debug(f"Skipping non-string message type: {type(embedding_msg.message)}")
                self.report_success()
                return data

            # Initialize embedder if not already initialized
            if not self._embedder:
                config = get_openviking_config()
                self._initialize_embedder(config)

            # Generate embedding vector(s)
            if self._embedder:
                # Multimodal path: read bytes from viking_fs and call embed_multimodal()
                if (
                    embedding_msg.media_uri
                    and embedding_msg.media_mime_type
                    and getattr(self._embedder, "supports_multimodal", False)
                ):
                    # Security: validate media_uri matches the record's own URI to prevent
                    # forged queue messages from reading arbitrary files.
                    expected_uri = embedding_msg.context_data.get("uri", "")
                    if embedding_msg.media_uri != expected_uri:
                        logger.warning(
                            f"media_uri {embedding_msg.media_uri!r} does not match context uri "
                            f"{expected_uri!r}, falling back to text embedding"
                        )
                        result: EmbedResult = await asyncio.to_thread(
                            self._embedder.embed, embedding_msg.message
                        )
                    else:
                        # TODO(security): reconstruct a tenant-scoped RequestContext from
                        # context_data["account_id"] to prevent ROOT-context file reads.
                        # Blocked on UserIdentifier requiring user_id/agent_id fields that
                        # are not currently propagated through EmbeddingMsg.context_data.
                        try:
                            from openviking.core.context import ModalContent, Vectorize
                            from openviking.storage.viking_fs import get_viking_fs

                            viking_fs = get_viking_fs()
                            # read_file_bytes is async — await directly (not asyncio.to_thread)
                            raw_bytes = await viking_fs.read_file_bytes(embedding_msg.media_uri, ctx=None)
                            vectorize = Vectorize(
                                text=embedding_msg.message,
                                media=ModalContent(
                                    mime_type=embedding_msg.media_mime_type,
                                    uri=embedding_msg.media_uri,
                                    data=raw_bytes,
                                ),
                            )
                            result: EmbedResult = await asyncio.to_thread(
                                self._embedder.embed_multimodal, vectorize
                            )
                        except Exception as e:
                            logger.warning(
                                f"Multimodal embed failed for {embedding_msg.media_uri!r}: {e}, "
                                "falling back to text embedding"
                            )
                            result: EmbedResult = await asyncio.to_thread(
                                self._embedder.embed, embedding_msg.message
                            )
                else:
                    # embed() is a blocking HTTP call; offload to thread pool to avoid
                    # blocking the event loop and allow real concurrency.
                    result: EmbedResult = await asyncio.to_thread(
                        self._embedder.embed, embedding_msg.message
                    )

                # Add dense vector
                if result.dense_vector:
                    inserted_data["vector"] = result.dense_vector
                    # Validate vector dimension
                    if len(result.dense_vector) != self._vector_dim:
                        error_msg = f"Dense vector dimension mismatch: expected {self._vector_dim}, got {len(result.dense_vector)}"
                        logger.error(error_msg)
                        self.report_error(error_msg, data)
                        return None

                # Add sparse vector if present
                if result.sparse_vector:
                    inserted_data["sparse_vector"] = result.sparse_vector
                    logger.debug(f"Generated sparse vector with {len(result.sparse_vector)} terms")
            else:
                error_msg = "Embedder not initialized, skipping vector generation"
                logger.warning(error_msg)
                self.report_error(error_msg, data)
                return None

            # Write to vector database
            try:
                # Ensure vector DB has deterministic IDs per semantic layer.
                uri = inserted_data.get("uri")
                if uri:
                    account_id = inserted_data.get("account_id", "default")
                    seed_uri = self._seed_uri_for_id(uri, inserted_data.get("level", 2))
                    id_seed = f"{account_id}:{seed_uri}"
                    inserted_data["id"] = hashlib.md5(id_seed.encode("utf-8")).hexdigest()

                record_id = await self._vikingdb.upsert(inserted_data)
                if record_id:
                    logger.debug(
                        f"Successfully wrote embedding to database: {record_id} abstract {inserted_data['abstract']} vector {inserted_data['vector'][:5]}"
                    )
            except CollectionNotFoundError as db_err:
                # During shutdown, queue workers may finish one dequeued item.
                if self._vikingdb.is_closing:
                    logger.debug(f"Skip embedding write during shutdown: {db_err}")
                    self.report_success()
                    return None
                logger.error(f"Failed to write to vector database: {db_err}")
                self.report_error(str(db_err), data)
                return None
            except Exception as db_err:
                if self._vikingdb.is_closing:
                    logger.debug(f"Skip embedding write during shutdown: {db_err}")
                    self.report_success()
                    return None
                logger.error(f"Failed to write to vector database: {db_err}")
                traceback.print_exc()
                self.report_error(str(db_err), data)
                return None

            self.report_success()
            return inserted_data

        except Exception as e:
            logger.error(f"Error processing embedding message: {e}")
            traceback.print_exc()
            self.report_error(str(e), data)
            return None
