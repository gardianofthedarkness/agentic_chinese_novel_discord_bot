"""
Qdrant Vector Database Adapter
================================
Wraps qdrant-client for semantic similarity search over novel text.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None


class QdrantAdapter:
    """
    Thin wrapper around qdrant-client for novel vector search.

    Responsibilities:
    - Store text embeddings for events, passages, characters
    - Semantic similarity search
    """

    def __init__(self, config: Any):
        self.config = config
        self.client: Optional[QdrantClient] = None
        self.collection = config.qdrant_collection

        if not QDRANT_AVAILABLE:
            logger.warning("qdrant-client not installed — Qdrant disabled")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        if not QDRANT_AVAILABLE:
            return False
        try:
            self.client = QdrantClient(url=self.config.qdrant_url, timeout=5.0)
            self.client.get_collections()
            logger.info(f"✅ Qdrant connected: {self.config.qdrant_url}")
            return True
        except Exception as e:
            logger.warning(f"Qdrant connection failed: {e}")
            self.client = None
            return False

    def is_connected(self) -> bool:
        return self.client is not None

    def disconnect(self) -> None:
        self.client = None

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self, vector_size: int = 1536) -> None:
        """Create collection if it doesn't exist."""
        if not self.client:
            return
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            if self.collection not in existing:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection: {self.collection}")
        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection: {e}")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, vector: List[float], payload: Dict[str, Any], point_id: Optional[str] = None) -> bool:
        """Store a single vector with payload."""
        if not self.client:
            return False
        try:
            pid = point_id or str(uuid.uuid4())
            self.client.upsert(
                collection_name=self.collection,
                points=[PointStruct(id=pid, vector=vector, payload=payload)],
            )
            return True
        except Exception as e:
            logger.error(f"Qdrant upsert failed: {e}")
            return False

    def upsert_batch(self, items: List[Dict]) -> int:
        """
        Store multiple vectors.

        Each item: {"vector": [...], "payload": {...}, "id": optional str}
        """
        if not self.client or not items:
            return 0
        try:
            points = [
                PointStruct(
                    id=item.get("id") or str(uuid.uuid4()),
                    vector=item["vector"],
                    payload=item.get("payload", {}),
                )
                for item in items
            ]
            self.client.upsert(collection_name=self.collection, points=points)
            return len(points)
        except Exception as e:
            logger.error(f"Qdrant batch upsert failed: {e}")
            return 0

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vector: List[float], limit: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Semantic similarity search. Returns list of payload dicts."""
        if not self.client:
            return []
        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
            return [{"score": r.score, **r.payload} for r in results]
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    def scroll_all(self, limit: int = 100) -> List[Dict]:
        """Return all stored payloads (for small collections)."""
        if not self.client:
            return []
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection,
                limit=limit,
                with_payload=True,
            )
            return [r.payload for r in records]
        except Exception as e:
            logger.error(f"Qdrant scroll failed: {e}")
            return []
