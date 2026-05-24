# src/agentops/rag/qdrant_client.py
"""
Qdrant Vector Client — Wrapper

Gerçek qdrant-client kütüphanesini kullanır.
Bağlantı ortam değişkenlerinden okunur:
  QDRANT_URL      → Qdrant sunucu adresi   (varsayılan: http://localhost:6333)
  QDRANT_API_KEY  → Cloud/auth token       (opsiyonel)

Yerel geliştirme için Docker ile hızlıca başlatabilirsiniz:
  docker run -p 6333:6333 qdrant/qdrant
"""

from __future__ import annotations

import os
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    CountResult,
)

__all__ = [
    "QdrantClient",
    "get_qdrant_client",
    "VectorParams",
    "PointStruct",
    "Filter",
    "FieldCondition",
    "MatchValue",
    "Distance",
    "CountResult",
]


def get_qdrant_client() -> QdrantClient:
    """
    Ortam değişkenlerinden Qdrant bağlantısı oluşturur.

    Öncelik sırası:
      1. QDRANT_URL env var
      2. Varsayılan: http://localhost:6333

    API key varsa (Qdrant Cloud vb.) QDRANT_API_KEY env var ile verilir.
    """
    url     = os.getenv("QDRANT_URL",    "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY", None)

    if api_key:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(url=url)
