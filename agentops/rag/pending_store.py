# src/agentops/rag/pending_store.py
"""
Pending Analyses Store

Her analiz sonucunu geçici olarak Qdrant'a yazar.
PR merge olunca feedback_service aracılığıyla historical_fixes'a promote edilir.
Bu koleksiyon RAG retrieve kaynağı DEĞİLDİR — sadece bekleme deposudur.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from .embedder import embed_text


COLLECTION_NAME = "pending_analyses"
VECTOR_SIZE = 768  # nomic-embed-text


@dataclass
class PendingRecord:
    """Bir analiz sonucunu pending olarak saklar."""

    run_id: int
    pipeline_name: str
    branch: str
    platform: str
    commit_sha: str
    main_category: str
    category: str
    root_cause: str
    explanation: str
    suggestion: str
    confidence: float
    failure_chunk: str
    affected_files: list[str] = field(default_factory=list)
    affected_classes: list[str] = field(default_factory=list)
    conflict_type: str = ""
    resolution_strategy: str = ""
    jira_task_id: str = ""
    target_branch: str = ""
    pr_id: str = ""
    status: str = "pending"
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_payload(self) -> dict:
        return {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "branch": self.branch,
            "platform": self.platform,
            "commit_sha": self.commit_sha,
            "main_category": self.main_category,
            "category": self.category,
            "root_cause": self.root_cause,
            "explanation": self.explanation,
            "suggestion": self.suggestion,
            "confidence": self.confidence,
            "failure_chunk": self.failure_chunk,
            "affected_files": self.affected_files,
            "affected_classes": self.affected_classes,
            "conflict_type": self.conflict_type,
            "resolution_strategy": self.resolution_strategy,
            "jira_task_id": self.jira_task_id,
            "target_branch": self.target_branch,
            "pr_id": self.pr_id,
            "status": self.status,
            "created_at": self.created_at,
        }

    def get_embed_text(self) -> str:
        return f"{self.root_cause} {self.suggestion}"

    def get_id(self) -> str:
        """Deterministic UUID from run_id."""
        raw = str(self.run_id)
        return str(uuid.UUID(hashlib.md5(raw.encode()).hexdigest()))


class PendingStore:
    """
    Pending analyses Qdrant koleksiyonu yönetimi.

    Kullanım:
        store = PendingStore()
        store.ensure_collection()
        store.write(PendingRecord(...))
        record = store.get_by_run_id(1846)
        store.delete_by_run_id(1846)
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        try:
            self.client = QdrantClient(url=qdrant_url, check_compatibility=False)
        except TypeError:
            self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name

    def ensure_collection(self) -> None:
        """Collection yoksa oluşturur."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            print(f"✅ Collection '{self.collection_name}' oluşturuldu.")

    def write(self, record: PendingRecord) -> str:
        """Analiz sonucunu pending olarak yazar. Aynı run_id varsa atlar."""
        self.ensure_collection()

        # Duplicate kontrolü — aynı run_id varsa yazma
        existing = self.get_by_run_id(record.run_id)
        if existing is not None:
            point_id = record.get_id()
            print(f"ℹ️ run_id={record.run_id} zaten pending'de, atlanıyor.")
            return point_id

        vector = embed_text(record.get_embed_text())
        point_id = record.get_id()

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=record.to_payload(),
                )
            ],
        )
        return point_id

    def get_by_run_id(self, run_id: int) -> Optional[PendingRecord]:
        """run_id ile pending kaydı bul."""
        self.ensure_collection()
        filt = Filter(
            must=[FieldCondition(key="run_id", match=MatchValue(value=run_id))]
        )
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filt,
                limit=1,
                with_payload=True,
            )
            points = results[0] if isinstance(results, tuple) else results
        except Exception:
            return None

        if not points:
            return None

        p = points[0].payload
        return PendingRecord(
            run_id=p["run_id"],
            pipeline_name=p.get("pipeline_name", ""),
            branch=p.get("branch", ""),
            platform=p.get("platform", ""),
            commit_sha=p.get("commit_sha", ""),
            main_category=p.get("main_category", ""),
            category=p.get("category", ""),
            root_cause=p.get("root_cause", ""),
            explanation=p.get("explanation", ""),
            suggestion=p.get("suggestion", ""),
            confidence=p.get("confidence", 0.0),
            failure_chunk=p.get("failure_chunk", ""),
            affected_files=p.get("affected_files", []),
            affected_classes=p.get("affected_classes", []),
            conflict_type=p.get("conflict_type", ""),
            resolution_strategy=p.get("resolution_strategy", ""),
            jira_task_id=p.get("jira_task_id", ""),
            target_branch=p.get("target_branch", ""),
            pr_id=p.get("pr_id", ""),
            status=p.get("status", "pending"),
            created_at=p.get("created_at", ""),
        )

    def delete_by_run_id(self, run_id: int) -> None:
        """run_id ile pending kaydını sil."""
        self.ensure_collection()
        filt = Filter(
            must=[FieldCondition(key="run_id", match=MatchValue(value=run_id))]
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=filt,
        )

    def list_pending(self, limit: int = 20) -> list[dict]:
        """Tüm pending kayıtları listele."""
        self.ensure_collection()
        filt = Filter(
            must=[FieldCondition(key="status", match=MatchValue(value="pending"))]
        )
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filt,
                limit=limit,
                with_payload=True,
            )
            points = results[0] if isinstance(results, tuple) else results
        except Exception:
            return []

        return [p.payload for p in points]

    def count(self) -> int:
        """Collection'daki toplam kayıt sayısı."""
        self.ensure_collection()
        return self.client.count(collection_name=self.collection_name).count
