# src/agentops/rag/conflict_resolution.py
"""
Conflict Resolution RAG Module

Mobile-specific conflict örneklerini ve çözümlerini index'ler.
Taxonomy tipine göre filtreleme destekler.

Conflict taxonomy:
- spurious:    Gereksiz conflict — otomatik çözülebilir (UUID değişimleri vs.)
- auto_merge:  Git otomatik merge yapabilir ama yapmadı
- syntactic:   Kod yapısı conflict'i — anlam değişmiyor
- semantic:    Mantıksal conflict — dikkatli inceleme gerekiyor

Özellikle mobile-specific dosyalar:
- .pbxproj (iOS Xcode project)
- Podfile / Podfile.lock
- build.gradle / gradle.properties
- AndroidManifest.xml
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .qdrant_client import (
    get_qdrant_client,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    Distance,
)
from .embedder import embed_text


COLLECTION_NAME = "conflict_resolution"
VECTOR_SIZE = 768

CONFLICT_TYPES = {"spurious", "auto_merge", "syntactic", "semantic"}
CONFLICT_FILES = {
    "pbxproj", "podfile", "podfile_lock",
    "build_gradle", "gradle_properties",
    "android_manifest", "other"
}


@dataclass
class ConflictRecord:
    """
    Bir conflict örneği ve çözümü.
    """
    conflict_snippet: str       # Conflict içeren kod bloğu (<<<< HEAD ... >>>>)
    resolution: str             # Nasıl çözüldüğünün açıklaması
    resolved_snippet: str       # Çözülmüş hali
    conflict_type: str          # spurious | auto_merge | syntactic | semantic
    file_type: str              # pbxproj | podfile | podfile_lock | build_gradle vs.
    platform: str               # ios | android | shared
    explanation: str            # Neden bu conflict oluştu
    tags: list[str] = field(default_factory=list)

    def to_payload(self) -> dict:
        return {
            "conflict_snippet": self.conflict_snippet,
            "resolution": self.resolution,
            "resolved_snippet": self.resolved_snippet,
            "conflict_type": self.conflict_type,
            "file_type": self.file_type,
            "platform": self.platform,
            "explanation": self.explanation,
            "tags": self.tags,
        }

    def get_embed_text(self) -> str:
        return (
            f"FILE TYPE: {self.file_type}\n"
            f"CONFLICT TYPE: {self.conflict_type}\n"
            f"CONFLICT:\n{self.conflict_snippet}\n"
            f"RESOLUTION: {self.resolution}"
        )

    def get_id(self) -> str:
        raw = self.conflict_snippet + self.conflict_type + self.file_type
        return str(uuid.UUID(hashlib.md5(raw.encode()).hexdigest()))


@dataclass
class ConflictResult:
    resolution: str
    resolved_snippet: str
    conflict_type: str
    file_type: str
    platform: str
    explanation: str
    score: float
    tags: list[str] = field(default_factory=list)


class ConflictResolutionRAG:
    """
    Conflict Resolution RAG modülü.

    Kullanım:
        rag = ConflictResolutionRAG()
        rag.ensure_collection()
        results = rag.retrieve(
            "<<<<<<< HEAD\\nFILEREF uuid1...\\n=======\\nFILEREF uuid2...\\n>>>>>>>",
            conflict_type="spurious",
            file_type="pbxproj"
        )
    """

    def __init__(
        self,
        qdrant_url: str = "",  # Artık kullanılmıyor, geriye dönük uyumluluk için tutuldu
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.client = get_qdrant_client()
        self.collection_name = collection_name

    def ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            print(f"✅ Collection '{self.collection_name}' oluşturuldu.")
        else:
            print(f"ℹ️  Collection '{self.collection_name}' zaten mevcut.")

    def index(self, record: ConflictRecord) -> str:
        vector = embed_text(record.get_embed_text())
        point_id = record.get_id()
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=vector, payload=record.to_payload())],
        )
        return point_id

    def index_batch(self, records: list[ConflictRecord]) -> list[str]:
        points, ids = [], []
        for record in records:
            vector = embed_text(record.get_embed_text())
            point_id = record.get_id()
            points.append(PointStruct(id=point_id, vector=vector, payload=record.to_payload()))
            ids.append(point_id)
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"✅ {len(points)} conflict kaydı index'lendi.")
        return ids

    def retrieve(
        self,
        query: str,
        conflict_type: Optional[str] = None,
        file_type: Optional[str] = None,
        platform: Optional[str] = None,
        top_k: int = 5,
    ) -> list[ConflictResult]:
        vector = embed_text(query)

        conditions = []
        if conflict_type:
            conditions.append(FieldCondition(key="conflict_type", match=MatchValue(value=conflict_type)))
        if file_type:
            conditions.append(FieldCondition(key="file_type", match=MatchValue(value=file_type)))
        if platform:
            conditions.append(FieldCondition(key="platform", match=MatchValue(value=platform)))

        query_filter = Filter(must=conditions) if conditions else None

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            ConflictResult(
                resolution=r.payload.get("resolution", ""),
                resolved_snippet=r.payload.get("resolved_snippet", ""),
                conflict_type=r.payload.get("conflict_type", ""),
                file_type=r.payload.get("file_type", ""),
                platform=r.payload.get("platform", ""),
                explanation=r.payload.get("explanation", ""),
                score=r.score,
                tags=r.payload.get("tags", []),
            )
            for r in results
        ]

    def load_from_json(self, path: str) -> int:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = [ConflictRecord(**item) for item in data]
        self.index_batch(records)
        return len(records)

    def seed_from_dataset(self, path: str) -> int:
        """
        dataset.jsonl'daki merge_conflict kayıtlarını index'ler.
        conflict_type → error_type'dan tahmin edilir.
        file_type     → log_chunk'taki dosya uzantılarından çıkarılır.
        """
        import re

        CONFLICT_ERROR_TYPES = {
            "merge_conflict", "git_merge_conflict", "git_merge_conflict_pre_build",
            "merge_conflict_mixed_add_add_and_content",
            "merge_conflict_spm_dependency_files",
            "merge_conflict_single_constants_file",
            "merge_conflict_add_add_github_instructions_file",
            "merge_conflict_remote_config_infrastructure",
        }

        def _infer_conflict_type(error_type: str, log: str) -> str:
            if "spurious" in error_type or "pbxproj" in log.lower():
                return "spurious"
            if "add_add" in error_type:
                return "auto_merge"
            if "spm" in error_type or "package" in log.lower():
                return "syntactic"
            return "semantic"

        def _infer_file_type(log: str) -> str:
            log_lower = log.lower()
            if "pbxproj" in log_lower:
                return "pbxproj"
            if "podfile.lock" in log_lower:
                return "podfile_lock"
            if "podfile" in log_lower:
                return "podfile"
            if "build.gradle" in log_lower:
                return "build_gradle"
            if "gradle.properties" in log_lower:
                return "gradle_properties"
            if "androidmanifest" in log_lower:
                return "android_manifest"
            return "other"

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                error_type = item.get("metadata", {}).get("error_type", "")
                if error_type not in CONFLICT_ERROR_TYPES:
                    continue

                log = item.get("log_chunk", "")
                rca = item.get("ground_truth_rca", "")
                fix_obj = item.get("fix") or {}
                fix_code = fix_obj.get("code", "") if isinstance(fix_obj, dict) else ""
                platform = item.get("metadata", {}).get("platform", "ios")

                conflict_type = _infer_conflict_type(error_type, log)
                file_type = _infer_file_type(log)

                # log_chunk'tan conflict snippet'ini çıkar (<<<< ... >>>> bloğu)
                snippet_match = re.search(
                    r"(<<<<<<.*?>>>>>>>.*?)(?:\n(?![\s<=>])|\Z)", log, re.DOTALL
                )
                conflict_snippet = snippet_match.group(1)[:500] if snippet_match else log[:300]

                record = ConflictRecord(
                    conflict_snippet=conflict_snippet,
                    resolution=rca,
                    resolved_snippet=fix_code or "",
                    conflict_type=conflict_type,
                    file_type=file_type,
                    platform=platform,
                    explanation=rca,
                    tags=[item.get("id", ""), error_type],
                )
                records.append(record)

        if records:
            self.index_batch(records)
        return len(records)

    def seed_defaults_if_empty(self, dataset_path: Optional[str] = None) -> int:
        """Collection boşsa dataset'ten seed eder."""
        try:
            current = self.count()
        except Exception:
            self.ensure_collection()
            current = self.count()
        if current > 0:
            return 0
        if dataset_path is None:
            import os
            dataset_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "dataset.jsonl"
            )
        return self.seed_from_dataset(dataset_path)

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name).count
