# src/agentops/rag/platform_knowledge.py
"""
Platform Knowledge RAG Module

Xcode ve Gradle ekosistemlerine özgü bilinen hata pattern'larını,
release notes bilgilerini ve platform-specific fix'leri index'ler.

Kullanım alanları:
- Xcode version incompatibility
- Gradle plugin version issues
- iOS signing configuration errors
- Android SDK missing errors
- Apple/Google release notes bilgileri
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


COLLECTION_NAME = "platform_knowledge"
VECTOR_SIZE = 768


@dataclass
class PlatformRecord:
    """
    Platform-specific bir hata/çözüm kaydı.
    """
    title: str                  # Kısa başlık
    error_pattern: str          # Hata mesajı pattern'ı
    explanation: str            # Hatanın açıklaması
    fix: str                    # Çözüm
    platform: str               # ios | android
    tool: str                   # xcode | gradle | fastlane | cocoapods | spm
    tool_version: str           # Hangi versiyonda görüldüğü (opsiyonel, "" olabilir)
    source: str                 # apple_docs | gradle_docs | stackoverflow | manual
    tags: list[str] = field(default_factory=list)

    def to_payload(self) -> dict:
        return {
            "title": self.title,
            "error_pattern": self.error_pattern,
            "explanation": self.explanation,
            "fix": self.fix,
            "platform": self.platform,
            "tool": self.tool,
            "tool_version": self.tool_version,
            "source": self.source,
            "tags": self.tags,
        }

    def get_embed_text(self) -> str:
        return (
            f"TITLE: {self.title}\n"
            f"ERROR: {self.error_pattern}\n"
            f"EXPLANATION: {self.explanation}\n"
            f"FIX: {self.fix}"
        )

    def get_id(self) -> str:
        raw = self.title + self.error_pattern + self.platform
        return str(uuid.UUID(hashlib.md5(raw.encode()).hexdigest()))


@dataclass
class PlatformResult:
    title: str
    explanation: str
    fix: str
    platform: str
    tool: str
    source: str
    score: float
    tags: list[str] = field(default_factory=list)


class PlatformKnowledgeRAG:
    """
    Platform Knowledge RAG modülü.

    Kullanım:
        rag = PlatformKnowledgeRAG()
        rag.ensure_collection()
        results = rag.retrieve(
            "CodeSign error: No signing certificate iOS Distribution",
            platform="ios",
            tool="xcode"
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

    def index(self, record: PlatformRecord) -> str:
        vector = embed_text(record.get_embed_text())
        point_id = record.get_id()
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=vector, payload=record.to_payload())],
        )
        return point_id

    def index_batch(self, records: list[PlatformRecord]) -> list[str]:
        points, ids = [], []
        for record in records:
            vector = embed_text(record.get_embed_text())
            point_id = record.get_id()
            points.append(PointStruct(id=point_id, vector=vector, payload=record.to_payload()))
            ids.append(point_id)
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"✅ {len(points)} platform kaydı index'lendi.")
        return ids

    def retrieve(
        self,
        query: str,
        platform: Optional[str] = None,
        tool: Optional[str] = None,
        top_k: int = 5,
    ) -> list[PlatformResult]:
        self.ensure_collection()
        vector = embed_text(query)

        conditions = []
        if platform:
            conditions.append(FieldCondition(key="platform", match=MatchValue(value=platform)))
        if tool:
            conditions.append(FieldCondition(key="tool", match=MatchValue(value=tool)))

        query_filter = Filter(must=conditions) if conditions else None

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            PlatformResult(
                title=r.payload.get("title", ""),
                explanation=r.payload.get("explanation", ""),
                fix=r.payload.get("fix", ""),
                platform=r.payload.get("platform", ""),
                tool=r.payload.get("tool", ""),
                source=r.payload.get("source", ""),
                score=r.score,
                tags=r.payload.get("tags", []),
            )
            for r in results
        ]

    def load_from_json(self, path: str) -> int:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = [PlatformRecord(**item) for item in data]
        self.index_batch(records)
        return len(records)

    def seed_from_dataset(self, path: str) -> int:
        """
        dataset.jsonl'dan platform-specific hataları okuyup index'ler.
        iOS → xcode/fastlane/swiftlint tool'ları
        Android → gradle/kotlin tool'ları
        """
        PLATFORM_ERROR_TYPES = {
            # iOS / Xcode
            "xcode_build_error":                      ("ios", "xcode"),
            "swiftlint_xcfilelist_missing":           ("ios", "xcode"),
            "archive_failed_swiftformat_assetcatalog":("ios", "xcode"),
            "archive_failed_swiftformat_only":        ("ios", "xcode"),
            "archive_failed_swiftformat_plugin_artifact_missing": ("ios", "xcode"),
            "archive_failed_swiftlint_runner_devtools_invalid_manifest": ("ios", "xcode"),
            "swift_compilation_error":                ("ios", "xcode"),
            "xcode_test_scheme_14_failures_exit_65":  ("ios", "xcode"),
            "xcodebuild_settings_timeout":            ("ios", "xcode"),
            "derived_data_module_cache_corruption":   ("ios", "xcode"),
            "snapshot_test_mismatch":                 ("ios", "xcode"),
            "export_failed":                          ("ios", "xcode"),
            "fastlane_scheme_resolution_failure":     ("ios", "fastlane"),
            # Android / Gradle
            "gradle_daemon_disappeared":              ("android", "gradle"),
            "gradle_daemon_connection_error":         ("android", "gradle"),
            "gradle_r8_hung_no_progress":             ("android", "gradle"),
            "gradle_journal_lock_contention":         ("android", "gradle"),
            "gradle_cache_lock_timeout_concurrent_build": ("android", "gradle"),
            "gradle_daemon_jvm_crash_during_d8_dexing": ("android", "gradle"),
            "agp_classloader_instrumentation_failure":("android", "gradle"),
            "kotlin_compile_error":                   ("android", "kotlin"),
            "kotlin_compile_error_missing_localization_keys": ("android", "kotlin"),
            "kotlin_daemon_crash_on_manual_abort":    ("android", "kotlin"),
            "kapt_compile_error":                     ("android", "kotlin"),
            "manifest_merger_duplicate_deeplink_destination": ("android", "gradle"),
            "unit_test_compile_error":                ("android", "gradle"),
            "compose_parameter_not_found_breaking_api_change": ("android", "kotlin"),
            "android_lint_vital_baseline_created":    ("android", "gradle"),
        }

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                error_type = item.get("metadata", {}).get("error_type", "")
                if error_type not in PLATFORM_ERROR_TYPES:
                    continue

                platform, tool = PLATFORM_ERROR_TYPES[error_type]
                error_msg = item.get("error_message", "") or item.get("log_chunk", "")[:200]
                rca = item.get("ground_truth_rca", "")
                fix_obj = item.get("fix") or {}
                fix_code = fix_obj.get("code", "") if isinstance(fix_obj, dict) else ""

                record = PlatformRecord(
                    title=error_type.replace("_", " ").title(),
                    error_pattern=error_msg[:300],
                    explanation=rca,
                    fix=fix_code or rca,
                    platform=platform,
                    tool=tool,
                    tool_version="",
                    source="jenkins",
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
