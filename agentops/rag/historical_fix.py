# src/agentops/rag/historical_fix.py
"""
Historical Fix RAG Module

Geçmiş failure → fix pairs'i Qdrant'a index'ler ve sorgular.
Her failure kategorisi bu modülü kullanır.

Knowledge base kaynakları:
- Şirket Jenkins logları + fix commit'leri
- GitHub Actions public mobile failure logs
- Manuel olarak eklenen failure-fix pair'leri
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


COLLECTION_NAME = "historical_fixes"
VECTOR_SIZE = 768  # nomic-embed-text


@dataclass
class FixRecord:
    """
    Bir failure → fix pair'ini temsil eder.
    """
    failure_log: str          # Ham log veya log özeti
    fix_description: str      # Fix'in açıklaması
    fix_diff: str             # Unified diff (varsa)
    category: str             # Ana kategori (build_artifact, dependency, vs.)
    platform: str             # ios | android | shared
    source: str               # jenkins | github_actions | manual
    tags: list[str] = field(default_factory=list)

    def to_payload(self) -> dict:
        return {
            "failure_log": self.failure_log,
            "fix_description": self.fix_description,
            "fix_diff": self.fix_diff,
            "category": self.category,
            "platform": self.platform,
            "source": self.source,
            "tags": self.tags,
        }

    def get_embed_text(self) -> str:
        """
        Index'lenecek metin: failure log + fix description birleşimi.
        Semantik arama bu metin üzerinden yapılır.
        """
        return f"FAILURE:\n{self.failure_log}\n\nFIX:\n{self.fix_description}"

    def get_id(self) -> str:
        """
        Deterministic UUID — aynı log+fix pair tekrar index'lenmez.
        """
        raw = self.failure_log + self.fix_description
        return str(uuid.UUID(hashlib.md5(raw.encode()).hexdigest()))


@dataclass
class FixResult:
    """
    RAG sorgusunun döndürdüğü bir sonuç.
    """
    fix_description: str
    fix_diff: str
    category: str
    platform: str
    source: str
    score: float
    tags: list[str] = field(default_factory=list)


class HistoricalFixRAG:
    """
    Historical Fix RAG modülü.

    Kullanım:
        rag = HistoricalFixRAG()
        rag.ensure_collection()

        # Index'le
        rag.index(FixRecord(...))

        # Sorgula
        results = rag.retrieve("pod install failed", category="dependency", top_k=3)
    """

    def __init__(
        self,
        qdrant_url: str = "",  # Artık kullanılmıyor, geriye dönük uyumluluk için tutuldu
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.client = get_qdrant_client()
        self.collection_name = collection_name

    # ------------------------------------------------------------------
    # Collection yönetimi
    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        """Collection yoksa oluşturur, varsa dokunmaz."""
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
        else:
            print(f"ℹ️  Collection '{self.collection_name}' zaten mevcut.")

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def index(self, record: FixRecord) -> str:
        """
        Tek bir FixRecord'u Qdrant'a ekler.
        Aynı ID zaten varsa üzerine yazar (upsert).
        Returns: point ID
        """
        text = record.get_embed_text()
        vector = embed_text(text)
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

    def index_batch(self, records: list[FixRecord]) -> list[str]:
        """
        Birden fazla FixRecord'u batch olarak index'ler.
        """
        points = []
        ids = []

        for record in records:
            text = record.get_embed_text()
            vector = embed_text(text)
            point_id = record.get_id()

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=record.to_payload(),
                )
            )
            ids.append(point_id)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        print(f"✅ {len(points)} kayıt index'lendi.")
        return ids

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        platform: Optional[str] = None,
        top_k: int = 5,
    ) -> list[FixResult]:
        """
        Verilen query'ye en yakın fix'leri döndürür.

        Args:
            query: Failure log veya hata mesajı
            category: Filtre — sadece bu kategoriden getir (opsiyonel)
            platform: Filtre — ios | android | shared (opsiyonel)
            top_k: Kaç sonuç döndürülsün
        """
        vector = embed_text(query)

        # Filtre oluştur
        conditions = []
        if category:
            conditions.append(
                FieldCondition(key="category", match=MatchValue(value=category))
            )
        if platform:
            conditions.append(
                FieldCondition(key="platform", match=MatchValue(value=platform))
            )

        query_filter = Filter(must=conditions) if conditions else None

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            FixResult(
                fix_description=r.payload.get("fix_description", ""),
                fix_diff=r.payload.get("fix_diff", ""),
                category=r.payload.get("category", ""),
                platform=r.payload.get("platform", ""),
                source=r.payload.get("source", ""),
                score=r.score,
                tags=r.payload.get("tags", []),
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Yardımcı
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Collection'daki toplam kayıt sayısı."""
        return self.client.count(collection_name=self.collection_name).count

    def load_from_json(self, path: str) -> int:
        """
        JSON dosyasından toplu index.
        Beklenen format:
        [
          {
            "failure_log": "...",
            "fix_description": "...",
            "fix_diff": "...",
            "category": "dependency",
            "platform": "ios",
            "source": "jenkins",
            "tags": ["cocoapods", "version"]
          },
          ...
        ]
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = [FixRecord(**item) for item in data]
        self.index_batch(records)
        return len(records)

    def load_from_jsonl(self, path: str) -> int:
        """
        JSONL dosyasından toplu index.
        Her satır dataset.jsonl formatında bir kayıt içerir.

        dataset.jsonl → FixRecord dönüşümü:
          log_chunk          → failure_log
          expected_response  → fix_description
          fix.code           → fix_diff
          metadata.platform  → platform
          metadata.error_type→ category (main_category'e map edilir)
          "jenkins"          → source (sabit)
          id                 → tags[0]
        """
        from ..log_models import MAIN_CATEGORIES

        MAIN_CATEGORY_MAP = {
            # Resource / network
            "git_fetch_timeout":                            "resource",
            "git_clone_timeout":                            "resource",
            "git_fetch_disconnect_during_large_clone":      "resource",
            "disk_full":                                    "resource",
            "agent_connection_drop":                        "resource",
            "agent_channel_disconnected":                   "resource",
            "upload_network_failure":                       "resource",
            "curl_connection_timeout_internal_api":         "resource",
            "slack_notification_dns_failure":               "resource",
            "upload_server_rejection":                      "resource",
            "upload_distro_api_nil_response":               "resource",
            "upload_response_empty_json_array":             "resource",
            "jenkins_agent_offline_cleanup_plus_ai_review_http500": "resource",
            # Dependency / package management
            "dependency_error":                             "dependency",
            "cocoapods_error":                              "dependency",
            "spm_error":                                    "dependency",
            "spm_dns_resolution_failure":                   "dependency",
            "spm_dns_resolution_failure_inside_gym":        "dependency",
            "spm_dns_resolution_failure_public_github":     "dependency",
            "spm_resolution_failure":                       "dependency",
            "spm_package_resolution_network_dns_failure":   "dependency",
            "spm_submodule_and_binary_artifact_resolution_failure": "dependency",
            "unresolved_reference_missing_dependency":      "dependency",
            # Merge conflict
            "merge_conflict":                               "merge_conflict",
            "git_merge_conflict":                           "merge_conflict",
            "git_merge_conflict_pre_build":                 "merge_conflict",
            "merge_conflict_mixed_add_add_and_content":     "merge_conflict",
            "merge_conflict_spm_dependency_files":          "merge_conflict",
            "merge_conflict_single_constants_file":         "merge_conflict",
            "merge_conflict_add_add_github_instructions_file": "merge_conflict",
            "merge_conflict_remote_config_infrastructure":  "merge_conflict",
            # Signing / auth
            "codesign_error":                               "signing",
            "authentication_error":                         "authentication",
            "credential_error":                             "authentication",
            "spm_credential_error":                         "authentication",
            # Build artifact (her şey için fallback + explicit tipler)
            "xcfilelist_error":                             "build_artifact",
            "build_error":                                  "build_artifact",
            "xcode_build_error":                            "build_artifact",
            "swiftlint_xcfilelist_missing":                 "build_artifact",
            "swift_compilation_error":                      "build_artifact",
            "gradle_daemon_disappeared":                    "build_artifact",
            "gradle_daemon_connection_error":               "build_artifact",
            "gradle_r8_hung_no_progress":                   "build_artifact",
            "gradle_journal_lock_contention":               "build_artifact",
            "gradle_cache_lock_timeout_concurrent_build":   "build_artifact",
            "gradle_daemon_jvm_crash_during_d8_dexing":     "build_artifact",
            "agp_classloader_instrumentation_failure":      "build_artifact",
            "kotlin_compile_error":                         "build_artifact",
            "kotlin_compile_error_missing_localization_keys": "build_artifact",
            "kotlin_daemon_crash_on_manual_abort":          "build_artifact",
            "kapt_compile_error":                           "build_artifact",
            "unit_test_compile_error":                      "build_artifact",
            "unit_test_compilation_failure_sonar_quality_gate_unknown": "build_artifact",
            "unit_test_kotlin_compilation_failure_non_blocking": "build_artifact",
            "compose_parameter_not_found_breaking_api_change": "build_artifact",
            "derived_data_module_cache_corruption":         "build_artifact",
            "archive_failed_swiftformat_assetcatalog":      "build_artifact",
            "archive_failed_swiftformat_only":              "build_artifact",
            "archive_failed_swiftformat_plugin_artifact_missing": "build_artifact",
            "archive_failed_swiftlint_runner_devtools_invalid_manifest": "build_artifact",
            "export_failed":                                "build_artifact",
            "xcodebuild_settings_timeout":                  "build_artifact",
            "manifest_merger_duplicate_deeplink_destination": "build_artifact",
            "fastlane_scheme_resolution_failure":           "build_artifact",
            "fastlane_versioning_plugin_version_extraction_failed": "build_artifact",
            "jenkinsfile_syntax_error":                     "build_artifact",
            "snapshot_test_mismatch":                       "build_artifact",
            "xcode_test_scheme_14_failures_exit_65":        "build_artifact",
            "android_lint_vital_baseline_created":          "build_artifact",
            "branch_not_found":                             "build_artifact",
            "git_push_non_fast_forward":                    "build_artifact",
            "git_checkout_null_branch":                     "build_artifact",
            "manual_abort_gradle_clean":                    "build_artifact",
            "manual_abort_before_gradle":                   "build_artifact",
            "manual_abort_during_r8_minification":          "build_artifact",
            "upstream_pipeline_cancelled_during_gradle_build": "build_artifact",
        }

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                error_type = item.get("metadata", {}).get("error_type", "build_artifact")
                category = MAIN_CATEGORY_MAP.get(error_type, "build_artifact")
                if category not in MAIN_CATEGORIES:
                    category = "build_artifact"

                fix_code = ""
                fix = item.get("fix", {})
                if isinstance(fix, dict):
                    fix_code = fix.get("code", fix.get("diff", ""))

                # fix_description: önce expected_response, yoksa ground_truth_rca
                fix_description = (
                    item.get("expected_response")
                    or item.get("ground_truth_rca")
                    or ""
                )

                record = FixRecord(
                    failure_log=item.get("log_chunk", ""),
                    fix_description=fix_description,
                    fix_diff=fix_code,
                    category=category,
                    platform=item.get("metadata", {}).get("platform", "ios"),
                    source="jenkins",
                    tags=[item.get("id", "")],
                )
                records.append(record)

        self.index_batch(records)
        return len(records)