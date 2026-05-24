# src/agentops/rag/dependency.py
"""
Dependency RAG Module

Podfile.lock, Package.resolved, build.gradle, Podfile gibi
dependency manifest dosyalarını index'ler ve sorgular.

Kullanım alanları:
- CocoaPods version conflict
- SPM package resolution failure
- Gradle dependency resolution failure
- Version incompatibility patterns
"""

from __future__ import annotations

import hashlib
import json
import os
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


COLLECTION_NAME = "dependency_knowledge"
VECTOR_SIZE = 768


DEFAULT_DEPENDENCY_SEED = [
    {
        "package_name": "internal-spm/CoreNFCSPM",
        "error_pattern": "Could not resolve host: internal-spm.example.com",
        "resolution": "Internal SPM host DNS/route issue. Check agent DNS and network route; use mirror/cache fallback if internal host is unavailable.",
        "manifest_type": "spm",
        "platform": "ios",
        "example_fix": "Validate DNS on agent, retry xcodebuild -resolvePackageDependencies, and fallback to mirrored SPM source.",
        "tags": ["dns", "spm", "internal-host"],
    },
    {
        "package_name": "CocoaPods",
        "error_pattern": "Could not find compatible versions for pod",
        "resolution": "Pod version conflict. Align Podfile constraints with Podfile.lock or run pod update for conflicting pod set.",
        "manifest_type": "podfile",
        "platform": "ios",
        "example_fix": "pod repo update && pod update <POD_NAME> followed by deterministic Podfile.lock commit.",
        "tags": ["cocoapods", "version-conflict"],
    },
    {
        "package_name": "Swift Package Manager",
        "error_pattern": "Could not resolve package dependencies",
        "resolution": "SPM dependency graph could not be resolved due to host/auth/version constraints.",
        "manifest_type": "spm",
        "platform": "ios",
        "example_fix": "Clear Package.resolved mismatch, ensure repo credentials, then re-run xcodebuild -resolvePackageDependencies.",
        "tags": ["spm", "resolution"],
    },
    {
        "package_name": "Gradle Kotlin Daemon",
        "error_pattern": "Connection to the Kotlin daemon has been unexpectedly lost",
        "resolution": "Kotlin daemon crashed or was killed (often memory pressure or manual interrupt).",
        "manifest_type": "gradle",
        "platform": "android",
        "example_fix": "Increase org.gradle.jvmargs and kotlin.daemon.jvmargs, run ./gradlew --stop, then retry.",
        "tags": ["kotlin", "daemon", "resource"],
    },
    {
        "package_name": "Gradle Build Daemon",
        "error_pattern": "Gradle build daemon disappeared unexpectedly",
        "resolution": "Gradle daemon process crashed or was terminated.",
        "manifest_type": "gradle",
        "platform": "android",
        "example_fix": "Inspect daemon logs, stop stale daemons, and reduce parallelism for constrained agents.",
        "tags": ["gradle", "daemon", "crash"],
    },
    {
        "package_name": "Android Dependencies",
        "error_pattern": "Could not resolve all files for configuration",
        "resolution": "Repository availability or dependency version mismatch in Gradle configuration.",
        "manifest_type": "gradle",
        "platform": "android",
        "example_fix": "Verify repositories block, enforce version catalog alignment, and refresh dependencies.",
        "tags": ["gradle", "dependency", "repository"],
    },
    {
        "package_name": "Shared Network Dependency",
        "error_pattern": "getaddrinfo|Could not resolve host",
        "resolution": "Network/DNS dependency endpoint unavailable from CI runner.",
        "manifest_type": "shared",
        "platform": "shared",
        "example_fix": "Check DNS resolver, network ACL, and temporary host override only for controlled CI tests.",
        "tags": ["dns", "network", "shared"],
    },
]


@dataclass
class DependencyRecord:
    """
    Bir dependency pattern kaydı.
    """
    package_name: str          # Örn: "Firebase/Analytics", "com.google.firebase:firebase-bom"
    error_pattern: str         # Hangi hata mesajıyla tetikleniyor
    resolution: str            # Çözüm açıklaması
    manifest_type: str         # podfile | spm | gradle | npm
    platform: str              # ios | android | shared
    example_fix: str           # Somut fix örneği (diff veya komut)
    tags: list[str] = field(default_factory=list)

    def to_payload(self) -> dict:
        return {
            "package_name": self.package_name,
            "error_pattern": self.error_pattern,
            "resolution": self.resolution,
            "manifest_type": self.manifest_type,
            "platform": self.platform,
            "example_fix": self.example_fix,
            "tags": self.tags,
        }

    def get_embed_text(self) -> str:
        return (
            f"PACKAGE: {self.package_name}\n"
            f"ERROR: {self.error_pattern}\n"
            f"RESOLUTION: {self.resolution}"
        )

    def get_id(self) -> str:
        raw = self.package_name + self.error_pattern
        return str(uuid.UUID(hashlib.md5(raw.encode()).hexdigest()))


@dataclass
class DependencyResult:
    package_name: str
    resolution: str
    manifest_type: str
    platform: str
    example_fix: str
    score: float
    tags: list[str] = field(default_factory=list)


class DependencyRAG:
    """
    Dependency RAG modülü.

    Kullanım:
        rag = DependencyRAG()
        rag.ensure_collection()
        results = rag.retrieve("CocoaPods could not find compatible versions for pod Firebase")
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

    def index(self, record: DependencyRecord) -> str:
        vector = embed_text(record.get_embed_text())
        point_id = record.get_id()
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=vector, payload=record.to_payload())],
        )
        return point_id

    def index_batch(self, records: list[DependencyRecord]) -> list[str]:
        points, ids = [], []
        for record in records:
            vector = embed_text(record.get_embed_text())
            point_id = record.get_id()
            points.append(PointStruct(id=point_id, vector=vector, payload=record.to_payload()))
            ids.append(point_id)
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"✅ {len(points)} dependency kaydı index'lendi.")
        return ids

    def seed_from_dataset(self, path: str) -> int:
        """
        dataset.jsonl'daki SPM/dependency hatalarını DependencyRecord olarak index'ler.
        Aynı zamanda DEFAULT_DEPENDENCY_SEED'i de ekler (hardcoded genel pattern'lar).
        """
        DEPENDENCY_ERROR_TYPES = {
            "spm_dns_resolution_failure":                       ("spm", "ios"),
            "spm_dns_resolution_failure_inside_gym":            ("spm", "ios"),
            "spm_dns_resolution_failure_public_github":         ("spm", "ios"),
            "spm_resolution_failure":                           ("spm", "ios"),
            "spm_package_resolution_network_dns_failure":       ("spm", "ios"),
            "spm_submodule_and_binary_artifact_resolution_failure": ("spm", "ios"),
            "spm_credential_error":                             ("spm", "ios"),
            "unresolved_reference_missing_dependency":          ("gradle", "android"),
        }

        records = []

        # 1. Hardcoded seed pattern'lar
        for item in DEFAULT_DEPENDENCY_SEED:
            records.append(DependencyRecord(**item))

        # 2. Dataset'ten gerçek kayıtlar
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                error_type = item.get("metadata", {}).get("error_type", "")
                if error_type not in DEPENDENCY_ERROR_TYPES:
                    continue

                manifest_type, platform = DEPENDENCY_ERROR_TYPES[error_type]
                log = item.get("log_chunk", "")
                rca = item.get("ground_truth_rca", "")
                fix_obj = item.get("fix") or {}
                fix_code = fix_obj.get("code", "") if isinstance(fix_obj, dict) else ""

                # Paket adını log'dan çıkarmaya çalış
                import re
                pkg_match = re.search(r"'([^']+\.git|[^']+SPM[^']*|github\.com/[^\s']+)'", log)
                package_name = pkg_match.group(1) if pkg_match else error_type.replace("_", " ").title()

                record = DependencyRecord(
                    package_name=package_name[:100],
                    error_pattern=item.get("error_message", log[:200]),
                    resolution=rca,
                    manifest_type=manifest_type,
                    platform=platform,
                    example_fix=fix_code or rca[:200],
                    tags=[item.get("id", ""), error_type],
                )
                records.append(record)

        self.index_batch(records)
        return len(records)

    def seed_defaults_if_empty(self, dataset_path: Optional[str] = None) -> int:
        """Collection boşsa dataset + hardcoded seed ile doldurur."""
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

    def retrieve(
        self,
        query: str,
        manifest_type: Optional[str] = None,
        platform: Optional[str] = None,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> list[DependencyResult]:
        self.ensure_collection()
        self.seed_defaults_if_empty()
        vector = embed_text(query)
        if min_score is None:
            min_score = float(os.getenv("RAG_MIN_SCORE", "0.60"))

        conditions = []
        if manifest_type:
            conditions.append(FieldCondition(key="manifest_type", match=MatchValue(value=manifest_type)))
        if platform:
            query_filter = Filter(
                must=conditions,
                should=[
                    FieldCondition(key="platform", match=MatchValue(value=platform)),
                    FieldCondition(key="platform", match=MatchValue(value="shared")),
                ],
            )
        else:
            query_filter = Filter(must=conditions) if conditions else None

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            DependencyResult(
                package_name=r.payload.get("package_name", ""),
                resolution=r.payload.get("resolution", ""),
                manifest_type=r.payload.get("manifest_type", ""),
                platform=r.payload.get("platform", ""),
                example_fix=r.payload.get("example_fix", ""),
                score=r.score,
                tags=r.payload.get("tags", []),
            )
            for r in results
            if r.score >= min_score
        ]

    def load_from_json(self, path: str) -> int:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = [DependencyRecord(**item) for item in data]
        self.index_batch(records)
        return len(records)

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name).count
