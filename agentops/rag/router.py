# src/agentops/rag/router.py
"""
RAG Router

Failure kategorisine göre hangi RAG modüllerinin
aktive edileceğine karar verir ve sonuçları birleştirir.

Routing tablosu:
    Build Artifact → Historical + Dependency + Platform
    Dependency     → Historical + Dependency
    Merge Conflict → Historical + Conflict
    Signing        → Historical + Platform
    Authentication → Historical
    Resource       → Historical
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .historical_fix import HistoricalFixRAG, FixResult
from .dependency import DependencyRAG, DependencyResult
from .platform_knowledge import PlatformKnowledgeRAG, PlatformResult
from .conflict_resolution import ConflictResolutionRAG, ConflictResult


# Routing tablosu
ROUTING_TABLE: dict[str, list[str]] = {
    "build_artifact": ["historical", "platform"],
    "dependency":     ["historical", "dependency", "platform"],
    "merge_conflict": ["historical", "conflict"],
    "signing":        ["historical", "platform"],
    "authentication": ["historical"],
    "resource":       ["historical", "platform"],
}


@dataclass
class RAGContext:
    """
    Router'ın döndürdüğü birleşik bağlam.
    Diff generator bu nesneyi kullanır.
    """
    category: str
    activated_modules: list[str]

    historical_results: list[FixResult] = field(default_factory=list)
    dependency_results: list[DependencyResult] = field(default_factory=list)
    platform_results: list[PlatformResult] = field(default_factory=list)
    conflict_results: list[ConflictResult] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """
        Tüm RAG sonuçlarını LLM prompt'una eklenecek
        tek bir string'e dönüştürür.
        """
        parts = []

        if self.historical_results:
            parts.append("=== HISTORICAL FIXES ===")
            for i, r in enumerate(self.historical_results, 1):
                parts.append(
                    f"[{i}] Category: {r.category} | Platform: {r.platform} | Score: {r.score:.2f}\n"
                    f"Fix: {r.fix_description}\n"
                    f"Diff:\n{r.fix_diff}" if r.fix_diff else
                    f"[{i}] Category: {r.category} | Score: {r.score:.2f}\n"
                    f"Fix: {r.fix_description}"
                )

        if self.dependency_results:
            parts.append("\n=== DEPENDENCY CONTEXT ===")
            for i, r in enumerate(self.dependency_results, 1):
                parts.append(
                    f"[{i}] Package: {r.package_name} | Type: {r.manifest_type}\n"
                    f"Resolution: {r.resolution}\n"
                    f"Example: {r.example_fix}"
                )

        if self.platform_results:
            parts.append("\n=== PLATFORM KNOWLEDGE ===")
            for i, r in enumerate(self.platform_results, 1):
                parts.append(
                    f"[{i}] {r.title} | Tool: {r.tool}\n"
                    f"Explanation: {r.explanation}\n"
                    f"Fix: {r.fix}"
                )

        if self.conflict_results:
            parts.append("\n=== CONFLICT RESOLUTION EXAMPLES ===")
            for i, r in enumerate(self.conflict_results, 1):
                parts.append(
                    f"[{i}] Type: {r.conflict_type} | File: {r.file_type}\n"
                    f"Explanation: {r.explanation}\n"
                    f"Resolution: {r.resolution}\n"
                    f"Resolved:\n{r.resolved_snippet}"
                )

        return "\n".join(parts)


class RAGRouter:
    """
    RAG Router with enhanced query preprocessing and ranking.

    Kullanım:
        router = RAGRouter()
        router.ensure_all_collections()

        context = router.route(
            category="dependency",
            query="CocoaPods could not find compatible versions for pod Firebase",
            platform="ios",
        )

        prompt_context = context.to_prompt_context()
    """

    def __init__(
        self,
        qdrant_url: str = "",  # Artık kullanılmıyor, geriye dönük uyumluluk için tutuldu
        es_url: str = "http://localhost:9200",
    ) -> None:
        self.historical = HistoricalFixRAG()
        self.dependency = DependencyRAG()
        self.platform = PlatformKnowledgeRAG()
        self.conflict = ConflictResolutionRAG()

    def _qdrant_available(self, retries: int = 1) -> bool:
        """PostgreSQL bağlantısını doğrular."""
        for attempt in range(retries + 1):
            try:
                self.historical.client.get_collections()
                return True
            except Exception as e:
                if attempt < retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                return False
        return False

    def _retrieve_with_retry(self, fetcher, retries: int = 1):
        """Retriever çağrılarını geçici bağlantı hatalarına karşı tekrar dener."""
        for attempt in range(retries + 1):
            try:
                return fetcher()
            except Exception as e:
                msg = str(e).lower()
                transient = (
                    "connection refused" in msg
                    or "failed to establish" in msg
                    or "temporarily unavailable" in msg
                )
                if transient and attempt < retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise

    def ensure_all_collections(self) -> None:
        """Tüm modüllerin collection/index'lerini oluşturur ve boşsa seed eder."""
        self.historical.ensure_collection()
        self.dependency.ensure_collection()
        self.dependency.seed_defaults_if_empty()
        self.platform.ensure_collection()
        n_platform = self.platform.seed_defaults_if_empty()
        if n_platform:
            print(f"✅ platform_knowledge: {n_platform} kayıt seed edildi.")
        self.conflict.ensure_collection()
        n_conflict = self.conflict.seed_defaults_if_empty()
        if n_conflict:
            print(f"✅ conflict_resolution: {n_conflict} kayıt seed edildi.")
        print("✅ Tüm RAG collection'ları hazır.")

    @staticmethod
    def _preprocess_query(query: str, category: str) -> str:
        """
        Query preprocessing: hata logunu daha retrieval-friendly hale getir.
        - Stack trace kısaltma
        - Tekrarlayan satırları temizle
        - Error markers öne çıkart
        - Platform-specific keywords vurgula
        """
        if not query:
            return ""
        
        lines = query.split("\n")
        
        # Stack trace'i kısalt (genelde hata ilk 3-5 satırda)
        relevant_lines = []
        error_keyword_found = False
        for line in lines:
            lower = line.lower()
            if any(kw in lower for kw in ["error", "failed", "fatal", "exception"]):
                error_keyword_found = True
            if error_keyword_found:
                relevant_lines.append(line)
            if len(relevant_lines) > 15:  # Max 15 lines
                break
        
        preprocessed = "\n".join(relevant_lines) if relevant_lines else query
        
        # Platform-specific keywords vurgula
        if category == "dependency":
            preprocessed = re.sub(
                r"(pod|podfile|cocoapods|spm|cartfile|gradle|maven)",
                r"[\1]",
                preprocessed,
                flags=re.IGNORECASE
            )
        elif category == "merge_conflict":
            preprocessed = re.sub(
                r"(conflict|merge|CONFLICT)",
                r"[\1]",
                preprocessed,
                flags=re.IGNORECASE
            )
        elif category == "signing":
            preprocessed = re.sub(
                r"(codesign|keystore|provisioning|signing)",
                r"[\1]",
                preprocessed,
                flags=re.IGNORECASE
            )
        
        # Duplicate lines kaldır
        unique_lines = []
        seen = set()
        for line in preprocessed.split("\n"):
            normalized = line.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_lines.append(line)
        
        final = "\n".join(unique_lines)
        return final[:1200]  # Max 1200 chars

    @staticmethod
    def _rank_results(results: list, category: str, query: str) -> list:
        """
        Better ranking: score + relevance check.
        """
        if not results:
            return []
        
        query_lower = query.lower()
        
        # Add relevance bonus if result mentions same tools/patterns
        ranked = []
        for result in results:
            score = getattr(result, 'score', 0.5)
            relevance_boost = 0.0
            
            # Check common attributes for relevance keywords
            for attr in ['fix_description', 'title', 'explanation', 'resolution', 'package_name']:
                if hasattr(result, attr):
                    text = str(getattr(result, attr, "")).lower()
                    # Boost if same keywords appear in both query and result
                    keywords = re.findall(r'\b\w{4,}\b', query_lower)
                    matches = sum(1 for kw in keywords if kw in text)
                    relevance_boost += matches * 0.05
            
            final_score = min(score + relevance_boost, 1.0)
            ranked.append((final_score, result))
        
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in ranked]

    def route(
        self,
        category: str,
        query: str,
        platform: Optional[str] = None,
        run_id: Optional[str] = None,
        conflict_type: Optional[str] = None,
        top_k: int = 3,
        min_score: Optional[float] = None,
    ) -> RAGContext:
        """
        Kategori bazlı routing: ilgili modülleri aktive eder
        ve RAGContext döndürür.

        Args:
            category: Failure kategorisi (build_artifact, dependency, vs.)
            query: Ham hata mesajı veya log özeti
            platform: ios | android | shared (opsiyonel)
            run_id: Geriye dönük uyumluluk için tutulur (aktif kullanılmaz)
            conflict_type: spurious | auto_merge | syntactic | semantic (opsiyonel)
            top_k: Her modülden kaç sonuç
        """
        category = category.lower().strip()
        modules = ROUTING_TABLE.get(category, ["historical"])
        
        # Optional: disable platform module for specific testing (v11a)
        if os.getenv("DISABLE_PLATFORM_KNOWLEDGE", "0").lower() in {"1", "true", "yes"}:
            modules = [m for m in modules if m != "platform"]
            if not modules:
                modules = ["historical"]

        # ✨ Query preprocessing: better retrieval
        preprocessed_query = self._preprocess_query(query, category)

        context = RAGContext(
            category=category,
            activated_modules=modules,
        )

        if min_score is None:
            min_score = float(os.getenv("RAG_MIN_SCORE", "0.60"))

        print(f"🔀 RAG routing: category={category} → modules={modules}")

        # Qdrant yoksa her retriever için ayrı hata üretmek yerine tek noktada graceful fallback.
        if not self._qdrant_available(retries=1):
            print("ℹ️ RAG retrieval atlandı: PostgreSQL erişilemiyor")
            return context

        if "historical" in modules:
            try:
                raw_results = self._retrieve_with_retry(
                    lambda: self.historical.retrieve(
                        query=preprocessed_query,
                        category=category,
                        platform=platform,
                        top_k=top_k + 2,  # Get extra for ranking
                    )
                )
                # ✨ Better ranking
                ranked = self._rank_results(raw_results, category, query)
                context.historical_results = [
                    r for r in ranked[:top_k] if r.score >= min_score
                ]
            except Exception as e:
                print(f"ℹ️ historical retrieval atlandı: {e}")

        if "dependency" in modules:
            try:
                raw_results = self._retrieve_with_retry(
                    lambda: self.dependency.retrieve(
                        query=preprocessed_query,
                        platform=platform,
                        top_k=top_k + 2,
                        min_score=min_score * 0.9,  # Slightly relaxed
                    )
                )
                # ✨ Better ranking
                ranked = self._rank_results(raw_results, category, query)
                context.dependency_results = ranked[:top_k]
            except Exception as e:
                print(f"ℹ️ dependency retrieval atlandı: {e}")

        if "platform" in modules:
            try:
                raw_results = self._retrieve_with_retry(
                    lambda: self.platform.retrieve(
                        query=preprocessed_query,
                        platform=platform,
                        top_k=top_k + 2,
                    )
                )
                # ✨ Better ranking
                ranked = self._rank_results(raw_results, category, query)
                context.platform_results = [
                    r for r in ranked[:top_k] if r.score >= min_score
                ]
            except Exception as e:
                print(f"ℹ️ platform retrieval atlandı: {e}")

        if "conflict" in modules:
            try:
                raw_results = self._retrieve_with_retry(
                    lambda: self.conflict.retrieve(
                        query=preprocessed_query,
                        conflict_type=conflict_type,
                        platform=platform,
                        top_k=top_k + 2,
                    )
                )
                # ✨ Better ranking
                ranked = self._rank_results(raw_results, category, query)
                context.conflict_results = [
                    r for r in ranked[:top_k] if r.score >= min_score
                ]
            except Exception as e:
                print(f"ℹ️ conflict retrieval atlandı: {e}")

        return context

    def get_routing_table(self) -> dict[str, list[str]]:
        """Mevcut routing tablosunu döndürür."""
        return ROUTING_TABLE
