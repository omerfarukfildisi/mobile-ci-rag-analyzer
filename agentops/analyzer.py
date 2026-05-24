# src/agentops/analyzer.py

import json
import os

import requests

from .log_models import CiLog, CiAnalysis, MAIN_CATEGORIES
from .knowledge.error_knowledge import match_categories


# Sub-kategori → main_category map
# error_ontology.json'daki main_category alanıyla birebir uyumlu
SUB_TO_MAIN: dict[str, str] = {
    # build_artifact
    "build_error": "build_artifact",
    "compilation_error": "build_artifact",
    "ios_toolchain_error": "build_artifact",
    "xcodebuild_error": "build_artifact",
    "kotlin_compiler_error": "build_artifact",
    "android_manifest_error": "build_artifact",
    "test_failure": "build_artifact",
    "gradle_sync_error": "build_artifact",
    "toolchain_error": "build_artifact",

    # dependency
    "dependency_error": "dependency",
    "version_conflict": "dependency",
    "cocoapods_error": "dependency",
    "swift_package_manager_error": "dependency",

    # merge_conflict
    "merge_conflict_error": "merge_conflict",
    "pbxproj_conflict": "merge_conflict",
    "podfile_conflict": "merge_conflict",
    "gradle_conflict": "merge_conflict",

    # signing
    "codesign_error": "signing",
    "provisioning_profile_error": "signing",
    "keystore_error": "signing",
    "ios_bundle_id_error": "signing",

    # authentication
    "authentication_error": "authentication",
    "authorization_error": "authentication",
    "credential_error": "authentication",
    "secret_missing_error": "authentication",
    "api_rate_limit_error": "authentication",

    # resource
    "memory_limit_error": "resource",
    "disk_space_error": "resource",
    "timeout_error": "resource",
    "runner_environment_error": "resource",
    "network_error": "resource",
    "simulator_error": "resource",
    "resource_not_found_error": "resource",
    "cache_corruption_error": "resource",
}


def resolve_main_category(sub_category: str) -> str:
    """
    Sub-kategoriyi 6 ana kategoriden birine map eder.
    Bilinmiyorsa 'build_artifact' döndürür (en genel fallback).
    """
    return SUB_TO_MAIN.get(sub_category.lower().strip(), "build_artifact")


class SimpleAnalyzer:
    """
    Demo / fallback analizci.
    """

    def analyze(self, log: CiLog) -> CiAnalysis:
        return CiAnalysis(
            main_category="build_artifact",
            category="build_error",
            root_cause="Dependency version mismatch",
            explanation="The build failed because of an incompatible library version.",
            suggestion="Update the conflicting dependency to a compatible version.",
            confidence=0.5,
        )


class OllamaAnalyzer:
    """
    Local LLM (Ollama) tabanlı analizci.
    Ontology hint + 6 main_category destekli.

        Hız optimizasyonu:
            - Modele tüm log yerine failure chunk gönderir.
            - Classification + RCA yerine tek çağrıda nihai JSON üretir.
    """

    def __init__(
        self,
        classification_model: str = "llama3.1:8b",
        rca_model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        classification_model = os.getenv("OLLAMA_CLASSIFICATION_MODEL", classification_model)
        rca_model = os.getenv("OLLAMA_RCA_MODEL", rca_model)
        self.classification_model = classification_model
        self.rca_model = rca_model
        self.base_url = base_url.rstrip("/")

    def _call_ollama(self, model: str, prompt: str) -> str:
        """Ollama API'ye istek atar, ham response string döndürür."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
            "options": {
                "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
                "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "180")),
            },
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def _parse_json(self, content: str) -> dict:
        """LLM çıktısından JSON bloğunu çıkarır."""
        clean = content
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())

    def _extract_failure_chunk(self, raw_log: str, max_chars: int = 3500) -> str:
        """
        Uzun logtan hata odaklı bir kesit çıkarır.
        Hiç eşleşme bulunamazsa son satırlardan bir pencere döndürür.
        """
        if not raw_log:
            return ""

        lines = raw_log.splitlines()
        total = len(lines)
        lowered = [ln.lower() for ln in lines]

        error_markers = [
            "error",
            "fatal",
            "failed",
            "exception",
            "traceback",
            "exit status",
            "script returned exit code",
            "could not",
            "unable to",
        ]

        hit_indices: list[int] = []
        for i, ln in enumerate(lowered):
            if any(marker in ln for marker in error_markers):
                hit_indices.append(i)

        # Sonuç yoksa kuyruğu al; CI hataları genelde sona yakın olur.
        if not hit_indices:
            tail_start = max(0, total - 220)
            chunk = "\n".join(lines[tail_start:])
            return chunk[-max_chars:]

        # Hata etrafından pencere al ve indeksleri birleştir.
        selected: set[int] = set()
        for i in hit_indices:
            start = max(0, i - 20)
            end = min(total, i + 41)
            for j in range(start, end):
                selected.add(j)

        ordered_indices = sorted(selected)
        chunk_lines = [lines[i] for i in ordered_indices]
        chunk = "\n".join(chunk_lines).strip()

        if len(chunk) > max_chars:
            chunk = chunk[-max_chars:]
        return chunk

    def _extract_affected_files(self, raw_log: str) -> tuple[list[str], list[str]]:
        """Log'dan etkilenen dosya ve sınıfları regex ile çıkarır. LLM kullanmaz."""
        import re

        patterns = [
            # Build errors
            r"error:.*?['\"](/?\S+?\.\w{1,5})['\"]",
            r"error:\s+(\S+\.(?:swift|m|h|kt|java)):\d+",
            r"CompileSwift\s+.*?/(\S+\.swift)",
            r"CompileC\s+.*?/(\S+\.[mhc])",
            r"No such file.*?['\"](/?\S+?\.\w{1,5})['\"]",
            # Merge conflicts
            r"CONFLICT.*?in\s+(\S+)",
            r"Auto-merging\s+(\S+)",
            # Signing
            r"CodeSign\s+.*?/(\S+\.app)",
            r"Provisioning profile.*?['\"](\S+)['\"]",
            # Dependency — config dosyaları
            r"(?:^|\s)(Podfile(?:\.lock)?)\b",
            r"(?:^|\s)(Package\.(?:swift|resolved))\b",
            r"(?:^|\s)(Cartfile(?:\.resolved)?)\b",
            r"(?:^|\s)(build\.gradle(?:\.kts)?)\b",
            r"(?:^|\s)(settings\.gradle(?:\.kts)?)\b",
            r"(?:^|\s)(libs\.versions\.toml)\b",
            r"(?:^|\s)(Gemfile(?:\.lock)?)\b",
            # Dependency — pod/package adları
            r"for pod ['\"](\S+?)['\"]",
            r"Updating from https://github\.com/\S+?/(\S+?)(?:\.git)?$",
        ]

        files: set[str] = set()
        for pat in patterns:
            for match in re.findall(pat, raw_log, re.MULTILINE):
                name = match.rsplit("/", 1)[-1].strip()
                if name and len(name) < 120:
                    files.add(name)

        # Dosya adından class türet (Swift/ObjC/Kotlin convention)
        source_exts = {".swift", ".m", ".h", ".kt", ".java"}
        classes: set[str] = set()
        for f in files:
            base, _, ext = f.rpartition(".")
            if f".{ext}" in source_exts and base and base[0].isupper():
                classes.add(base)

        return sorted(files), sorted(classes)

    def _detect_conflict_type(self, raw_log: str, affected_files: list[str]) -> str:
        """Log'dan conflict tipini tespit eder. Conflict yoksa boş string döner."""
        import re

        # Conflict var mı?
        has_conflict = bool(re.search(
            r"CONFLICT|<<<<<<<|merge conflict|Automatic merge failed",
            raw_log, re.IGNORECASE
        ))
        if not has_conflict:
            return ""

        # Conflict'li dosya uzantılarına göre tip belirle
        auto_merge_files = {
            "Podfile.lock", "Package.resolved", "Cartfile.resolved",
            ".xcworkspace", "Pods",
        }
        spurious_files = {
            "project.pbxproj", ".xcodeproj",
            "gradle.properties", "gradle-wrapper.properties",
        }
        source_exts = {".swift", ".m", ".h", ".kt", ".java", ".py", ".ts", ".js"}
        config_exts = {".plist", ".entitlements", ".xcconfig", ".yml", ".yaml", ".json", ".xml"}

        conflict_files = []
        for match in re.findall(r"CONFLICT.*?in\s+(\S+)", raw_log):
            conflict_files.append(match.rsplit("/", 1)[-1])

        if not conflict_files:
            conflict_files = affected_files

        has_source = False
        has_spurious = False
        has_auto = False
        has_config = False

        for f in conflict_files:
            _, _, ext = f.rpartition(".")
            ext_dot = f".{ext}" if ext else ""

            if any(s in f for s in spurious_files):
                has_spurious = True
            elif any(s in f for s in auto_merge_files):
                has_auto = True
            elif ext_dot in source_exts:
                has_source = True
            elif ext_dot in config_exts:
                has_config = True

        # Öncelik: semantic > syntactic > spurious > auto_merge
        if has_source:
            return "semantic"
        if has_config:
            return "syntactic"
        if has_spurious:
            return "spurious"
        if has_auto:
            return "auto_merge"

        return "semantic"  # bilinmeyen dosya tipi → güvenli tarafta kal

    def _detect_resolution_strategy(self, conflict_type: str, affected_files: list[str]) -> str:
        """Conflict tipine ve dosya türüne göre çözüm stratejisi belirler."""
        if not conflict_type:
            return ""

        regenerate_files = {
            "Podfile.lock", "Package.resolved", "Cartfile.resolved",
        }
        # Regenerate edilebilir dosya varsa
        if any(f in regenerate_files for f in affected_files):
            return "regenerate"

        strategy_map = {
            "spurious": "union_merge",
            "auto_merge": "regenerate",
            "syntactic": "pick_newer",
            "semantic": "pick_newer",
        }
        return strategy_map.get(conflict_type, "pick_newer")

    def _extract_jira_task_id(self, branch: str) -> str:
        """Branch adından Jira task ID'si çıkarır."""
        import re
        match = re.search(r"(MOB-\d+|REQ-\d+|PROJ-\d+|STAGE-\d+)", branch, re.IGNORECASE)
        return match.group(1).upper() if match else ""

    def _build_analysis_prompt(self, log: CiLog, log_chunk: str, ontology_hint: str, rag_context: str) -> str:
        """
        Tek adımda kategori + RCA + öneri üreten prompt.
        """
        rag_section = f"=== RAG CONTEXT ===\n{rag_context}\n\n" if rag_context else ""
        return (
            "You are an expert CI/CD failure analysis agent for mobile (iOS/Android) pipelines. "
            "Do NOT explain your reasoning. Respond ONLY with valid JSON.\n\n"
            f"=== CI LOG ===\n"
            f"Platform: {log.platform}\n"
            f"Pipeline: {log.pipeline_name}\n"
            f"Branch: {log.branch}\n"
            f"Commit: {log.commit_sha}\n\n"
            f"Failure Chunk:\n{log_chunk}\n\n"
            f"=== ONTOLOGY HINTS ===\n{ontology_hint}\n\n"
            f"{rag_section}"
            "=== TASK ===\n"
            "Analyze this failure and return final output. main_category MUST be exactly one of:\n"
            "  build_artifact | dependency | merge_conflict | signing | authentication | resource\n\n"
            "Return ONLY this JSON:\n"
            "{\n"
            '  "main_category": "...",\n'
            '  "category": "...",\n'
            '  "root_cause": "...",\n'
            '  "explanation": "...",\n'
            '  "suggestion": "...",\n'
            '  "confidence": 0.0\n'
            "}"
        )

    def _build_rag_context(self, guessed_category: str, query: str, log: CiLog) -> str:
        """
        Hafif ve güvenli RAG: routing tablosunu kullanır, pahalı modülleri sınırlar.
        Hata alırsa sessizce boş bağlam döner.
        """
        enabled = os.getenv("ENABLE_RAG_ROUTING", "1").lower() not in {"0", "false", "no"}
        if not enabled:
            return ""

        try:
            from .rag.router import RAGRouter, ROUTING_TABLE
        except Exception as e:
            print(f"ℹ️ RAG import atlandı: {e}")
            return ""

        try:
            modules = ROUTING_TABLE.get(guessed_category, ["historical"])
            max_modules = int(os.getenv("RAG_MAX_MODULES", "2"))
            modules = modules[:max_modules]

            router = RAGRouter(
                qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                es_url=os.getenv("ES_URL", "http://localhost:9200"),
            )

            top_k = int(os.getenv("RAG_TOP_K", "1"))
            summary: list[str] = []

            if "historical" in modules:
                try:
                    res = router.historical.retrieve(
                        query=query,
                        category=guessed_category,
                        platform=log.platform,
                        top_k=top_k,
                    )
                    for r in res:
                        summary.append(f"[historical] {r.fix_description}")
                except Exception as e:
                    print(f"ℹ️ RAG historical atlandı: {e}")

            if "dependency" in modules:
                try:
                    res = router.dependency.retrieve(
                        query=query,
                        platform=log.platform,
                        top_k=top_k,
                    )
                    for r in res:
                        summary.append(f"[dependency] {r.package_name}: {r.resolution}")
                except Exception as e:
                    print(f"ℹ️ RAG dependency atlandı: {e}")

            if "platform" in modules:
                try:
                    res = router.platform.retrieve(
                        query=query,
                        platform=log.platform,
                        top_k=top_k,
                    )
                    for r in res:
                        summary.append(f"[platform] {r.title}: {r.fix}")
                except Exception as e:
                    print(f"ℹ️ RAG platform atlandı: {e}")

            if "conflict" in modules:
                try:
                    res = router.conflict.retrieve(
                        query=query,
                        platform=log.platform,
                        top_k=top_k,
                    )
                    for r in res:
                        summary.append(f"[conflict] {r.conflict_type}: {r.resolution}")
                except Exception as e:
                    print(f"ℹ️ RAG conflict atlandı: {e}")

            rag_context = "\n".join(summary).strip()
            max_rag_chars = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "1200"))
            if len(rag_context) > max_rag_chars:
                rag_context = rag_context[:max_rag_chars]
            return rag_context
        except Exception as e:
            print(f"ℹ️ RAG routing atlandı: {e}")
            return ""

    def analyze(self, log: CiLog) -> CiAnalysis:
        # Tek çağrı: classification + rca birlikte
        model = self.rca_model
        print(f"⚡ Analysis [{model}]...")
        try:
            matches = match_categories(log.raw_log, top_k=3)
            if matches:
                hint_lines = []
                for m in matches:
                    hint_lines.append(
                        f"- sub: {m.name} → main: {resolve_main_category(m.name)} (score={m.score:.2f})"
                    )
                ontology_hint = "Pattern-based hints:\n" + "\n".join(hint_lines)
            else:
                ontology_hint = "No strong pattern match found."

            log_chunk = self._extract_failure_chunk(log.raw_log)
            guessed_category = resolve_main_category(matches[0].name) if matches else "resource"
            rag_query = log_chunk[-1200:] if len(log_chunk) > 1200 else log_chunk
            rag_context = self._build_rag_context(guessed_category, rag_query, log)
            prompt = self._build_analysis_prompt(log, log_chunk, ontology_hint, rag_context)
            content = self._call_ollama(model, prompt)
            data = self._parse_json(content)

            main_category = data.get("main_category", "")
            category = data.get("category", "")
            if main_category not in MAIN_CATEGORIES:
                main_category = resolve_main_category(category)

            affected_files, affected_classes = self._extract_affected_files(log.raw_log)
            conflict_type = self._detect_conflict_type(log.raw_log, affected_files)
            resolution_strategy = self._detect_resolution_strategy(conflict_type, affected_files)
            jira_task_id = self._extract_jira_task_id(log.branch)

            analysis = CiAnalysis(
                main_category=main_category,
                category=category or "build_error",
                root_cause=data.get("root_cause", "Root cause could not be extracted."),
                explanation=data.get("explanation", "Analysis completed with partial output."),
                suggestion=data.get("suggestion", "Review the failing stage and retry after remediation."),
                confidence=float(data.get("confidence", 0.3)),
                affected_files=affected_files,
                affected_classes=affected_classes,
                conflict_type=conflict_type,
                resolution_strategy=resolution_strategy,
                jira_task_id=jira_task_id,
            )
            self._write_pending(log, analysis, log_chunk)
            return analysis
        except Exception as e:
            print(f"❌ Tek adım analiz hatası: {e} → ontology fallback.")
            return self._ontology_fallback(log)

    def _ontology_fallback(self, log: CiLog) -> CiAnalysis:
        """Her iki adım da başarısız olursa ontology sonucuyla fallback."""
        matches = match_categories(log.raw_log, top_k=1)
        if matches:
            best = matches[0]
            main = resolve_main_category(best.name)
            return CiAnalysis(
                main_category=main,
                category=best.name,
                root_cause=best.explanation,
                explanation=f"Pattern-based classification: {best.explanation}",
                suggestion=best.fix_template,
                confidence=best.score,
            )
        return SimpleAnalyzer().analyze(log)

    def _write_pending(self, log: CiLog, analysis: CiAnalysis, failure_chunk: str) -> None:
        """Analiz sonucunu pending_analyses koleksiyonuna yazar."""
        if os.getenv("ENABLE_PENDING_STORE", "1") != "1":
            return
        try:
            from .rag.pending_store import PendingStore, PendingRecord

            record = PendingRecord(
                run_id=log.run_id,
                pipeline_name=log.pipeline_name,
                branch=log.branch,
                platform=log.platform,
                commit_sha=log.commit_sha,
                main_category=analysis.main_category,
                category=analysis.category,
                root_cause=analysis.root_cause,
                explanation=analysis.explanation,
                suggestion=analysis.suggestion,
                confidence=analysis.confidence,
                failure_chunk=failure_chunk,
                affected_files=analysis.affected_files,
                affected_classes=analysis.affected_classes,
                conflict_type=analysis.conflict_type,
                resolution_strategy=analysis.resolution_strategy,
                jira_task_id=analysis.jira_task_id,
                target_branch=log.target_branch,
                pr_id=log.pr_id,
            )
            store = PendingStore()
            point_id = store.write(record)
            print(f"📝 Pending'e yazıldı: run_id={log.run_id} (point={point_id[:8]}...)")
        except Exception as e:
            print(f"ℹ️ Pending store yazılamadı (analiz etkilenmez): {e}")