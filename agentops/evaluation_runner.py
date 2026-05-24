from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .analyzer import OllamaAnalyzer
from .log_models import CiLog


@dataclass
class EvalConfig:
    name: str
    model: str
    rag_enabled: bool
    rag_all: bool = False
    think: bool | None = None
    gt_mode: str = "legacy"


class EvalAnalyzer(OllamaAnalyzer):
    """Evaluation-only analyzer.

    Keeps production defaults intact while allowing eval-only knobs
    (e.g., think=false for reasoning models that otherwise fill only
    the `thinking` field).
    """

    def __init__(self, *args: Any, think: bool | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.eval_think = think


class RAGAllAnalyzer(EvalAnalyzer):
    """Evaluation analyzer that bypasses routing and queries ALL RAG modules."""

    def __init__(self, *args: Any, think: bool | None = None, **kwargs: Any) -> None:
        super().__init__(*args, think=think, **kwargs)
        self.rag_all_mode = True

    def _build_rag_context(self, guessed_category: str, query: str, log: "CiLog") -> str:
        enabled = os.getenv("ENABLE_RAG_ROUTING", "1").lower() not in {"0", "false", "no"}
        if not enabled:
            return ""
        try:
            from .rag.router import RAGRouter
        except Exception as e:
            print(f"ℹ️ RAG import atlandı: {e}")
            return ""
        try:
            router = RAGRouter(
                es_url=os.getenv("ES_URL", "http://localhost:9200"),
            )
            # RAG-All baseline: no routing — query every module with mild
            # context. Should be a middle-ground between No-RAG (no help)
            # and RAG-Routing (targeted, filtered help).
            top_k = int(os.getenv("RAG_ALL_TOP_K", "5"))
            summary: list[str] = []
            # RAG-All baseline: query EVERY module (including dependency) with
            # NO category/platform filtering and a generous top_k. This is the
            # honest "throw everything at the prompt" comparison: high recall,
            # high noise, no routing logic.
            for module, retriever, formatter in [
                ("historical",  lambda: router.historical.retrieve(query=query, category=None, platform=None, top_k=top_k), lambda r: f"[historical] {r.fix_description}"),
                ("platform",    lambda: router.platform.retrieve(query=query, platform=None, top_k=top_k),                  lambda r: f"[platform] {r.title}: {r.fix}"),
                ("conflict",    lambda: router.conflict.retrieve(query=query, platform=None, top_k=top_k),                  lambda r: f"[conflict] {r.conflict_type}: {r.resolution}"),
                ("dependency",  lambda: router.dependency.retrieve(query=query, platform=None, top_k=top_k),                lambda r: f"[dependency] {r.package_name}: {r.resolution}"),
            ]:
                try:
                    for r in retriever():
                        summary.append(formatter(r))
                except Exception as e:
                    print(f"ℹ️ RAG {module} atlandı: {e}")
            rag_context = "\n".join(summary).strip()
            max_chars = int(os.getenv("RAG_ALL_CONTEXT_MAX_CHARS", "2500"))
            return rag_context[:max_chars] if len(rag_context) > max_chars else rag_context
        except Exception as e:
            print(f"ℹ️ RAG-All routing atlandı: {e}")
            return ""

    def _call_ollama(self, model: str, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
            "options": {
                "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
                "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "180")),
            },
        }
        if os.getenv("OLLAMA_FORCE_JSON", "1").lower() not in {"0", "false", "no"}:
            payload["format"] = "json"
        if self.eval_think is not None:
            payload["think"] = self.eval_think

        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()


class Evaluator:
    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = Path(dataset_path)

    def load_rows(self, limit: int) -> list[dict[str, Any]]:
        rows = [json.loads(line) for line in self.dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if limit > 0:
            rows = rows[:limit]
        return rows

    @staticmethod
    def gt_main_category_legacy(row: dict[str, Any]) -> str:
        # Explicit overrides for error_types that defeat keyword matching.
        # These are aligned with each row's actual `root_cause` semantics so
        # the heuristic GT does not penalise correct model predictions.
        _OVERRIDES: dict[str, str] = {
            "export_failed": "signing",
            # Agent / daemon / process termination → resource events
            "agent_connection_drop": "resource",
            "agent_channel_disconnected": "resource",
            "gradle_daemon_connection_error": "resource",
            "gradle_r8_hung_no_progress": "resource",
            "gradle_journal_lock_contention": "resource",
            "gradle_daemon_jvm_crash_during_d8_dexing": "resource",
            "gradle_cache_lock_timeout_concurrent_build": "resource",
            "manual_abort_gradle_clean": "resource",
            "manual_abort_before_gradle": "resource",
            "manual_abort_during_r8_minification": "resource",
            "kotlin_daemon_crash_on_manual_abort": "resource",
            "upstream_pipeline_cancelled_during_gradle_build": "resource",
            "jenkins_agent_offline_cleanup_plus_ai_review_http500": "resource",
            "git_fetch_disconnect_during_large_clone": "resource",
            # Build/compile-time failures that surface during a merge/branch stage
            "manifest_merger_duplicate_deeplink_destination": "build_artifact",
            "compose_parameter_not_found_breaking_api_change": "build_artifact",
            "git_checkout_null_branch": "build_artifact",
            "branch_not_found": "build_artifact",
            "fastlane_scheme_resolution_failure": "build_artifact",
            "archive_failed_swiftformat_only": "build_artifact",
        }

        md = row.get("metadata", {})
        err = str(md.get("error_type", "")).lower()
        stage = str(md.get("failed_stage", "")).lower()
        root = str(row.get("root_cause", "")).lower()

        if err in _OVERRIDES:
            return _OVERRIDES[err]

        if "merge_conflict" in err or "non_fast_forward" in err:
            return "merge_conflict"
        if any(k in err for k in ["sign", "provision", "keystore"]):
            return "signing"
        if any(k in root for k in ["provisioning_profile", "signing_identity"]):
            return "signing"
        if any(k in err for k in ["auth", "credential", "token", "permission"]):
            return "authentication"
        if any(k in err for k in ["dependency", "spm", "pod", "version_conflict"]):
            return "dependency"
        if any(k in err for k in ["timeout", "disk", "resource", "network", "oom", "daemon_disappeared"]):
            return "resource"
        if any(k in root for k in ["agent_offline", "channel_terminated", "dns_failure", "no_space_left"]):
            return "resource"
        if any(k in err for k in ["build", "compile", "xcode", "kotlin", "archive", "lint"]):
            return "build_artifact"
        if "merge" in stage:
            return "merge_conflict"
        if "sign" in stage:
            return "signing"
        return "build_artifact"

    @staticmethod
    def gt_main_category_semantic(row: dict[str, Any]) -> str:
        """Text-semantics ağırlıklı GT: özellikle anonim kullanıcı loglarında daha adil."""
        md = row.get("metadata", {})
        err = str(md.get("error_type", "")).lower()
        stage = str(md.get("failed_stage", "")).lower()
        root = str(row.get("root_cause", "")).lower()
        text = "\n".join([
            str(row.get("log_chunk", "")),
            str(row.get("error_message", "")),
            str(row.get("ground_truth_rca", "")),
            root,
            err,
            stage,
        ]).lower()

        if any(k in text for k in ["merge conflict", "conflict (", "automatic merge failed", "<<<<<<<"]):
            return "merge_conflict"
        if any(k in text for k in ["codesign", "provisioning profile", "keystore", "cfbundleversion", "exportarchive"]):
            return "signing"
        if any(k in text for k in ["unauthorized", "forbidden", "permission denied", "invalid token", "expired token", "authentication failed", "access denied"]):
            return "authentication"
        if any(k in text for k in ["could not resolve package dependencies", "could not resolve host", "podfile", "package.resolved", "swift package manager", "spm", "cocoapods", "dependency resolution"]):
            return "dependency"
        if any(k in text for k in ["daemon disappeared", "kotlin daemon", "outofmemory", "no space left", "agent offline", "requestabortedexception", "timed out", "exit code 143", "aborted by"]):
            return "resource"

        # Legacy kurala geri düş.
        return Evaluator.gt_main_category_legacy(row)

    @staticmethod
    def gt_main_category(row: dict[str, Any], mode: str = "legacy") -> str:
        if mode == "semantic":
            return Evaluator.gt_main_category_semantic(row)
        return Evaluator.gt_main_category_legacy(row)

    @staticmethod
    def to_cilog(row: dict[str, Any], run_id: int | str) -> CiLog:
        md = row.get("metadata", {})
        return CiLog(
            platform=md.get("platform", row.get("platform", "shared")),
            pipeline_name=md.get("job", "eval-job"),
            run_id=str(run_id),
            status=str(md.get("status", "FAILURE")).lower(),
            raw_log=row.get("log_chunk") or row.get("error_message") or "",
            branch="eval/branch",
            commit_sha="eval-sha",
            target_branch="develop",
            pr_id="",
        )

    def run(self, cfg: EvalConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
        os.environ["ENABLE_PENDING_STORE"] = "0"
        os.environ["ENABLE_RAG_ROUTING"] = "1" if cfg.rag_enabled else "0"
        os.environ["OLLAMA_RCA_MODEL"] = cfg.model

        if cfg.rag_all:
            analyzer: EvalAnalyzer = RAGAllAnalyzer(rca_model=cfg.model, think=cfg.think)
        else:
            analyzer = EvalAnalyzer(rca_model=cfg.model, think=cfg.think)

        latencies: list[float] = []
        correct = 0
        parse_fallbacks = 0
        fallback_ids: list[str] = []
        hard_errors = 0
        per_row: list[dict[str, Any]] = []

        for i, row in enumerate(rows, start=1):
            log = self.to_cilog(row, i)
            gt = self.gt_main_category(row, cfg.gt_mode)
            t0 = time.perf_counter()
            try:
                out = analyzer.analyze(log)
                dt = time.perf_counter() - t0
                latencies.append(dt)
                ok = out.main_category == gt
                if ok:
                    correct += 1
                if out.explanation.lower().startswith("pattern-based classification"):
                    parse_fallbacks += 1
                    fallback_ids.append(str(row.get("id", f"row-{i}")))
                per_row.append({
                    "id": str(row.get("id", f"row-{i}")),
                    "platform": row.get("metadata", {}).get("platform", row.get("platform", "")),
                    "error_type": row.get("metadata", {}).get("error_type", ""),
                    "gt": gt,
                    "pred": out.main_category,
                    "ok": ok,
                    "confidence": round(float(getattr(out, "confidence", 0.0) or 0.0), 3),
                    "latency_sec": round(dt, 3),
                })
            except Exception as e:
                hard_errors += 1
                per_row.append({
                    "id": str(row.get("id", f"row-{i}")),
                    "gt": gt,
                    "pred": "",
                    "ok": False,
                    "error": str(e)[:200],
                })

        n = len(latencies)
        if n == 0:
            return {
                "config": cfg.name,
                "n_ok": 0,
                "n_total": len(rows),
                "hard_errors": hard_errors,
                "parse_fallbacks": parse_fallbacks,
                "fallback_ids_count": len(fallback_ids),
                "fallback_ids": fallback_ids,
                "gt_mode": cfg.gt_mode,
                "main_category_acc": 0.0,
                "latency_median_sec": None,
                "latency_p95_sec": None,
                "per_row": per_row,
            }

        ordered = sorted(latencies)
        p95_idx = max(0, math.ceil(0.95 * n) - 1)
        return {
            "config": cfg.name,
            "n_ok": n,
            "n_total": len(rows),
            "hard_errors": hard_errors,
            "parse_fallbacks": parse_fallbacks,
            "fallback_ids_count": len(fallback_ids),
            "fallback_ids": fallback_ids,
            "gt_mode": cfg.gt_mode,
            "main_category_acc": round(correct / n, 4),
            "latency_median_sec": round(statistics.median(ordered), 3),
            "latency_p95_sec": round(ordered[p95_idx], 3),
            "per_row": per_row,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AgentOps evaluation runner")
    p.add_argument("--dataset", default="agentops/data/dataset.jsonl")
    p.add_argument("--limit", type=int, default=5, help="0 means full dataset")
    p.add_argument("--model", default="llama3.1:8b")
    p.add_argument("--name", default="RAG-Routing (8B, ours)")
    p.add_argument("--rag", choices=["on", "off", "all"], default="on")
    p.add_argument("--think", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--gt-mode", choices=["legacy", "semantic"], default="legacy")
    p.add_argument("--out", default="", help="optional json output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    think_val: bool | None
    if args.think == "auto":
        think_val = None
    elif args.think == "on":
        think_val = True
    else:
        think_val = False

    cfg = EvalConfig(
        name=args.name,
        model=args.model,
        rag_enabled=(args.rag in {"on", "all"}),
        rag_all=(args.rag == "all"),
        think=think_val,
        gt_mode=args.gt_mode,
    )
    evaluator = Evaluator(args.dataset)
    rows = evaluator.load_rows(args.limit)
    result = evaluator.run(cfg, rows)
    text = json.dumps(result, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
