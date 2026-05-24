# src/agentops/feedback_service.py
"""
Feedback Service — FastAPI

n8n webhook'undan gelen PR merge sinyallerini alır,
pending_analyses → historical_fixes promote işlemini yapar.

Kullanım:
    cd src && uvicorn agentops.feedback_service:app --port 8000
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .log_models import CiLog
from .analyzer import OllamaAnalyzer

app = FastAPI(title="AgentOps Feedback", version="0.1.0")


class FeedbackRequest(BaseModel):
    run_id: int
    status: str  # "merged" | "declined" | "deleted"
    pr_title: str = ""
    changed_files: list[str] = []


class FeedbackResponse(BaseModel):
    promoted: bool
    run_id: int
    message: str = ""


class ConflictAnalyzeRequest(BaseModel):
    source_branch: str
    target_branch: str = ""
    pr_id: str = ""
    platform: str = "ios"
    pipeline_name: str = "conflict-pipeline"
    run_id: int
    commit_sha: str = ""
    conflict_files: list[str] = []
    conflict_contents: dict[str, str] = {}


class ConflictAnalyzeResponse(BaseModel):
    run_id: int
    source_branch: str
    target_branch: str
    pr_id: str
    analysis: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/pending")
def list_pending():
    from .rag.pending_store import PendingStore

    store = PendingStore()
    items = store.list_pending(limit=50)
    return {"count": len(items), "items": items}


@app.post("/feedback", response_model=FeedbackResponse)
def receive_feedback(req: FeedbackRequest):
    """
    n8n veya manuel curl ile çağrılır.
    status="merged" ise pending → historical promote yapar.
    """
    if req.status != "merged":
        return FeedbackResponse(
            promoted=False,
            run_id=req.run_id,
            message=f"Status '{req.status}' — promote atlandı.",
        )

    from .rag.pending_store import PendingStore
    from .rag.historical_fix import HistoricalFixRAG, FixRecord

    store = PendingStore()
    pending = store.get_by_run_id(req.run_id)

    if pending is None:
        raise HTTPException(
            status_code=404,
            detail=f"run_id={req.run_id} pending_analyses'ta bulunamadı.",
        )

    # Fix description: PR title varsa onu kullan, yoksa suggestion fallback
    fix_desc = req.pr_title if req.pr_title else pending.suggestion

    # Tags: run_id + changed_files
    tags = [f"run_id:{req.run_id}"]
    tags.extend(req.changed_files)

    fix_record = FixRecord(
        failure_log=pending.failure_chunk,
        fix_description=fix_desc,
        fix_diff="",  # ileride Bitbucket API'den çekilecek
        category=pending.main_category,
        platform=pending.platform,
        source="agentops",
        tags=tags,
    )

    rag = HistoricalFixRAG()
    rag.ensure_collection()
    point_id = rag.index(fix_record)

    # Pending'ten sil
    store.delete_by_run_id(req.run_id)

    return FeedbackResponse(
        promoted=True,
        run_id=req.run_id,
        message=f"historical_fixes'a promote edildi (point={point_id[:8]}...).",
    )


@app.post("/analyze-conflict", response_model=ConflictAnalyzeResponse)
def analyze_conflict(req: ConflictAnalyzeRequest):
    """
    Jenkins/n8n tarafından conflict payload'ı ile çağrılır.
    Conflict dosya içeriklerinden CiLog üretip analyzer ile zengin conflict analizi döner.
    """
    # Raw log'u, conflict dosya içeriklerini de kapsayacak şekilde oluştur.
    sections = [
        "[Conflict Analysis Payload]",
        f"Source Branch: {req.source_branch}",
        f"Target Branch: {req.target_branch}",
        f"PR ID: {req.pr_id}",
        f"Conflict Files: {', '.join(req.conflict_files)}",
        "",
    ]

    for path, content in req.conflict_contents.items():
        sections.append(f"--- FILE: {path} ---")
        sections.append(content)
        sections.append("")

    raw_log = "\n".join(sections).strip()

    log = CiLog(
        platform=req.platform,
        pipeline_name=req.pipeline_name,
        run_id=req.run_id,
        status="failure",
        raw_log=raw_log,
        branch=req.source_branch,
        commit_sha=req.commit_sha or "unknown",
        changed_files=req.conflict_files,
        target_branch=req.target_branch,
        pr_id=req.pr_id,
    )

    analyzer = OllamaAnalyzer()
    analysis = analyzer.analyze(log)

    return ConflictAnalyzeResponse(
        run_id=req.run_id,
        source_branch=req.source_branch,
        target_branch=req.target_branch,
        pr_id=req.pr_id,
        analysis=analysis.model_dump(),
    )


def promote_by_run_id(
    run_id: int,
    pr_title: str = "",
    changed_files: Optional[list[str]] = None,
) -> dict:
    """
    CLI'dan çağrılabilir promote fonksiyonu.
    feedback endpoint'iyle aynı mantığı kullanır.
    """
    from .rag.pending_store import PendingStore
    from .rag.historical_fix import HistoricalFixRAG, FixRecord

    store = PendingStore()
    pending = store.get_by_run_id(run_id)

    if pending is None:
        return {"promoted": False, "message": f"run_id={run_id} bulunamadı."}

    fix_desc = pr_title if pr_title else pending.suggestion
    tags = [f"run_id:{run_id}"]
    if changed_files:
        tags.extend(changed_files)

    fix_record = FixRecord(
        failure_log=pending.failure_chunk,
        fix_description=fix_desc,
        fix_diff="",
        category=pending.main_category,
        platform=pending.platform,
        source="agentops",
        tags=tags,
    )

    rag = HistoricalFixRAG()
    rag.ensure_collection()
    point_id = rag.index(fix_record)

    store.delete_by_run_id(run_id)

    return {
        "promoted": True,
        "run_id": run_id,
        "point_id": point_id,
        "message": f"historical_fixes'a promote edildi.",
    }
