# src/agentops/log_models.py

from pydantic import BaseModel, field_validator
from typing import Optional

# 6 ana kategori — RAG router bu değerleri bekliyor
MAIN_CATEGORIES = {
    "build_artifact",
    "dependency",
    "merge_conflict",
    "signing",
    "authentication",
    "resource",
}


class CiLog(BaseModel):
    platform: str           # ios | android | shared
    pipeline_name: str
    run_id: int
    status: str             # failure | success
    raw_log: str
    branch: str
    commit_sha: str
    changed_files: list[str] = []   # Değişen dosyalar (analiz bağlamı)
    target_branch: str = ""         # Merge hedefi (develop/main) — conflict analizi için
    pr_id: str = ""                 # Bitbucket PR numarası


class CiAnalysis(BaseModel):
    main_category: str      # build_artifact | dependency | merge_conflict | signing | authentication | resource
    category: str           # sub-kategori (dependency_error, codesign_error vs.)
    root_cause: str
    explanation: str
    suggestion: str
    confidence: float       # 0.0 - 1.0
    affected_files: list[str] = []      # Hatadan etkilenen dosyalar (log parse)
    affected_classes: list[str] = []    # Hatadan etkilenen sınıf/modüller
    conflict_type: str = ""              # spurious | auto_merge | syntactic | semantic | "" (conflict yoksa)
    resolution_strategy: str = ""        # union_merge | regenerate | pick_newer | "" (conflict yoksa)
    jira_task_id: str = ""               # Branch adından parse (MOB-23576, REQ-9127 vs.)

    @field_validator("main_category")
    @classmethod
    def validate_main_category(cls, v: str) -> str:
        if v not in MAIN_CATEGORIES:
            raise ValueError(
                f"main_category '{v}' geçersiz. "
                f"Geçerli değerler: {sorted(MAIN_CATEGORIES)}"
            )
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))
