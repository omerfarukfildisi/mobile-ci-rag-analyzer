# src/agentops/knowledge/error_knowledge.py

import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any


# Basit bir match modeli: kategori + skor + eşleşen patternler
@dataclass
class CategoryMatch:
    name: str
    score: float
    matched_patterns: List[str]
    platforms: List[str]
    explanation: str
    fix_template: str


def _load_ontology() -> Dict[str, Any]:
    """
    error_ontology.json dosyasını yükler.
    """
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, "error_ontology.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Modül import edildiğinde bir defa yükleyelim
ERROR_ONTOLOGY = _load_ontology()


def match_categories(raw_log: str, top_k: int = 5) -> List[CategoryMatch]:
    """
    Basit pattern bazlı bir kategori tahmini yapar.
    Her kategori için pattern eşleşme oranına göre bir skor hesaplar.
    """
    text = raw_log or ""
    text_lower = text.lower()

    matches: List[CategoryMatch] = []

    for name, data in ERROR_ONTOLOGY.items():
        patterns = data.get("patterns", [])
        matched = []

        for pattern in patterns:
            # Basit case-insensitive arama – regex özel karakterlerini de destekler
            if re.search(pattern, text, flags=re.IGNORECASE):
                matched.append(pattern)

        if not matched:
            continue

        # Skor: matched pattern sayısı / toplam pattern sayısı (0-1 arası)
        score = len(matched) / max(len(patterns), 1)

        matches.append(
            CategoryMatch(
                name=name,
                score=score,
                matched_patterns=matched,
                platforms=data.get("platforms", []),
                explanation=data.get("explanation", ""),
                fix_template=data.get("fix_template", "")
            )
        )

    # Skora göre sırala ve top_k döndür
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches[:top_k]
