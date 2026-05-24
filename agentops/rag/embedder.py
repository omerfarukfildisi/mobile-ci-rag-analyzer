# src/agentops/rag/embedder.py
"""
Embedding modülü.
Model: nomic-embed-text (Ollama üzerinden)
Boyut: 768

Ollama'da model kurulumu:
    ollama pull nomic-embed-text
"""

from __future__ import annotations

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
VECTOR_SIZE = 768


def _extract_embedding(data: dict) -> list[float] | None:
    """Ollama embed yanıtından embedding vektörünü güvenli çıkarır."""
    emb = data.get("embedding")
    if isinstance(emb, list):
        return emb

    embs = data.get("embeddings")
    if isinstance(embs, list) and embs:
        first = embs[0]
        if isinstance(first, list):
            return first

    return None


def embed_text(text: str) -> list[float]:
    """
    Verilen metni embed eder ve vektör döndürür.
    Yeni Ollama'da /api/embed, eski sürümlerde /api/embeddings endpoint'ini dener.
    """
    if requests is None:
        raise ImportError("requests is required for embed_text. Install it with: pip install requests")
    text = text.strip()
    if not text:
        return [0.0] * VECTOR_SIZE

    # Yeni endpoint (Ollama 0.3+)
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        vec = _extract_embedding(resp.json())
        if vec:
            return vec
    except Exception:
        pass

    # Eski endpoint (geriye dönük uyumluluk)
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        vec = _extract_embedding(resp.json())
        if vec:
            return vec
    except Exception as e:
        print(f"❌ Embedding hatası: {e}")

    # Fallback: sıfır vektör
    return [0.0] * VECTOR_SIZE


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Birden fazla metni embed eder.
    Şimdilik sıralı, ileride paralel yapılabilir.
    """
    return [embed_text(t) for t in texts]
