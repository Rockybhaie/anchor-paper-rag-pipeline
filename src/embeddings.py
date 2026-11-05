"""Turn text into vectors with OpenAI or with a small model that runs on this machine."""

from __future__ import annotations

import os
from functools import lru_cache

# Harmless on Windows, where the model cache cannot use symlinks without extra settings.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402
from fastembed import TextEmbedding  # noqa: E402

from src.config import ROOT  # noqa: E402
from src.llm import NETWORK_ERRORS, estimate_cost, get_client, with_retries  # noqa: E402

DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}
BATCH_SIZE = 100
CACHE_DIR = ROOT / ".cache" / "fastembed"


@lru_cache(maxsize=2)
def _local_model(model: str) -> TextEmbedding:
    return TextEmbedding(model_name=model, cache_dir=str(CACHE_DIR))


def embed_texts(texts: list[str], embedding: dict) -> tuple[np.ndarray, float]:
    """Return one normalized vector per text and the estimated cost in dollars."""
    cleaned = [" ".join(text.split()) or "empty" for text in texts]
    if embedding["provider"] == "local":
        vectors = np.array(list(_local_model(embedding["model"]).embed(cleaned)), dtype=np.float32)
        cost = 0.0
    elif embedding["provider"] == "openai":
        client = get_client("openai")
        rows, tokens = [], 0
        for start in range(0, len(cleaned), BATCH_SIZE):
            batch = cleaned[start : start + BATCH_SIZE]
            response = with_retries(
                lambda items=batch: client.embeddings.create(
                    model=embedding["model"], input=items
                ),
                NETWORK_ERRORS,
            )
            rows.extend(item.embedding for item in response.data)
            tokens += response.usage.prompt_tokens
        vectors = np.array(rows, dtype=np.float32)
        cost = estimate_cost(embedding["model"], tokens)
    else:
        raise ValueError(f"Unknown embedding provider: {embedding['provider']}")
    norms = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return vectors / norms, cost
