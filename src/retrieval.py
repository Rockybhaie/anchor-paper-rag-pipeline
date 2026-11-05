"""Find passages for a question using keywords, meaning, or both with a second reranking check."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from fastembed.rerank.cross_encoder import TextCrossEncoder

from src.build_index import LoadedIndex, tokenize
from src.embeddings import CACHE_DIR, embed_texts

RRF_K = 60
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
MODES = {"semantic", "keyword", "hybrid", "hybrid_rerank"}


@dataclass
class RetrievalResult:
    passages: list[dict]
    latency_s: float
    cost: float


def rrf_merge(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Merge ranked lists by position, since BM25 and cosine scores use different scales."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank + 1)
    return sorted(scores, key=lambda item: scores[item], reverse=True)


def keyword_search(question: str, index: LoadedIndex, top_k: int) -> list[str]:
    scores = index.bm25.get_scores(tokenize(question))
    order = np.argsort(scores)[::-1][:top_k]
    return [index.chunks[i]["id"] for i in order if scores[i] > 0]


def semantic_search(
    question: str, index: LoadedIndex, store, embedding: dict, top_k: int
) -> tuple[list[str], float]:
    vectors, cost = embed_texts([question], embedding)
    matches = store.query(index.manifest["namespace"], vectors[0], top_k)
    return [chunk_id for chunk_id, _ in matches if chunk_id in index.by_id], cost


@lru_cache(maxsize=1)
def _reranker() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=RERANK_MODEL, cache_dir=str(CACHE_DIR))


def rerank_scores(question: str, texts: list[str]) -> list[float]:
    """Read the question together with each passage and score how well they match."""
    return [float(score) for score in _reranker().rerank(question, texts)]


def retrieve(question: str, index: LoadedIndex, config: dict, store) -> RetrievalResult:
    """Return the final passages for a question, best first."""
    settings = config["retrieval"]
    mode, candidates, final_k = settings["mode"], settings["candidates"], settings["final_k"]
    if mode not in MODES:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    start = time.perf_counter()
    cost = 0.0
    if mode == "keyword":
        ranked = keyword_search(question, index, candidates)
    else:
        semantic, cost = semantic_search(question, index, store, config["embedding"], candidates)
        if mode == "semantic":
            ranked = semantic
        else:
            ranked = rrf_merge([semantic, keyword_search(question, index, candidates)])[:candidates]

    scores = [None] * len(ranked)
    if mode == "hybrid_rerank" and ranked:
        raw_scores = rerank_scores(question, [index.by_id[item]["text"] for item in ranked])
        pairs = sorted(zip(raw_scores, ranked, strict=True), key=lambda p: p[0], reverse=True)
        scores, ranked = [p[0] for p in pairs], [p[1] for p in pairs]

    passages = [
        index.by_id[item] | {"score": score}
        for item, score in list(zip(ranked, scores, strict=True))[:final_k]
    ]
    return RetrievalResult(passages, time.perf_counter() - start, cost)
