"""The question-answering path used by the API: load indexes, find passages, write an answer."""

from __future__ import annotations

import time

from src.build_index import LoadedIndex, load_index, setup_name
from src.config import load_anchors
from src.generation import generate
from src.retrieval import retrieve
from src.vector_store import PineconeStore


def unique_sources(passages: list[dict]) -> list[dict]:
    """List each source paper once, in the order it was first retrieved."""
    sources = {}
    for passage in passages:
        sources.setdefault(
            passage["paper_id"],
            {key: passage.get(key) for key in ("title", "authors", "year")},
        )
    return list(sources.values())


class RagEngine:
    def __init__(self, config: dict, store=None, anchors: list[str] | None = None):
        self.config = config
        self.store = store or PineconeStore(**{
            key: value for key, value in config.get("index", {}).items() if key != "provider"
        })
        names = load_anchors() if anchors is None else anchors
        self.indexes: dict[str, LoadedIndex] = {name: load_index(name, config) for name in names}

    def explain(self, anchor_id: str, question: str) -> dict:
        result = retrieve(question, self.indexes[anchor_id], self.config, self.store)
        return {
            "anchor_id": anchor_id,
            "question": question,
            "passages": [
                {
                    "rank": rank,
                    **{
                        key: passage.get(key)
                        for key in ("text", "paper_id", "title", "authors", "year", "score")
                    },
                }
                for rank, passage in enumerate(result.passages, start=1)
            ],
            "setup": setup_name(self.config),
        }

    def predict(self, anchor_id: str, question: str) -> dict:
        start = time.perf_counter()
        result = retrieve(question, self.indexes[anchor_id], self.config, self.store)
        answer = generate(question, result.passages, self.config["generation"])
        return {
            "anchor_id": anchor_id,
            "answer": answer.text,
            "sources": unique_sources(result.passages),
            "model": self.config["generation"]["model"],
            "setup": setup_name(self.config),
            "latency_ms": int((time.perf_counter() - start) * 1000),
        }
