import json

import numpy as np
import pytest

from src import build_index
from src.config import load_config
from src.vector_store import InMemoryStore

# The citing paper is shorter so its words stay rare enough for BM25 to score them above zero.
PAPERS = {
    "00_anchor": (
        "Anchor Paper",
        "Titans keep a long term memory that is updated at test time. ",
        40,
    ),
    "01_citing": (
        "Citing Paper",
        "This work applies test time memory to robot control tasks. ",
        10,
    ),
}


def fake_embed(texts, embedding=None):
    """Give related words similar vectors, with no model or network."""
    vocabulary = ["memory", "titans", "robot", "control", "test", "time"]
    vectors = np.array(
        [[text.lower().count(word) + 0.01 for word in vocabulary] for text in texts],
        dtype=np.float32,
    )
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True), 0.0


@pytest.fixture
def config():
    baseline = load_config("baseline")
    baseline["embedding"] = {"provider": "openai", "model": "text-embedding-3-small"}
    return baseline


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Create a tiny prepared corpus and point the code at it."""
    anchor_folder = tmp_path / "anchors" / "titans"
    anchor_folder.mkdir(parents=True)
    processed = tmp_path / "processed" / "titans"
    processed.mkdir(parents=True)
    manifest = []
    for paper_id, (title, sentence, repeats) in PAPERS.items():
        manifest.append(
            {"id": paper_id, "title": title, "authors": "A. Author", "year": 2026, "category": "X"}
        )
        (processed / f"{paper_id}.txt").write_text(sentence * repeats, encoding="utf-8")
    (anchor_folder / "manifest.json").write_text(json.dumps({"papers": manifest}), encoding="utf-8")
    monkeypatch.setattr(build_index, "ANCHOR_DATA", tmp_path / "anchors")
    monkeypatch.setattr(build_index, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(build_index, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(build_index, "embed_texts", fake_embed)
    return tmp_path


@pytest.fixture
def store():
    return InMemoryStore()
