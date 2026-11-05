import numpy as np
import pytest

from src.chunking import chunking_name, semantic_split, split_text

PARAGRAPH = "Memory modules store context across long sequences of tokens during inference. "
TEXT = "\n\n".join(PARAGRAPH * 4 for _ in range(6))


@pytest.mark.parametrize(
    "chunking",
    [
        {"method": "fixed", "chunk_size": 512, "overlap": 50},
        {"method": "recursive", "chunk_size": 400, "overlap": 80},
    ],
)
def test_split_text_respects_chunk_size(chunking):
    chunks = split_text(TEXT, chunking)
    assert len(chunks) > 1
    assert all(80 <= len(chunk) <= chunking["chunk_size"] for chunk in chunks)


def test_recursive_split_overlaps_neighbouring_passages():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = split_text(text, {"method": "recursive", "chunk_size": 400, "overlap": 80})
    assert chunks[0].split()[-1] in chunks[1].split()


def test_split_text_drops_very_short_passages():
    assert split_text("Too short.", {"method": "recursive", "chunk_size": 400, "overlap": 80}) == []


def test_split_text_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="Unknown chunking method"):
        split_text(TEXT, {"method": "magic"})


def test_semantic_chunking_needs_an_embedding_function():
    with pytest.raises(ValueError, match="embedding function"):
        split_text(TEXT, {"method": "semantic"})


def test_semantic_split_breaks_where_the_topic_changes():
    sentences = ["Cats sleep.", "Cats purr.", "Cats play.", "Rain falls.", "Rain pours."]

    def embed(texts):
        return np.array([[1.0, 0.0] if "Cats" in t else [0.0, 1.0] for t in texts])

    assert semantic_split(" ".join(sentences), embed) == [
        "Cats sleep. Cats purr. Cats play.",
        "Rain falls. Rain pours.",
    ]


def test_chunking_name_is_clear():
    recursive = {"method": "recursive", "chunk_size": 400, "overlap": 80}
    assert chunking_name(recursive) == "recursive-400-80"
    assert chunking_name({"method": "semantic"}) == "semantic-p70"
