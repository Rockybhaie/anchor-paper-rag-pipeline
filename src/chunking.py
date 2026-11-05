"""Split cleaned text into passages using one of the three strategies from the earlier project."""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

MIN_CHUNK_CHARS = 80
SEMANTIC_PERCENTILE = 70


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def semantic_split(
    text: str, embed: Callable[[list[str]], np.ndarray], percentile: float = SEMANTIC_PERCENTILE
) -> list[str]:
    """Start a new passage where the distance between neighbouring sentences is unusually large."""
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return sentences
    vectors = embed(sentences)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    distances = 1 - np.sum(vectors[:-1] * vectors[1:], axis=1)
    threshold = np.percentile(distances, percentile)

    passages, current = [], [sentences[0]]
    for sentence, distance in zip(sentences[1:], distances, strict=True):
        if distance > threshold:
            passages.append(" ".join(current))
            current = []
        current.append(sentence)
    passages.append(" ".join(current))
    return passages


def split_text(
    text: str, chunking: dict, embed: Callable[[list[str]], np.ndarray] | None = None
) -> list[str]:
    """Split text with the configured method and drop passages too short to be useful."""
    method = chunking["method"]
    if method == "fixed":
        pieces = CharacterTextSplitter(
            chunk_size=chunking["chunk_size"],
            chunk_overlap=chunking["overlap"],
            separator="\n",
        ).split_text(text)
    elif method == "recursive":
        pieces = RecursiveCharacterTextSplitter(
            chunk_size=chunking["chunk_size"],
            chunk_overlap=chunking["overlap"],
            separators=["\n\n", "\n", ". ", " "],
        ).split_text(text)
    elif method == "semantic":
        if embed is None:
            raise ValueError("Semantic chunking needs an embedding function.")
        pieces = semantic_split(text, embed)
    else:
        raise ValueError(f"Unknown chunking method: {method}")
    return [piece.strip() for piece in pieces if len(piece.strip()) >= MIN_CHUNK_CHARS]


def chunking_name(chunking: dict) -> str:
    if chunking["method"] == "semantic":
        return f"semantic-p{SEMANTIC_PERCENTILE}"
    return f"{chunking['method']}-{chunking['chunk_size']}-{chunking['overlap']}"
