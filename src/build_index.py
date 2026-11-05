"""Build the search index for each anchor. This is the RAG equivalent of a training script.

It reads prepared text, splits and embeds it, uploads the vectors to Pinecone, and saves a versioned
artifact holding the passages and a manifest. Evaluation is done by src/evaluate.py.

Example: python -m src.build_index --anchor titans --config serving
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.chunking import chunking_name, split_text
from src.config import ANCHOR_DATA, ARTIFACTS, PROCESSED, load_anchors, load_config
from src.embeddings import DIMENSIONS, embed_texts
from src.vector_store import PineconeStore

ARTIFACT_VERSION = "v1"
SEMANTIC_EMBEDDING = {"provider": "local", "model": "sentence-transformers/all-MiniLM-L6-v2"}


def setup_name(config: dict) -> str:
    """Name a setup clearly, such as recursive-400-80__text-embedding-3-small__v1."""
    model = config["embedding"]["model"].rsplit("/", 1)[-1]
    return f"{chunking_name(config['chunking'])}__{model}__{ARTIFACT_VERSION}"


def artifact_dir(anchor_id: str, config: dict) -> Path:
    return ARTIFACTS / anchor_id / setup_name(config)


def namespace_for(anchor_id: str, config: dict) -> str:
    return f"{anchor_id}__{setup_name(config)}"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_papers(anchor_id: str) -> list[tuple[dict, str]]:
    """Return each kept paper's manifest entry with its prepared text."""
    manifest_path = ANCHOR_DATA / anchor_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    papers = []
    for paper in manifest["papers"]:
        text_path = PROCESSED / anchor_id / f"{paper['id']}.txt"
        if text_path.exists():
            papers.append((paper, text_path.read_text(encoding="utf-8")))
    if not papers:
        raise FileNotFoundError(f"No prepared text for {anchor_id}. Run src.prepare_data first.")
    return papers


def make_chunks(papers: list[tuple[dict, str]], chunking: dict) -> list[dict]:
    """Split every paper and attach the details needed to cite each passage."""

    def embed(texts: list[str]):
        return embed_texts(texts, SEMANTIC_EMBEDDING)[0]

    chunks = []
    for paper, text in papers:
        for position, passage in enumerate(split_text(text, chunking, embed)):
            chunks.append(
                {
                    "id": f"{paper['id']}-{position}",
                    "text": passage,
                    "paper_id": paper["id"],
                    "title": paper["title"],
                    "authors": paper.get("authors"),
                    "year": paper.get("year"),
                    "category": paper.get("category"),
                }
            )
    return chunks


def validate_chunks(chunks: list[dict]) -> None:
    if not chunks:
        raise ValueError("No passages were created.")
    ids = [chunk["id"] for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Passage ids are not unique.")
    if any(not chunk["text"].strip() for chunk in chunks):
        raise ValueError("Empty passages were created.")


def build(anchor_id: str, config: dict, store=None) -> dict:
    """Build one anchor's artifact and upload its vectors unless they are already there."""
    index_settings = config.get("index", {})
    store = store or PineconeStore(
        cloud=index_settings.get("cloud", "aws"), region=index_settings.get("region", "us-east-1")
    )
    chunks = make_chunks(load_papers(anchor_id), config["chunking"])
    validate_chunks(chunks)

    namespace = namespace_for(anchor_id, config)
    dimension = DIMENSIONS[config["embedding"]["model"]]
    folder = artifact_dir(anchor_id, config)
    lines = "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks)
    passages_sha256 = hashlib.sha256(lines.encode("utf-8")).hexdigest()
    previous_path = folder / "manifest.json"
    previous_sha256 = passages_sha256
    if previous_path.exists():
        previous_sha256 = json.loads(previous_path.read_text("utf-8"))["passages_sha256"]

    cost = 0.0
    existing = store.count(namespace, dimension)
    if existing == len(chunks) and previous_sha256 == passages_sha256:
        print(f"{namespace}: vectors already in Pinecone, upload skipped")
    else:
        # Removes partial uploads and vectors for passages that have since changed.
        if existing:
            store.clear(namespace, dimension)
        vectors, cost = embed_texts([chunk["text"] for chunk in chunks], config["embedding"])
        store.upsert(namespace, [chunk["id"] for chunk in chunks], vectors)
        store.wait_for_count(namespace, dimension, len(chunks))

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "chunks.jsonl").write_text(lines, encoding="utf-8")
    manifest = {
        "anchor": anchor_id,
        "setup": setup_name(config),
        "version": ARTIFACT_VERSION,
        "namespace": namespace,
        "dimension": dimension,
        "chunking": config["chunking"],
        "embedding": config["embedding"],
        "passages": len(chunks),
        "papers": len({chunk["paper_id"] for chunk in chunks}),
        "passages_sha256": passages_sha256,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"{anchor_id}: {manifest['passages']} passages from {manifest['papers']} papers "
        f"in {folder.name}, estimated cost ${cost:.4f}"
    )
    return manifest | {"cost": cost}


@dataclass
class LoadedIndex:
    """A saved index, ready for searching."""

    anchor_id: str
    manifest: dict
    chunks: list[dict]
    by_id: dict[str, dict]
    bm25: BM25Okapi


def load_index(anchor_id: str, config: dict) -> LoadedIndex:
    """Load a saved artifact and check that it is complete."""
    folder = artifact_dir(anchor_id, config)
    manifest_path = folder / "manifest.json"
    chunks_path = folder / "chunks.jsonl"
    if not manifest_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(f"No index at {folder}. Run src.build_index first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lines = chunks_path.read_text(encoding="utf-8")
    if hashlib.sha256(lines.encode("utf-8")).hexdigest() != manifest["passages_sha256"]:
        raise ValueError(f"The passages in {folder} do not match the manifest. Rebuild the index.")
    # Not splitlines(): PDF text contains characters like U+2028 that it would also split on.
    chunks = [json.loads(line) for line in lines.split("\n")]
    return LoadedIndex(
        anchor_id=anchor_id,
        manifest=manifest,
        chunks=chunks,
        by_id={chunk["id"]: chunk for chunk in chunks},
        bm25=BM25Okapi([tokenize(chunk["text"]) for chunk in chunks]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    anchors = sorted(load_anchors())
    parser.add_argument("--anchor", choices=anchors, help="Build one anchor. Default is all.")
    parser.add_argument("--config", default="serving", help="Config name in configs/.")
    args = parser.parse_args()
    config = load_config(args.config)
    for anchor_id in [args.anchor] if args.anchor else anchors:
        build(anchor_id, config)


if __name__ == "__main__":
    main()
