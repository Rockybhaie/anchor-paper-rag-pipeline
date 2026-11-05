import json

import numpy as np
import pytest

from src import build_index
from src.build_index import artifact_dir, build, load_index, namespace_for, setup_name, tokenize


def test_setup_name_is_versioned_and_readable(config):
    assert setup_name(config) == "recursive-400-80__text-embedding-3-small__v1"
    config["embedding"]["model"] = "sentence-transformers/all-MiniLM-L6-v2"
    assert setup_name(config) == "recursive-400-80__all-MiniLM-L6-v2__v1"


def test_build_saves_artifact_and_uploads_vectors(corpus, store, config):
    manifest = build("titans", config, store)

    folder = artifact_dir("titans", config)
    assert (folder / "chunks.jsonl").exists()
    assert (folder / "manifest.json").exists()
    assert manifest["papers"] == 2
    assert store.count(namespace_for("titans", config), 1536) == manifest["passages"]


def test_build_skips_upload_when_vectors_exist(corpus, store, config, monkeypatch):
    build("titans", config, store)

    def must_not_embed(texts, embedding):
        raise AssertionError("should not embed again")

    monkeypatch.setattr(build_index, "embed_texts", must_not_embed)
    assert build("titans", config, store)["cost"] == 0.0


def test_build_replaces_vectors_when_passages_change(corpus, store, config, monkeypatch):
    build("titans", config, store)
    path = corpus / "processed" / "titans" / "00_anchor.txt"
    # Same passage count, different words: a count check alone would miss this.
    path.write_text(path.read_text("utf-8").replace("Titans", "Tiatns"), encoding="utf-8")
    cleared = []
    monkeypatch.setattr(store, "clear", lambda namespace, dimension: cleared.append(namespace))

    build("titans", config, store)

    assert cleared == [namespace_for("titans", config)]
    assert load_index("titans", config).chunks[0]["text"].startswith("Tiatns")


def test_build_clears_a_partial_upload(corpus, store, config):
    namespace = namespace_for("titans", config)
    store.upsert(namespace, ["leftover"], np.ones((1, 6), dtype=np.float32))

    manifest = build("titans", config, store)

    assert "leftover" not in store.namespaces[namespace][0]
    assert store.count(namespace, 1536) == manifest["passages"]


def test_load_index_returns_searchable_passages(corpus, store, config):
    build("titans", config, store)

    index = load_index("titans", config)

    assert index.anchor_id == "titans"
    assert index.manifest["setup"] == setup_name(config)
    first = index.chunks[0]
    assert index.by_id[first["id"]] == first
    assert first["title"] == "Anchor Paper"
    scores = index.bm25.get_scores(tokenize("robot control"))
    assert index.chunks[int(np.argmax(scores))]["paper_id"] == "01_citing"


def test_load_index_handles_unusual_line_breaks_from_pdfs(corpus, store, config):
    path = corpus / "processed" / "titans" / "01_citing.txt"
    path.write_text(path.read_text("utf-8").replace("tasks.", "tasks \u0085"), encoding="utf-8")
    manifest = build("titans", config, store)
    assert len(load_index("titans", config).chunks) == manifest["passages"]


def test_load_index_explains_a_missing_artifact(corpus, config):
    with pytest.raises(FileNotFoundError, match="Run src.build_index"):
        load_index("titans", config)


def test_load_index_rejects_changed_passages(corpus, store, config):
    build("titans", config, store)
    path = artifact_dir("titans", config) / "chunks.jsonl"
    first_line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    first_line["text"] = "changed"
    path.write_text(json.dumps(first_line), encoding="utf-8")

    with pytest.raises(ValueError, match="do not match"):
        load_index("titans", config)


def test_build_explains_missing_prepared_text(corpus, store, config):
    for path in (corpus / "processed" / "titans").iterdir():
        path.unlink()
    with pytest.raises(FileNotFoundError, match="Run src.prepare_data"):
        build("titans", config, store)
