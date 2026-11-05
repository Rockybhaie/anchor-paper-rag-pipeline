import pytest

from src import retrieval
from src.build_index import build, load_index, namespace_for
from src.retrieval import keyword_search, retrieve, rrf_merge


def test_rrf_merge_rewards_items_found_by_both_lists():
    merged = rrf_merge([["a", "b", "c"], ["d", "b", "e"]])
    assert merged[0] == "b"
    assert set(merged) == {"a", "b", "c", "d", "e"}


def test_rrf_merge_uses_position_with_k_60():
    merged = rrf_merge([["a", "b"], ["b", "c"]], k=60)
    assert merged == ["b", "a", "c"]


@pytest.fixture
def index(corpus, store, config, monkeypatch):
    from tests.conftest import fake_embed

    monkeypatch.setattr(retrieval, "embed_texts", fake_embed)
    # Passages mentioning robots score highest, so reranking changes the order visibly.
    monkeypatch.setattr(
        retrieval,
        "rerank_scores",
        lambda question, texts: [float("robot" in text) for text in texts],
    )
    build("titans", config, store)
    return load_index("titans", config)


def test_keyword_search_finds_the_paper_with_the_rare_word(index):
    ids = keyword_search("robot control", index, 3)
    assert ids
    assert all(index.by_id[item]["paper_id"] == "01_citing" for item in ids)


def test_keyword_search_returns_nothing_for_unknown_words(index):
    assert keyword_search("zzzz qqqq", index, 3) == []


@pytest.mark.parametrize("mode", ["semantic", "keyword", "hybrid", "hybrid_rerank"])
def test_retrieve_returns_final_k_passages_in_every_mode(index, store, config, mode):
    config["retrieval"].update(mode=mode, final_k=2)
    result = retrieve("robot control memory", index, config, store)
    assert len(result.passages) == 2
    assert all(passage["text"] for passage in result.passages)


def test_rerank_puts_the_best_match_first(index, store, config):
    config["retrieval"].update(mode="hybrid_rerank", final_k=1)
    result = retrieve("titans memory", index, config, store)
    assert "robot" in result.passages[0]["text"]
    assert result.passages[0]["score"] == 1.0


def test_retrieve_rejects_an_unknown_mode(index, store, config):
    config["retrieval"]["mode"] = "psychic"
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        retrieve("question", index, config, store)


def test_semantic_search_uses_the_namespace_for_this_setup(index, store, config):
    assert namespace_for("titans", config) in store.namespaces
