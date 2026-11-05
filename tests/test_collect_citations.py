import json

import pytest

from src.collect_citations import parse_score, rank_candidates, select_papers, title_key


def candidate(arxiv, category="Core Extension", influential=True, total=12, citations=0):
    return {
        "arxiv": arxiv,
        "title": f"Paper {arxiv}",
        "category": category,
        "influential": influential,
        "total": total,
        "citation_count": citations,
    }


def test_parse_score_adds_the_total_and_handles_code_fences():
    reply = '```json\n{"relevance": 5, "depth": 4, "density": 3, "category": "Application"}\n```'
    assert parse_score(reply)["total"] == 12


def test_parse_score_rejects_a_reply_that_is_not_json():
    with pytest.raises(json.JSONDecodeError):
        parse_score("I cannot score this paper.")


def test_rank_puts_influential_papers_above_high_abstract_scores():
    papers = [
        {"arxiv": "a", "influential": False, "citation_count": 900},
        {"arxiv": "b", "influential": True, "citation_count": 1},
        {"arxiv": "c", "influential": True, "citation_count": 50},
    ]
    scores = {
        "a": {"total": 15, "category": "Core Extension"},
        "b": {"total": 13, "category": "Application"},
        "c": {"total": 13, "category": "Application"},
    }
    assert [p["arxiv"] for p in rank_candidates(papers, scores)] == ["c", "b", "a"]


def test_rank_drops_weak_citations_and_unscored_papers():
    papers = [{"arxiv": "a", "influential": True, "citation_count": 0}, {"arxiv": "b"}]
    assert rank_candidates(papers, {"a": {"total": 15, "category": "Weak Citation"}}) == []


def test_select_keeps_the_category_mix_when_it_can():
    ranked = [candidate(f"core{i}") for i in range(20)]
    ranked += [candidate(f"app{i}", "Application") for i in range(5)]
    ranked += [candidate(f"theory{i}", "Related Theory") for i in range(5)]

    chosen = select_papers(ranked, 14, set(), lambda c: c)

    categories = [paper["category"] for paper in chosen]
    assert categories.count("Core Extension") == 7
    assert categories.count("Application") == 4
    assert categories.count("Related Theory") == 3


def test_select_fills_the_target_when_a_category_runs_short():
    ranked = [candidate(f"core{i}") for i in range(20)]
    assert len(select_papers(ranked, 14, set(), lambda c: c)) == 14


def test_select_skips_papers_used_elsewhere_and_ones_that_fail_checks():
    ranked = [candidate("used"), candidate("bad"), candidate("good")]
    used = {title_key("Paper used")}

    chosen = select_papers(ranked, 2, used, lambda c: None if c["arxiv"] == "bad" else c)

    assert [paper["arxiv"] for paper in chosen] == ["good"]


def test_title_key_ignores_case_and_punctuation():
    assert title_key("Attention Is Not: Retention!") == title_key("attention is not retention")
