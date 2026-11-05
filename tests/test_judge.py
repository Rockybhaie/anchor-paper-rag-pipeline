import numpy as np
import pytest

from src import judge as judge_module
from src.judge import Judge, parse_json_list, score_verdicts
from src.llm import ChatResult

SETTINGS = {"provider": "openai", "model": "gpt-4o-mini", "prompt": "binary"}


def test_parse_json_list_handles_code_fences_and_extra_text():
    assert parse_json_list('```json\n["a", "b"]\n```') == ["a", "b"]
    assert parse_json_list('Here you go: ["a"] Thanks.') == ["a"]


def test_parse_json_list_repairs_latex_in_a_reply():
    reply = r'["The rate \alpha controls updates.", "It uses \mathcal{L} as the loss."]'
    assert parse_json_list(reply) == [
        r"The rate \alpha controls updates.",
        r"It uses \mathcal{L} as the loss.",
    ]


def test_parse_json_list_leaves_valid_escapes_alone():
    assert parse_json_list(r'["He said \"yes\"", "a\\b"]') == ['He said "yes"', r"a\b"]


def test_parse_json_list_rejects_a_reply_with_no_list():
    with pytest.raises(ValueError, match="No JSON list"):
        parse_json_list("I cannot do that.")


def test_score_verdicts_gives_the_share_of_supported_claims():
    verdicts = [{"id": 1, "verdict": "SUPPORTED"}, {"id": 2, "verdict": "NOT_SUPPORTED"}]
    score, results = score_verdicts(verdicts, ["claim one", "claim two"])
    assert score == 0.5
    assert results[0] == {"claim": "claim one", "verdict": "SUPPORTED"}


def test_score_verdicts_counts_partial_as_half():
    verdicts = [{"id": 1, "verdict": "PARTIAL"}, {"id": 2, "verdict": "SUPPORTED"}]
    assert score_verdicts(verdicts, ["one", "two"])[0] == 0.75


def test_score_verdicts_treats_missing_and_unknown_verdicts_as_not_supported():
    score, results = score_verdicts([{"id": 1, "verdict": "MAYBE"}], ["one", "two"])
    assert score == 0.0
    assert [item["verdict"] for item in results] == ["NOT_SUPPORTED", "NOT_SUPPORTED"]


@pytest.fixture
def scripted(monkeypatch):
    """Answer each judge prompt with a fixed reply and record what the judge sent."""
    sent = []
    replies = {
        "Extract all": '["Titans store memory at test time.", "Memory improves recall a lot."]',
        "Context from": (
            '[{"id": 1, "verdict": "SUPPORTED"}, {"id": 2, "verdict": "NOT_SUPPORTED"}]'
        ),
        "Generate exactly": '["What do Titans store?", "How do they recall?", "Why memory?"]',
    }

    def fake_chat(provider, model, messages, temperature, max_tokens):
        content = messages[0]["content"]
        sent.append(content)
        reply = next(text for start, text in replies.items() if content.startswith(start))
        return ChatResult(reply, 1000, 100, 0.1)

    def fake_embed(texts, embedding):
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32), 0.0

    monkeypatch.setattr(judge_module, "chat", fake_chat)
    monkeypatch.setattr(judge_module, "embed_texts", fake_embed)
    return {"sent": sent, "replies": replies}


def test_score_returns_faithfulness_relevancy_and_cost(scripted):
    result = Judge(SETTINGS).score("What do Titans store?", "Titans store memory.", "Some context")

    assert result.faithfulness == 0.5
    assert result.relevancy == pytest.approx(1.0)
    assert len(result.claims) == 2
    assert len(result.generated_questions) == 3
    assert result.cost == pytest.approx(3 * (1000 * 0.15 + 100 * 0.60) / 1_000_000)


def test_judge_sees_the_whole_context_unlike_the_earlier_project(scripted):
    context = "start " + "x" * 5000 + " END_OF_CONTEXT"
    Judge(SETTINGS).score("Question?", "Answer.", context)
    assert any("END_OF_CONTEXT" in message for message in scripted["sent"])


def test_answer_with_no_claims_has_no_faithfulness_score(scripted):
    scripted["replies"]["Extract all"] = "[]"
    result = Judge(SETTINGS).score("Question?", "I cannot answer that.", "Context")
    assert result.faithfulness is None
    assert result.claims == []


def test_rubric_prompt_is_used_when_configured(scripted):
    settings = SETTINGS | {"prompt": "rubric"}
    Judge(settings).score("Question?", "Answer.", "Context")
    assert any("PARTIAL" in message for message in scripted["sent"])
