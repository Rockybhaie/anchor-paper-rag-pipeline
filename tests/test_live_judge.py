import pytest
from fastapi.testclient import TestClient

from src import generation, retrieval
from src.app import create_app
from src.build_index import build
from src.judge import JudgeResult
from src.live_judge import LiveJudge
from src.llm import ChatResult
from src.rag import RagEngine
from tests.conftest import fake_embed

QUESTION = {"anchor_id": "titans", "question": "How is memory used for robot control?"}


class FakeJudge:
    """Stands in for the LLM judge and records what it was asked to score."""

    calls: list[tuple[str, str, str]] = []

    def __init__(self, settings):
        self.settings = settings

    def score(self, question, answer, context):
        FakeJudge.calls.append((question, answer, context))
        return JudgeResult(faithfulness=0.9, relevancy=0.8, cost=0.001)


class FailingJudge(FakeJudge):
    def score(self, question, answer, context):
        raise ValueError("unreadable judge reply")


@pytest.fixture(autouse=True)
def clear_calls():
    FakeJudge.calls = []


@pytest.fixture
def make_client(corpus, store, config, monkeypatch):
    """Build the real service with a live judge, using stand-ins for every model call."""
    monkeypatch.setattr(retrieval, "embed_texts", fake_embed)
    monkeypatch.setattr(
        generation,
        "chat",
        lambda *args, **kwargs: ChatResult("Memory guides control (Source 1).", 100, 20, 0.1),
    )
    build("titans", config, store)
    engine = RagEngine(config, store=store, anchors=["titans"])

    def make(live_judge: LiveJudge) -> TestClient:
        return TestClient(create_app(engine, live_judge))

    return make


def test_answers_are_judged_after_the_response(make_client):
    with make_client(LiveJudge(sample_rate=1.0, judge_class=FakeJudge)) as client:
        response = client.post("/predict", json=QUESTION)
        summary = client.get("/stats").json()["live_judge"]

    assert response.status_code == 200
    question, answer, context = FakeJudge.calls[0]
    assert answer == "Memory guides control (Source 1)."
    # The judge checks the answer against the same passages the model was given.
    assert "[Source 1:" in context
    assert summary["judged"] == 1
    assert summary["average_faithfulness"] == 0.9
    assert summary["average_relevancy"] == 0.8


def test_nothing_is_judged_when_sampling_is_off(make_client):
    with make_client(LiveJudge(sample_rate=0.0, judge_class=FakeJudge)) as client:
        client.post("/predict", json=QUESTION)
        summary = client.get("/stats").json()["live_judge"]

    assert FakeJudge.calls == []
    assert summary["enabled"] is False
    assert summary["average_faithfulness"] is None


def test_the_hourly_cap_limits_spend(make_client):
    with make_client(LiveJudge(sample_rate=1.0, max_per_hour=2, judge_class=FakeJudge)) as client:
        for _ in range(4):
            client.post("/predict", json=QUESTION)

    assert len(FakeJudge.calls) == 2


def test_a_failed_judge_call_never_breaks_the_answer(make_client):
    with make_client(LiveJudge(sample_rate=1.0, judge_class=FailingJudge)) as client:
        response = client.post("/predict", json=QUESTION)
        summary = client.get("/stats").json()["live_judge"]

    assert response.status_code == 200
    assert summary["failed"] == 1
    assert summary["last"]["error"] == "ValueError"


def test_answers_without_claims_are_counted_apart():
    live_judge = LiveJudge(sample_rate=1.0, judge_class=FakeJudge)
    live_judge.judge_class = type(
        "NoClaims", (FakeJudge,), {"score": lambda self, q, a, c: JudgeResult(None, 0.4)}
    )

    live_judge.score("A question?", "I cannot answer from these passages.", [])

    summary = live_judge.summary()
    assert summary["no_claims"] == 1
    assert summary["average_faithfulness"] is None
    assert summary["average_relevancy"] == 0.4


def test_sampling_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("JUDGE_SAMPLE_RATE", "0.5")
    monkeypatch.setenv("JUDGE_MAX_PER_HOUR", "12")

    live_judge = LiveJudge.from_env()

    assert (live_judge.sample_rate, live_judge.max_per_hour) == (0.5, 12)
