import pytest
from fastapi.testclient import TestClient
from openai import OpenAIError

from src import app as app_module
from src import generation, retrieval
from src.app import create_app
from src.build_index import build
from src.llm import ChatResult
from src.rag import RagEngine
from tests.conftest import fake_embed

QUESTION = {"anchor_id": "titans", "question": "How is memory used for robot control?"}


@pytest.fixture
def client(corpus, store, config, monkeypatch):
    """The real service and retrieval path, with stand-ins for embeddings, reranking and the LLM."""
    monkeypatch.setattr(retrieval, "embed_texts", fake_embed)
    monkeypatch.setattr(retrieval, "rerank_scores", lambda q, texts: [0.5] * len(texts))
    monkeypatch.setattr(
        generation,
        "chat",
        lambda *args, **kwargs: ChatResult("Memory guides control (Source 1).", 100, 20, 0.1),
    )
    build("titans", config, store)
    engine = RagEngine(config, store=store, anchors=["titans"])
    with TestClient(create_app(engine)) as test_client:
        yield test_client


def test_predict_returns_answer_with_sources(client):
    response = client.post("/predict", json=QUESTION)

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Memory guides control (Source 1)."
    assert body["anchor_id"] == "titans"
    assert body["model"] == "gpt-4o-mini"
    assert body["setup"] == "recursive-400-80__text-embedding-3-small__v1"
    assert {source["title"] for source in body["sources"]} <= {"Anchor Paper", "Citing Paper"}
    assert body["sources"]


def test_predict_lists_each_source_paper_once(client):
    sources = client.post("/predict", json=QUESTION).json()["sources"]
    titles = [source["title"] for source in sources]
    assert len(titles) == len(set(titles))


def test_explain_returns_the_passages_and_their_papers(client):
    response = client.post("/explain", json=QUESTION)

    assert response.status_code == 200
    passages = response.json()["passages"]
    assert len(passages) == 5
    assert [passage["rank"] for passage in passages] == [1, 2, 3, 4, 5]
    assert all(passage["text"] and passage["title"] for passage in passages)


@pytest.mark.parametrize(
    "payload",
    [
        {"anchor_id": "not_an_anchor", "question": "A valid question here?"},
        {"anchor_id": "titans", "question": ""},
        {"anchor_id": "titans", "question": "   a  "},
        {"anchor_id": "titans", "question": "x" * 501},
        {"anchor_id": "titans"},
        {"question": "A valid question here?"},
    ],
)
@pytest.mark.parametrize("path", ["/predict", "/explain"])
def test_invalid_requests_are_rejected(client, path, payload):
    assert client.post(path, json=payload).status_code == 422


def test_anchor_without_a_loaded_index_returns_503(client):
    payload = {"anchor_id": "hrm", "question": "A valid question here?"}
    assert client.post("/predict", json=payload).status_code == 503


def test_a_failed_model_call_returns_503(client, monkeypatch):
    def fail(*args, **kwargs):
        raise OpenAIError("service down")

    monkeypatch.setattr(generation, "chat", fail)
    response = client.post("/predict", json=QUESTION)
    assert response.status_code == 503
    assert "OpenAIError" in response.json()["detail"]


def test_service_uses_the_config_named_in_the_environment(corpus, store, config, monkeypatch):
    loaded = []

    def fake_engine(chosen):
        loaded.append(chosen)
        return RagEngine(config, store=store, anchors=[])

    monkeypatch.setenv("ANCHOR_RAG_CONFIG", "free")
    monkeypatch.setattr(app_module, "RagEngine", fake_engine)
    with TestClient(create_app()):
        pass
    assert loaded[0]["generation"]["provider"] == "gemini"


def test_startup_explains_a_missing_index(corpus, store, config):
    with pytest.raises(FileNotFoundError, match="Run src.build_index"):
        RagEngine(config, store=store, anchors=["titans"])


def test_docs_are_generated(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert "/predict" in paths and "/explain" in paths
