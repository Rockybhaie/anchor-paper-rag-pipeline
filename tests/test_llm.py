from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError, InternalServerError

from src import llm
from src.llm import chat, estimate_cost, reasoning_effort, with_retries


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)


REQUEST = httpx.Request("POST", "https://example.com")
MESSAGES = [{"role": "user", "content": "Hello"}]


class FakeClient:
    """Stands in for the OpenAI client, raising the given errors before replying."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        message = SimpleNamespace(content=" Hi there. ")
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def test_chat_returns_the_text_and_token_counts():
    result = chat("openai", "gpt-4o-mini", MESSAGES, client=FakeClient())
    assert (result.text, result.input_tokens, result.output_tokens) == ("Hi there.", 10, 5)
    assert result.temperature_applied


def test_chat_retries_when_a_model_is_briefly_overloaded():
    error = InternalServerError(
        "overloaded", response=httpx.Response(503, request=REQUEST), body=None
    )
    client = FakeClient(error)
    assert chat("gemini", "gemini-3.8-flash", MESSAGES, client=client).text == "Hi there."
    assert len(client.calls) == 2


def test_chat_drops_temperature_when_a_model_rejects_it():
    error = BadRequestError(
        "temperature is not supported", response=httpx.Response(400, request=REQUEST), body=None
    )
    client = FakeClient(error)
    result = chat("openai", "gpt-5-nano", MESSAGES, temperature=0.7, client=client)
    assert not result.temperature_applied
    assert "temperature" in client.calls[0]
    assert "temperature" not in client.calls[1]


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        ("gpt-5-nano", "minimal"),
        ("openai/gpt-oss-20b", "low"),
        ("gemini-3.8-flash", "low"),
        ("gpt-4o-mini", None),
    ],
)
def test_reasoning_models_get_a_low_effort_setting(model, effort):
    assert reasoning_effort(model) == effort


def test_reasoning_models_get_extra_room_so_they_can_still_answer():
    client = FakeClient()
    chat("groq", "openai/gpt-oss-20b", MESSAGES, max_tokens=100, client=client)
    assert client.calls[0]["max_completion_tokens"] == 400
    assert client.calls[0]["reasoning_effort"] == "low"

    plain = FakeClient()
    chat("openai", "gpt-4o-mini", MESSAGES, max_tokens=100, client=plain)
    assert plain.calls[0]["max_tokens"] == 100
    assert "reasoning_effort" not in plain.calls[0]


def test_with_retries_recovers_from_a_short_network_failure():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("dns failure")
        return "done"

    assert with_retries(flaky, (ConnectionError,)) == "done"
    assert len(calls) == 3


def test_with_retries_gives_up_after_the_limit():
    calls = []

    def always_fails():
        calls.append(1)
        raise ConnectionError("still down")

    with pytest.raises(ConnectionError):
        with_retries(always_fails, (ConnectionError,))
    assert len(calls) == llm.MAX_RETRIES


def test_with_retries_does_not_hide_other_errors():
    calls = []

    def bug():
        calls.append(1)
        raise ValueError("a real bug")

    with pytest.raises(ValueError):
        with_retries(bug, (ConnectionError,))
    assert len(calls) == 1


def test_estimate_cost_uses_prices_and_counts_unknown_models_as_free():
    assert estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert estimate_cost("text-embedding-3-small", 1_000_000) == pytest.approx(0.02)
    assert estimate_cost("gemini-3.8-flash", 1_000_000, 1_000_000) == 0.0
