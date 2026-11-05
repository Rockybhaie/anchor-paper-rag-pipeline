"""One chat and embedding interface for OpenAI, Groq and Gemini.

Groq and Gemini both offer OpenAI-compatible endpoints, so one client library covers all three.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

T = TypeVar("T")
# Temporary failures, including free-tier models that are briefly overloaded.
NETWORK_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

# These models spend output tokens thinking before they answer. Without a low effort setting and
# extra room, they can use the whole allowance and return nothing.
REASONING_EFFORT = {"gpt-5": "minimal", "openai/gpt-oss": "low", "gemini-": "low"}
REASONING_TOKEN_FACTOR = 4

PROVIDERS = {
    "openai": {"base_url": None, "key": "OPENAI_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key": "GROQ_API_KEY"},
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key": "GEMINI_API_KEY",
    },
}

# US dollars per 1M tokens (input, output). Free tiers count as zero.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5-nano": (0.05, 0.40),
    "text-embedding-3-small": (0.02, 0.0),
}

MAX_RETRIES = 8


def with_retries(call: Callable[[], T], retry_on: tuple[type[Exception], ...]) -> T:
    """Run a network call, waiting longer after each failure, about four minutes in total."""
    wait = 5.0
    for attempt in range(MAX_RETRIES):
        try:
            return call()
        except retry_on:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(wait)
            wait = min(wait * 2, 60.0)
    raise RuntimeError("Retries ended without a result.")


@dataclass
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    temperature_applied: bool = True


def get_client(provider: str) -> OpenAI:
    settings = PROVIDERS[provider]
    key = os.environ.get(settings["key"])
    if not key:
        raise RuntimeError(f"{settings['key']} is not set. Add it to your .env file.")
    return OpenAI(api_key=key, base_url=settings["base_url"])


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    input_price, output_price = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def reasoning_effort(model: str) -> str | None:
    for prefix, effort in REASONING_EFFORT.items():
        if model.startswith(prefix):
            return effort
    return None


def chat(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 512,
    client: OpenAI | None = None,
) -> ChatResult:
    """Call a chat model, retrying on rate limits and dropping temperature if it is rejected."""
    client = client or get_client(provider)
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    effort = reasoning_effort(model)
    if effort:
        kwargs["max_completion_tokens"] = max_tokens * REASONING_TOKEN_FACTOR
        kwargs["reasoning_effort"] = effort
    else:
        kwargs["max_tokens"] = max_tokens

    temperature_applied = True
    wait = 5.0
    for attempt in range(MAX_RETRIES):
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(**kwargs)
        except BadRequestError as error:
            if "temperature" in str(error).lower() and "temperature" in kwargs:
                kwargs.pop("temperature")
                temperature_applied = False
                continue
            raise
        except NETWORK_ERRORS:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(wait)
            wait = min(wait * 2, 60.0)
            continue
        usage = response.usage
        return ChatResult(
            text=(response.choices[0].message.content or "").strip(),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_s=time.perf_counter() - start,
            temperature_applied=temperature_applied,
        )
    raise RuntimeError(f"{model} did not return a response after {MAX_RETRIES} attempts.")
