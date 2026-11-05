"""Write an answer from retrieved passages using one of the prompts in prompts/generation/."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import PROMPTS
from src.llm import chat, estimate_cost

REFUSAL_PHRASES = (
    "does not contain enough",
    "do not contain enough",
    "does not provide enough",
    "do not provide enough",
    "not enough information",
    "insufficient information",
    "context is insufficient",
    "excerpts are insufficient",
    "cannot answer",
)


@dataclass
class GeneratedAnswer:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost: float
    temperature_applied: bool


def read_prompt(path) -> tuple[str | None, str]:
    """Read a prompt file. A '### system' section is optional and '### user' holds the rest."""
    content = path.read_text(encoding="utf-8")
    match = re.match(r"\s*### system\n(.*?)\n### user\n(.*)", content, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, content.strip()


def format_context(passages: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[Source {position}: {passage['title']} ({passage.get('year') or 'n.d.'})]\n"
        f"{passage['text']}"
        for position, passage in enumerate(passages, start=1)
    )


def build_messages(prompt_name: str, question: str, passages: list[dict]) -> list[dict[str, str]]:
    system, user = read_prompt(PROMPTS / "generation" / f"{prompt_name}.txt")
    user = user.replace("{context}", format_context(passages)).replace("{question}", question)
    messages = [{"role": "system", "content": system}] if system else []
    return [*messages, {"role": "user", "content": user}]


def extract_answer(text: str) -> str:
    """Prompts that reason first put the final answer after an [Answer] marker."""
    return text.split("[Answer]", 1)[1].strip() if "[Answer]" in text else text.strip()


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def generate(question: str, passages: list[dict], generation: dict) -> GeneratedAnswer:
    result = chat(
        generation["provider"],
        generation["model"],
        build_messages(generation["prompt"], question, passages),
        temperature=generation["temperature"],
        max_tokens=generation["max_tokens"],
    )
    return GeneratedAnswer(
        text=extract_answer(result.text),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_s=result.latency_s,
        cost=estimate_cost(generation["model"], result.input_tokens, result.output_tokens),
        temperature_applied=result.temperature_applied,
    )
