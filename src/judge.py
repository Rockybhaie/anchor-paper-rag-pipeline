"""Score answers with an LLM judge.

Faithfulness is the share of an answer's claims that the retrieved passages support.
Relevancy is how close questions generated from the answer are to the question that was asked.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

from src.config import PROMPTS
from src.embeddings import embed_texts
from src.generation import read_prompt
from src.llm import chat, estimate_cost

# Relevancy always uses this model, so changing the judge does not change the yardstick.
RELEVANCY_EMBEDDING = {"provider": "openai", "model": "text-embedding-3-small"}
MAX_CLAIMS = 25
VERDICT_SCORES = {"SUPPORTED": 1.0, "PARTIAL": 0.5, "NOT_SUPPORTED": 0.0}


@dataclass
class JudgeResult:
    faithfulness: float | None
    relevancy: float
    claims: list[dict] = field(default_factory=list)
    generated_questions: list[str] = field(default_factory=list)
    cost: float = 0.0


def parse_json_list(text: str) -> list:
    """Read a JSON list from a model reply, tolerating code fences and text around it."""
    cleaned = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON list in judge reply: {text[:120]}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        # Answers with LaTeX such as \alpha produce replies that are not valid JSON. Dropping them
        # would leave the most mathematical answers out of the scores, so repair them instead.
        return json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", match.group(0)))


def score_verdicts(verdicts: list[dict], claims: list[str]) -> tuple[float, list[dict]]:
    """Match verdicts to claims by position. A missing verdict counts as not supported."""
    by_id = {int(item["id"]): str(item["verdict"]).strip().upper() for item in verdicts}
    results = []
    for position, claim in enumerate(claims, start=1):
        verdict = by_id.get(position, "NOT_SUPPORTED")
        verdict = verdict if verdict in VERDICT_SCORES else "NOT_SUPPORTED"
        results.append({"claim": claim, "verdict": verdict})
    score = sum(VERDICT_SCORES[item["verdict"]] for item in results) / len(results)
    return round(score, 4), results


class Judge:
    def __init__(self, settings: dict):
        self.provider = settings["provider"]
        self.model = settings["model"]
        self.prompt = settings["prompt"]
        self.cost = 0.0

    def _ask(self, prompt_file: str, replacements: dict[str, str], max_tokens: int) -> str:
        _, template = read_prompt(PROMPTS / "judge" / f"{prompt_file}.txt")
        for key, value in replacements.items():
            template = template.replace("{" + key + "}", value)
        result = chat(
            self.provider,
            self.model,
            [{"role": "user", "content": template}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        self.cost += estimate_cost(self.model, result.input_tokens, result.output_tokens)
        return result.text

    def extract_claims(self, answer: str) -> list[str]:
        reply = self._ask("extract_claims", {"answer": answer}, 700)
        claims = [str(claim).strip() for claim in parse_json_list(reply)]
        return [claim for claim in claims if len(claim) > 15][:MAX_CLAIMS]

    def verify_claims(self, claims: list[str], context: str) -> tuple[float, list[dict]]:
        """Check all claims against the full context. The earlier project cut context at 2,500."""
        numbered = "\n".join(f"{i}. {claim}" for i, claim in enumerate(claims, start=1))
        reply = self._ask(
            f"verify_{self.prompt}", {"claims": numbered, "context": context}, 40 * len(claims) + 50
        )
        return score_verdicts(parse_json_list(reply), claims)

    def relevancy(self, question: str, answer: str) -> tuple[float, list[str]]:
        reply = self._ask("generate_questions", {"answer": answer}, 200)
        generated = [str(item).strip() for item in parse_json_list(reply) if str(item).strip()][:3]
        if not generated:
            return 0.0, []
        vectors, cost = embed_texts([question, *generated], RELEVANCY_EMBEDDING)
        self.cost += cost
        return round(float(np.mean(vectors[1:] @ vectors[0])), 4), generated

    def score(self, question: str, answer: str, context: str) -> JudgeResult:
        """Score one answer. Faithfulness is None when the answer makes no factual claims."""
        start_cost = self.cost
        claims = self.extract_claims(answer)
        faithfulness, verdicts = (None, []) if not claims else self.verify_claims(claims, context)
        relevancy, generated = self.relevancy(question, answer)
        return JudgeResult(faithfulness, relevancy, verdicts, generated, self.cost - start_cost)
