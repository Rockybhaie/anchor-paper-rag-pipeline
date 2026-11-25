"""Score a sample of live answers with the LLM judge, after the response has been sent.

This is the start of the project's LLM-as-a-Judge focus: the same judge that chose the served
setup keeps checking it on real questions. Scores are kept in memory and shown by /stats.
"""

from __future__ import annotations

import os
import random
import threading
import time
from collections import deque

from src.generation import format_context
from src.judge import Judge

# The Stage 1 winner: the judge that best told real passages from unrelated ones.
JUDGE_SETTINGS = {"provider": "openai", "model": "gpt-4o-mini", "prompt": "rubric"}
HOUR_S = 3600


class LiveJudge:
    def __init__(
        self,
        sample_rate: float = 0.0,
        max_per_hour: int = 30,
        settings: dict | None = None,
        judge_class=Judge,
    ):
        self.sample_rate = sample_rate
        self.max_per_hour = max_per_hour
        self.settings = settings or JUDGE_SETTINGS
        self.judge_class = judge_class
        self.lock = threading.Lock()
        self.started: deque[float] = deque()
        self.judged = 0
        self.failed = 0
        self.refusals = 0
        self.faithfulness: list[float] = []
        self.relevancy: list[float] = []
        self.cost = 0.0
        self.last: dict | None = None

    @classmethod
    def from_env(cls) -> LiveJudge:
        """Off unless JUDGE_SAMPLE_RATE is set, so local runs and tests spend nothing."""
        return cls(
            sample_rate=float(os.environ.get("JUDGE_SAMPLE_RATE", "0")),
            max_per_hour=int(os.environ.get("JUDGE_MAX_PER_HOUR", "30")),
        )

    def should_judge(self) -> bool:
        """Sample answers, and cap how many are judged each hour to limit spend on a public URL."""
        if random.random() >= self.sample_rate:
            return False
        now = time.monotonic()
        with self.lock:
            while self.started and now - self.started[0] > HOUR_S:
                self.started.popleft()
            if len(self.started) >= self.max_per_hour:
                return False
            self.started.append(now)
            return True

    def score(self, question: str, answer: str, passages: list[dict]) -> None:
        judge = self.judge_class(self.settings)
        try:
            result = judge.score(question, answer, format_context(passages))
        except Exception as error:  # noqa: BLE001 - a failed check must never affect the service
            with self.lock:
                self.failed += 1
                self.last = {"question": question, "error": type(error).__name__}
            return
        with self.lock:
            self.judged += 1
            self.cost += result.cost
            self.relevancy.append(result.relevancy)
            if result.faithfulness is None:
                # No factual claims, which usually means the system declined to answer.
                self.refusals += 1
            else:
                self.faithfulness.append(result.faithfulness)
            self.last = {
                "question": question,
                "faithfulness": result.faithfulness,
                "relevancy": result.relevancy,
            }

    def summary(self) -> dict:
        def average(values: list[float]) -> float | None:
            return round(sum(values) / len(values), 4) if values else None

        with self.lock:
            return {
                "enabled": self.sample_rate > 0,
                "judge": self.settings["model"],
                "sample_rate": self.sample_rate,
                "max_per_hour": self.max_per_hour,
                "judged": self.judged,
                "failed": self.failed,
                "no_claims": self.refusals,
                "average_faithfulness": average(self.faithfulness),
                "average_relevancy": average(self.relevancy),
                "estimated_cost_usd": round(self.cost, 5),
                "last": self.last,
            }
