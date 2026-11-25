"""FastAPI service. Start it with: uvicorn src.app:app"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from openai import OpenAIError
from pinecone.exceptions import PineconeException

from src.config import load_config
from src.live_judge import LiveJudge
from src.rag import RagEngine
from src.schemas import ExplainResponse, PredictResponse, QuestionRequest

VERSION = "2.0.0"


class RequestStats:
    """Request count and latency since the service started. Restarting resets them."""

    def __init__(self):
        self.count = 0
        self.total_ms = 0.0
        self.by_path: dict[str, list[float]] = {}

    def record(self, path: str, elapsed_ms: float) -> None:
        self.count += 1
        self.total_ms += elapsed_ms
        totals = self.by_path.setdefault(path, [0, 0.0])
        totals[0] += 1
        totals[1] += elapsed_ms

    def summary(self) -> dict:
        def average(total: float, count: int) -> float:
            return round(total / count, 1) if count else 0.0

        return {
            "request_count": self.count,
            "average_latency_ms": average(self.total_ms, self.count),
            # Health checks are frequent and fast, so each path is also shown on its own.
            "by_path": {
                path: {"count": int(count), "average_latency_ms": average(total, count)}
                for path, (count, total) in sorted(self.by_path.items())
            },
        }


def create_app(engine: RagEngine | None = None, live_judge: LiveJudge | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fails at startup with a clear message if the saved index has not been built.
        config_name = os.environ.get("ANCHOR_RAG_CONFIG", "serving")
        app.state.engine = engine or RagEngine(load_config(config_name))
        app.state.live_judge = live_judge or LiveJudge.from_env()
        app.state.started_at = time.time()
        yield

    app = FastAPI(
        title="Anchor Paper RAG",
        description="Answers questions from four anchor papers and the papers that cite them.",
        version=VERSION,
        lifespan=lifespan,
    )
    app.state.stats = RequestStats()

    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        app.state.stats.record(request.url.path, (time.perf_counter() - start) * 1000)
        return response

    def run(request: Request, method: str, body: QuestionRequest):
        rag = request.app.state.engine
        anchor_id = body.anchor_id.value
        if anchor_id not in rag.indexes:
            raise HTTPException(status_code=503, detail=f"No index is loaded for {anchor_id}.")
        try:
            return getattr(rag, method)(anchor_id, body.question)
        except (OpenAIError, PineconeException, RuntimeError) as error:
            raise HTTPException(
                status_code=503, detail=f"A required service failed: {type(error).__name__}"
            ) from error

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: Request, body: QuestionRequest, background: BackgroundTasks) -> dict:
        """Answer a question and name the source papers."""
        response, passages = run(request, "answer", body)
        live_judge = request.app.state.live_judge
        if live_judge.should_judge():
            # Runs after the response is sent, so judging never slows the answer down.
            background.add_task(live_judge.score, body.question, response["answer"], passages)
        return response

    @app.post("/explain", response_model=ExplainResponse)
    def explain(request: Request, body: QuestionRequest) -> dict:
        """Return the passages used for a question and the paper each came from."""
        return run(request, "explain", body)

    @app.get("/health")
    def health(request: Request) -> dict:
        """Report that the service is up and which anchors it serves. Calls no outside service."""
        return {
            "status": "ok",
            "version": VERSION,
            "anchors": sorted(request.app.state.engine.indexes),
        }

    @app.get("/stats")
    def stats(request: Request) -> dict:
        """Request count and average latency since start, and the live judge's scores."""
        uptime_s = time.time() - request.app.state.started_at
        return request.app.state.stats.summary() | {
            "uptime_s": int(uptime_s),
            "live_judge": request.app.state.live_judge.summary(),
        }

    return app


app = create_app()
