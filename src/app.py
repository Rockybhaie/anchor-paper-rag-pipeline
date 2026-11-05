"""FastAPI service. Start it with: uvicorn src.app:app"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from openai import OpenAIError
from pinecone.exceptions import PineconeException

from src.config import load_config
from src.rag import RagEngine
from src.schemas import ExplainResponse, PredictResponse, QuestionRequest


def create_app(engine: RagEngine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fails at startup with a clear message if the saved index has not been built.
        config_name = os.environ.get("ANCHOR_RAG_CONFIG", "serving")
        app.state.engine = engine or RagEngine(load_config(config_name))
        yield

    app = FastAPI(
        title="Anchor Paper RAG",
        description="Answers questions from four anchor papers and the papers that cite them.",
        version="1.0.0",
        lifespan=lifespan,
    )

    def run(request: Request, method: str, body: QuestionRequest) -> dict:
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
    def predict(request: Request, body: QuestionRequest) -> dict:
        """Answer a question and name the source papers."""
        return run(request, "predict", body)

    @app.post("/explain", response_model=ExplainResponse)
    def explain(request: Request, body: QuestionRequest) -> dict:
        """Return the passages used for a question and the paper each came from."""
        return run(request, "explain", body)

    return app


app = create_app()
