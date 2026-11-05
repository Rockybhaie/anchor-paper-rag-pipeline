"""Where vectors live. PineconeStore is used for real runs and InMemoryStore for tests."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pinecone import Pinecone, ServerlessSpec
from pinecone.errors.exceptions import (
    PineconeConnectionError,
    PineconeTimeoutError,
    RateLimitError,
    ServiceError,
)

from src.llm import with_retries

UPSERT_BATCH = 100
UPLOAD_WORKERS = 4
INDEX_PREFIX = "anchor-rag"
NETWORK_ERRORS = (PineconeConnectionError, PineconeTimeoutError, RateLimitError, ServiceError)


class PineconeStore:
    """One Pinecone index per vector size, and one namespace per anchor and setup."""

    def __init__(self, cloud: str = "aws", region: str = "us-east-1"):
        key = os.environ.get("PINECONE_API_KEY")
        if not key:
            raise RuntimeError("PINECONE_API_KEY is not set. Add it to your .env file.")
        self.client = Pinecone(api_key=key)
        self.cloud = cloud
        self.region = region
        self.indexes = {}

    def _create_if_missing(self, name: str, dimension: int) -> None:
        if name in self.client.list_indexes().names():
            return
        self.client.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=self.cloud, region=self.region),
        )
        while not self.client.describe_index(name).status["ready"]:
            time.sleep(2)

    def _index(self, dimension: int):
        """Connect once per vector size, so searches do not repeat this lookup."""
        if dimension not in self.indexes:
            name = f"{INDEX_PREFIX}-{dimension}"
            with_retries(lambda: self._create_if_missing(name, dimension), NETWORK_ERRORS)
            self.indexes[dimension] = self.client.Index(name)
        return self.indexes[dimension]

    def count(self, namespace: str, dimension: int) -> int:
        index = self._index(dimension)
        stats = with_retries(index.describe_index_stats, NETWORK_ERRORS)
        namespaces = stats.get("namespaces") or {}
        return namespaces[namespace]["vector_count"] if namespace in namespaces else 0

    def clear(self, namespace: str, dimension: int) -> None:
        index = self._index(dimension)
        with_retries(lambda: index.delete(delete_all=True, namespace=namespace), NETWORK_ERRORS)

    def upsert(self, namespace: str, ids: list[str], vectors: np.ndarray) -> None:
        index = self._index(vectors.shape[1])

        def upload(start: int) -> None:
            end = start + UPSERT_BATCH
            batch = zip(ids[start:end], vectors[start:end], strict=True)
            rows = [(chunk_id, vector.tolist()) for chunk_id, vector in batch]
            with_retries(lambda: index.upsert(vectors=rows, namespace=namespace), NETWORK_ERRORS)

        # A batch of vectors is several megabytes, so sending a few at once saves a lot of time.
        with ThreadPoolExecutor(UPLOAD_WORKERS) as pool:
            list(pool.map(upload, range(0, len(ids), UPSERT_BATCH)))

    def wait_for_count(self, namespace: str, dimension: int, expected: int, timeout_s: int = 120):
        """Pinecone updates its counts shortly after an upload, so wait until they match."""
        deadline = time.monotonic() + timeout_s
        while self.count(namespace, dimension) != expected:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Pinecone did not reach {expected} vectors in {namespace}.")
            time.sleep(3)

    def query(self, namespace: str, vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        index = self._index(len(vector))
        response = with_retries(
            lambda: index.query(
                vector=vector.tolist(), top_k=top_k, namespace=namespace, include_metadata=False
            ),
            NETWORK_ERRORS,
        )
        return [(match["id"], float(match["score"])) for match in response["matches"]]


class InMemoryStore:
    """Same methods as PineconeStore with no network, so tests need no key."""

    def __init__(self):
        self.namespaces: dict[str, tuple[list[str], np.ndarray]] = {}

    def count(self, namespace: str, dimension: int) -> int:
        return len(self.namespaces[namespace][0]) if namespace in self.namespaces else 0

    def clear(self, namespace: str, dimension: int) -> None:
        self.namespaces.pop(namespace, None)

    def upsert(self, namespace: str, ids: list[str], vectors: np.ndarray) -> None:
        self.namespaces[namespace] = (list(ids), np.array(vectors))

    def wait_for_count(self, namespace: str, dimension: int, expected: int, timeout_s: int = 120):
        if self.count(namespace, dimension) != expected:
            raise TimeoutError(namespace)

    def query(self, namespace: str, vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        ids, vectors = self.namespaces[namespace]
        scores = vectors @ vector
        order = np.argsort(scores)[::-1][:top_k]
        return [(ids[i], float(scores[i])) for i in order]
