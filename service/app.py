"""FastAPI wrapper around the CiteMentor RAG pipeline.

This module is intentionally thin: it exposes the *existing* compiled LangGraph
pipeline (`src/core/graph.py::app_graph`) over HTTP without duplicating any
business logic. The same `app` object runs:

  - locally / in Docker via uvicorn  (`uvicorn service.app:app`)
  - on AWS Lambda via the Mangum adapter (`handler`, used by the container CMD)

The core pipeline uses *relative* paths (config/, catalog.json, storage/), so the
process must run with the repository root as its working directory (the Docker
image sets WORKDIR accordingly).
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel, Field

# Populate OPENAI_API_KEY from Secrets Manager (Lambda) before anything reads it.
# No-op locally where the key is already in the environment (.env).
from service.secrets import load_openai_key_from_secrets

load_openai_key_from_secrets()

# Importing app_graph runs graph.py's module-level setup (loads catalog/config/
# prompts, calls load_dotenv). The retriever and answer cache stay lazy — they
# are only constructed on the first /query, so /health never touches them.
from src.core.graph import app_graph

app = FastAPI(title="CiteMentor API", version="2.0.0")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question for the mentor.")


class Source(BaseModel):
    id: Optional[str] = None
    text: Optional[str] = None
    book_id: Optional[str] = None
    author: Optional[str] = None
    cross_score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    route: Optional[str] = None
    sources: List[Source] = []
    cache_hit: bool = False
    cache_similarity: Optional[float] = None
    timings: Dict[str, float] = {}


@app.get("/health")
def health() -> Dict[str, str]:
    """Cheap liveness probe. Does not initialize the retriever or any model,
    so it stays fast even on a cold Lambda."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Run one full RAG turn through the LangGraph pipeline and return the result.

    Not every graph path populates every state key (e.g. a cache hit or greeting
    skips routing/retrieval), so all fields are read defensively with defaults.
    """
    final_state: Dict[str, Any] = app_graph.invoke({"query": req.query, "timings": {}})
    return QueryResponse(
        answer=final_state.get("answer", ""),
        route=final_state.get("route"),
        sources=final_state.get("retrieved_chunks") or [],
        cache_hit=final_state.get("cache_hit", False),
        cache_similarity=final_state.get("cache_similarity"),
        timings=final_state.get("timings") or {},
    )


# AWS Lambda entry point. The container image's CMD is "service.app.handler".
handler = Mangum(app)
