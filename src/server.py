"""FastAPI server exposing the batching engine over HTTP.

POST /generate {prompt, priority, max_new_tokens} enqueues the request into the
shared BatchingEngine and awaits its future. Because many clients can hit this
endpoint concurrently, their requests naturally pile up in the engine queue and
get grouped into batches.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.engine import BatchingEngine
from src.model import CausalLMWrapper

load_dotenv()


class GenerateRequest(BaseModel):
    prompt: str
    priority: int = Field(default=0, description="Lower number = served first.")
    max_new_tokens: int = Field(default=32, ge=1, le=512)


class GenerateResponse(BaseModel):
    completion: str


# Populated during the lifespan startup.
_engine: BatchingEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    wrapper = CausalLMWrapper()
    engine = BatchingEngine(
        generate_fn=wrapper.generate,
        max_batch_size=8,
        max_wait_ms=20.0,
    )
    await engine.start()
    _engine = engine
    try:
        yield
    finally:
        await engine.stop()
        _engine = None


app = FastAPI(title="Continuous Batching Inference Toy", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "engine_running": _engine is not None}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    assert _engine is not None, "engine not started"
    completion = await _engine.submit(
        prompt=req.prompt,
        priority=req.priority,
        max_new_tokens=req.max_new_tokens,
    )
    return GenerateResponse(completion=completion)
