"""Tests for the batching engine using a mocked model (no torch/HF needed).

The mock records every batch it is asked to generate, letting us assert that the
scheduler groups multiple queued requests into a single generate() call and that
every submitted future eventually resolves.
"""

from __future__ import annotations

import asyncio

import pytest

from src.engine import BatchingEngine


class RecordingModel:
    """Stand-in for CausalLMWrapper.generate that logs batch sizes."""

    def __init__(self, delay: float = 0.01) -> None:
        self.delay = delay
        self.batch_sizes: list[int] = []
        self.seen_prompts: list[list[str]] = []

    def generate(self, prompts, max_new_tokens):
        # Runs in a worker thread via asyncio.to_thread; a small sleep mimics
        # model compute and gives concurrent submits time to accumulate.
        import time

        self.batch_sizes.append(len(prompts))
        self.seen_prompts.append(list(prompts))
        time.sleep(self.delay)
        return [f"{p}<gen>" for p in prompts]


@pytest.mark.asyncio
async def test_multiple_requests_grouped_into_one_batch():
    model = RecordingModel(delay=0.05)
    engine = BatchingEngine(
        generate_fn=model.generate,
        max_batch_size=8,
        max_wait_ms=50.0,
    )
    await engine.start()
    try:
        # Submit 5 requests near-simultaneously; they should land in one batch
        # because the wait window (50ms) is wide relative to enqueue time.
        results = await asyncio.gather(
            *(engine.submit(prompt=f"p{i}", priority=0) for i in range(5))
        )
    finally:
        await engine.stop()

    assert len(results) == 5
    assert all(r.endswith("<gen>") for r in results)
    # The key assertion: at least one dispatched batch had more than one prompt,
    # i.e. batching actually happened rather than one call per request.
    assert max(model.batch_sizes) > 1
    # And total prompts processed equals total submitted.
    assert sum(model.batch_sizes) == 5


@pytest.mark.asyncio
async def test_every_future_resolves():
    model = RecordingModel(delay=0.0)
    engine = BatchingEngine(generate_fn=model.generate, max_batch_size=4, max_wait_ms=10.0)
    await engine.start()
    try:
        results = await asyncio.gather(
            *(engine.submit(prompt=f"q{i}", priority=i % 3) for i in range(20))
        )
    finally:
        await engine.stop()

    assert len(results) == 20
    assert all(isinstance(r, str) and r for r in results)


@pytest.mark.asyncio
async def test_priority_ordering_within_a_batch_dispatch():
    """When a full queue is drained, higher priority (lower number) goes first.

    Use max_batch_size=1 so each generate() call carries exactly one prompt; the
    recorded prompt order then equals the dispatch order, which must follow
    (priority, arrival_seq).
    """
    model = RecordingModel(delay=0.02)
    engine = BatchingEngine(generate_fn=model.generate, max_batch_size=1, max_wait_ms=5.0)

    # Pre-load the heap before starting the scheduler so all requests are queued
    # together and ordering is deterministic.
    futures = []
    # Enqueue in mixed priority order; arrival_seq breaks ties.
    specs = [("a", 5), ("b", 1), ("c", 1), ("d", 0)]
    for prompt, prio in specs:
        futures.append(asyncio.ensure_future(engine.submit(prompt=prompt, priority=prio)))
    # Let the submits register in the heap.
    await asyncio.sleep(0.01)

    await engine.start()
    try:
        await asyncio.gather(*futures)
    finally:
        await engine.stop()

    dispatched = [p[0] for p in model.seen_prompts]
    # Expected order: d (prio 0), then b, c (prio 1, FIFO by arrival), then a (prio 5).
    assert dispatched == ["d", "b", "c", "a"]


@pytest.mark.asyncio
async def test_exception_propagates_to_callers():
    def boom(prompts, max_new_tokens):
        raise RuntimeError("model failed")

    engine = BatchingEngine(generate_fn=boom, max_batch_size=2, max_wait_ms=5.0)
    await engine.start()
    try:
        with pytest.raises(RuntimeError, match="model failed"):
            await engine.submit(prompt="x", priority=0)
    finally:
        await engine.stop()
