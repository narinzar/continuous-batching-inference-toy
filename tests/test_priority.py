"""Test that a late high-priority request jumps ahead of earlier low ones."""

from __future__ import annotations

import asyncio

import pytest

from src.engine import BatchingEngine


class SlowModel:
    def generate(self, prompts, max_new_tokens):
        import time

        time.sleep(0.02)
        return [f"{p}!" for p in prompts]


@pytest.mark.asyncio
async def test_high_priority_dequeued_before_earlier_low_priority():
    """A priority-0 request enqueued LAST should complete before the low ones.

    We use max_batch_size=1 so the engine services one request per generate call;
    that isolates scheduling order. Many low-priority (priority=5) requests are
    enqueued first, then a single high-priority (priority=0) request last. It must
    not finish last.
    """
    model = SlowModel()
    engine = BatchingEngine(generate_fn=model.generate, max_batch_size=1, max_wait_ms=2.0)

    completion_order: list[str] = []
    lock = asyncio.Lock()

    async def submit(tag: str, priority: int) -> None:
        await engine.submit(prompt=tag, priority=priority)
        async with lock:
            completion_order.append(tag)

    n_low = 6
    futures = [asyncio.ensure_future(submit(f"low{i}", 5)) for i in range(n_low)]
    # Ensure the low-priority requests are all enqueued before the urgent one.
    await asyncio.sleep(0.005)
    futures.append(asyncio.ensure_future(submit("HIGH", 0)))

    await engine.start()
    try:
        await asyncio.gather(*futures)
    finally:
        await engine.stop()

    # HIGH arrived last (index n_low) but must finish well before last place.
    high_index = completion_order.index("HIGH")
    assert high_index < n_low, (
        f"expected HIGH to jump ahead, but it finished at index {high_index}: "
        f"{completion_order}"
    )
    # Every request still completed.
    assert len(completion_order) == n_low + 1


@pytest.mark.asyncio
async def test_priority_tiebreak_is_fifo():
    """Equal-priority requests are served in arrival order (arrival_seq tie-break)."""
    model = SlowModel()
    engine = BatchingEngine(generate_fn=model.generate, max_batch_size=1, max_wait_ms=2.0)

    completion_order: list[str] = []

    async def submit(tag: str) -> None:
        await engine.submit(prompt=tag, priority=3)
        completion_order.append(tag)

    tags = [f"t{i}" for i in range(5)]
    futures = []
    for t in tags:
        futures.append(asyncio.ensure_future(submit(t)))
        # Tiny stagger to fix arrival order deterministically.
        await asyncio.sleep(0.001)

    await engine.start()
    try:
        await asyncio.gather(*futures)
    finally:
        await engine.stop()

    assert completion_order == tags
