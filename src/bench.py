"""Benchmark: naive one-at-a-time generation vs the async batching engine.

Two comparisons are produced:

1. Throughput / latency: fire N requests and measure
   (a) naive - call generate() one prompt at a time, sequentially, and
   (b) batched - submit all N into the BatchingEngine concurrently.
   We report req/s throughput and mean / p95 per-request latency for each.

2. Priority demonstration: enqueue one high-priority request AFTER many
   low-priority ones and show it finishes earlier than its arrival order would
   suggest under FIFO.

No numbers are hardcoded; everything here is measured at run time.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import Callable, Dict, List

from tqdm import tqdm


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def run_naive(
    generate_fn: Callable[[List[str], int], List[str]],
    prompts: List[str],
    max_new_tokens: int,
) -> Dict[str, float]:
    """Sequential baseline: one prompt per generate() call."""
    latencies: List[float] = []
    t0 = time.perf_counter()
    for p in tqdm(prompts, desc="naive", unit="req"):
        s = time.perf_counter()
        generate_fn([p], max_new_tokens)
        latencies.append(time.perf_counter() - s)
    wall = time.perf_counter() - t0
    return _summarize(latencies, wall, len(prompts))


async def run_batched(
    engine,
    prompts: List[str],
    max_new_tokens: int,
) -> Dict[str, float]:
    """Submit every prompt concurrently and let the engine batch them."""
    latencies: List[float] = []

    async def one(prompt: str) -> None:
        s = time.perf_counter()
        await engine.submit(prompt=prompt, priority=0, max_new_tokens=max_new_tokens)
        latencies.append(time.perf_counter() - s)

    t0 = time.perf_counter()
    await asyncio.gather(*(one(p) for p in prompts))
    wall = time.perf_counter() - t0
    return _summarize(latencies, wall, len(prompts))


def _summarize(latencies: List[float], wall: float, n: int) -> Dict[str, float]:
    return {
        "requests": float(n),
        "wall_seconds": wall,
        "throughput_req_per_s": (n / wall) if wall > 0 else 0.0,
        "mean_latency_ms": statistics.mean(latencies) * 1000 if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 95) * 1000,
    }


async def demo_priority(engine, n_low: int, max_new_tokens: int) -> Dict[str, float]:
    """Show a late high-priority request jumping ahead of earlier low ones.

    We enqueue `n_low` low-priority requests (priority=5) and then, immediately
    after, one high-priority request (priority=0). Each records the order in
    which its result comes back. The high-priority request's completion index is
    expected to be far lower than its arrival index (which is last).
    """
    completion_order: List[str] = []
    lock = asyncio.Lock()

    async def submit(tag: str, priority: int) -> None:
        await engine.submit(prompt=f"tag {tag}: ", priority=priority, max_new_tokens=max_new_tokens)
        async with lock:
            completion_order.append(tag)

    tasks = [asyncio.create_task(submit(f"low{i}", 5)) for i in range(n_low)]
    # Yield so the low-priority ones are enqueued first, then add the urgent one.
    await asyncio.sleep(0)
    tasks.append(asyncio.create_task(submit("HIGH", 0)))
    await asyncio.gather(*tasks)

    high_finish_index = completion_order.index("HIGH")
    return {
        "n_low_priority": float(n_low),
        "high_priority_arrived_index": float(n_low),  # enqueued last
        "high_priority_finished_index": float(high_finish_index),
    }
