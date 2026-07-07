"""Asyncio continuous-batching engine with a request-priority queue.

Idea: instead of running one generate() call per request, a single scheduler
coroutine drains incoming requests, groups up to `max_batch_size` of them (or
whatever has arrived once a `max_wait_ms` window elapses), and runs ONE batched
model.generate call. Each caller waits on its own asyncio.Future, so from the
caller's side the API is a normal `await engine.submit(...)`.

Priority queue: requests are ordered by a min-heap keyed on
`(priority, arrival_seq)`.

  - `priority` is an integer where LOWER numbers are served first
    (priority 0 is more urgent than priority 5). This matches the Unix "niceness"
    convention and lets callers pass 0 for "urgent".
  - `arrival_seq` is a monotonically increasing counter assigned at submit time.
    It is the tie-break: among requests of equal priority, the one that arrived
    first wins (FIFO within a priority level). It also makes the heap key total,
    so we never compare the payloads themselves.

The model call is synchronous (PyTorch), so it is offloaded to a thread via
asyncio.to_thread. That keeps the event loop free to accept new requests while a
batch is running, which is what makes continuous batching continuous.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional


@dataclass(order=True)
class _Request:
    """One queued generation request.

    Only `priority` and `arrival_seq` participate in ordering (see field
    metadata below); everything else is excluded from comparison so two requests
    with the same key never try to compare prompts or futures.
    """

    priority: int
    arrival_seq: int
    prompt: str = field(compare=False)
    max_new_tokens: int = field(compare=False)
    future: "asyncio.Future[str]" = field(compare=False)


# A generate function takes (prompts, max_new_tokens) and returns one string per
# prompt. The real one is CausalLMWrapper.generate; tests pass a mock.
GenerateFn = Callable[[List[str], int], List[str]]


class BatchingEngine:
    def __init__(
        self,
        generate_fn: GenerateFn,
        max_batch_size: int = 8,
        max_wait_ms: float = 20.0,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self._generate_fn = generate_fn
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        # Min-heap of _Request ordered by (priority, arrival_seq).
        self._heap: List[_Request] = []
        self._counter = itertools.count()
        # Signalled whenever a request is pushed, so the scheduler can wake up.
        self._not_empty = asyncio.Event()
        self._lock = asyncio.Lock()

        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False

        # Simple observability: batch sizes actually dispatched, in order.
        self.batch_sizes: List[int] = []

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        self._running = False
        # Wake the scheduler so it can observe _running == False and exit.
        self._not_empty.set()
        if self._scheduler_task is not None:
            await self._scheduler_task
            self._scheduler_task = None

    async def submit(
        self, prompt: str, priority: int = 0, max_new_tokens: int = 32
    ) -> str:
        """Enqueue a request and await its generated continuation."""
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[str]" = loop.create_future()
        req = _Request(
            priority=priority,
            arrival_seq=next(self._counter),
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            future=future,
        )
        async with self._lock:
            heapq.heappush(self._heap, req)
            self._not_empty.set()
        return await future

    async def _collect_batch(self) -> List[_Request]:
        """Wait for the first request, then gather more up to the batch window.

        Returns the highest-priority `max_batch_size` requests currently queued,
        after allowing `max_wait_ms` for stragglers to accumulate behind the
        first arrival.
        """
        # Block until at least one request exists (or we are shutting down).
        while True:
            async with self._lock:
                if self._heap:
                    break
            if not self._running:
                return []
            self._not_empty.clear()
            # Wait to be signalled by submit() or stop().
            await self._not_empty.wait()

        # A request is present. Give a short window for more to arrive so we can
        # fill the batch, but stop early once the batch is full.
        deadline = time.monotonic() + self.max_wait_ms / 1000.0
        while True:
            async with self._lock:
                queued = len(self._heap)
            if queued >= self.max_batch_size:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Sleep in small slices so newly arrived requests are counted.
            await asyncio.sleep(min(remaining, 0.001))

        async with self._lock:
            take = min(self.max_batch_size, len(self._heap))
            # heappop returns items in (priority, arrival_seq) order, so the most
            # urgent requests are pulled first even if they arrived last.
            batch = [heapq.heappop(self._heap) for _ in range(take)]
            if not self._heap:
                self._not_empty.clear()
        return batch

    async def _run_batch(self, batch: List[_Request]) -> None:
        prompts = [r.prompt for r in batch]
        # All requests in a batch share max_new_tokens for a single generate call;
        # use the max requested so no caller is short-changed.
        max_new = max(r.max_new_tokens for r in batch)
        self.batch_sizes.append(len(batch))

        try:
            # Offload the blocking model call so the loop keeps accepting work.
            results = await asyncio.to_thread(self._generate_fn, prompts, max_new)
        except Exception as exc:  # propagate failure to every waiting caller
            for r in batch:
                if not r.future.done():
                    r.future.set_exception(exc)
            return

        for r, text in zip(batch, results):
            if not r.future.done():
                r.future.set_result(text)

    async def _scheduler_loop(self) -> None:
        while self._running or self._heap:
            batch = await self._collect_batch()
            if not batch:
                if not self._running:
                    break
                continue
            await self._run_batch(batch)
