# Continuous Batching Inference Toy

A minimal async inference server that dynamically batches incoming generation
requests for a small Hugging Face causal LM, plus a benchmark comparing it to
naive one-at-a-time inference. Includes a request-priority queue so urgent
requests jump the line.

## Problem

Serving an LLM one request at a time wastes the accelerator: each `generate`
call under-fills the GPU, so throughput is bounded by per-request overhead rather
than compute. Batching many requests into a single forward pass fixes that, but
requests do not all arrive at the same instant. Continuous batching solves the
timing problem: a scheduler collects whatever requests are waiting (up to a batch
size, or until a short time window expires) and runs them together, so the server
stays busy without forcing callers to arrive in lockstep. Doing this correctly
under async concurrency, while keeping the event loop responsive and returning
each result to the right caller, is the non-trivial part.

## Approach

- An asyncio `BatchingEngine` accepts requests via `submit()` and hands each
  caller an `asyncio.Future`. A single scheduler coroutine drains the queue.
- The scheduler collects up to `max_batch_size` requests, or waits at most
  `max_wait_ms` after the first arrival, then runs one batched `model.generate`.
- Requests are ordered by a min-heap keyed on `(priority, arrival_seq)`. Lower
  priority number is served first; `arrival_seq` is the FIFO tie-break within a
  priority level and keeps the heap key total so payloads are never compared.
- The blocking PyTorch call runs in a worker thread (`asyncio.to_thread`) so the
  loop keeps accepting new requests while a batch is in flight.
- The model wrapper pads a batch of prompts (left padding, EOS as pad token) and
  returns only the newly generated tokens per prompt.

## Setup

```bash
# 1. Create and activate a virtual env (either works)
uv venv --python 3.12 .venv        # or: python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# 2. Install torch from the CUDA 12.8 wheel index (RTX 5090 / sm_120)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3. Install the rest
pip install -r requirements.txt

# 4. Copy the env template (no secrets required)
cp .env.example .env
```

The default model is `sshleifer/tiny-gpt2` (tiny, fast, public). Swap in
`distilgpt2` for slightly more coherent output by setting `MODEL_ID=distilgpt2`
or passing `model_id` to `CausalLMWrapper`.

## How to run

Run the benchmark (naive vs batched + priority demo), which writes
`outputs/bench.json`:

```bash
python scripts/run_bench.py
# tunables: N=64 MAX_NEW_TOKENS=32 MAX_BATCH_SIZE=8 MAX_WAIT_MS=20 python scripts/run_bench.py
```

Launch the HTTP server:

```bash
python scripts/run_server.py
# or: HOST=0.0.0.0 PORT=8000 python scripts/run_server.py
```

Send a request:

```bash
curl -s http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "priority": 0, "max_new_tokens": 16}'
```

Run the tests:

```bash
pytest
```

## Results

Reproduce with:

```bash
python scripts/run_bench.py
```

Expected qualitative behavior:

- Batching raises throughput (req/s) and GPU utilization at the cost of a small
  added latency window (`max_wait_ms`) per request.
- The throughput speedup over the naive baseline grows with concurrency until
  the batch fills (roughly `max_batch_size`), then plateaus: once every batch is
  full you are compute-bound, not overhead-bound.
- The priority queue lets an urgent (low priority-number) request that arrives
  after many low-priority ones finish earlier than its arrival order would allow
  under FIFO. `bench.json` reports the high-priority request's arrival index vs
  its finish index; the finish index should be much smaller.

Measured run below: `sshleifer/tiny-gpt2` on a single RTX 5090 (CUDA), N=64
requests, max_new_tokens=32, max_batch_size=8, max_wait_ms=20. Because the model
is tiny, the naive baseline is already fast, yet continuous batching still gives
a large throughput win by filling the GPU with several prompts per forward pass.

| metric                     | naive  | batched |
| -------------------------- | ------ | ------- |
| throughput (req/s)         | 12.35  | 105.24  |
| mean latency (ms)          | 80.92  | 354.64  |
| p95 latency (ms)           | 98.50  | 607.79  |
| throughput speedup (x)     | -      | 8.52    |

Batching trades a small per-request latency increase (requests wait up to
`max_wait_ms` to be grouped, and share a forward pass) for roughly 8.5x higher
throughput. Mean/p95 latency rise because each request now spends time queued
and batched rather than being served the instant it arrives.

Priority demo (from `bench.json`): 24 low-priority requests are enqueued, then
one high-priority request arrives last (arrival index 24). It finishes at index
8 instead of last, jumping ahead of the earlier low-priority work.

## What I'd do next at larger scale

Add token-level continuous batching (admit and evict sequences mid-generation
using a KV cache, instead of batching only at request start) so long and short
generations no longer block each other. Replace the in-process engine with a
paged-KV-cache backend and a multi-GPU worker pool, and add admission control /
backpressure plus per-tenant fair scheduling on top of the priority queue.
