"""Run the naive-vs-batched comparison and the priority demo, save outputs/bench.json.

Usage:
    python scripts/run_bench.py
    N=64 MAX_NEW_TOKENS=32 MAX_BATCH_SIZE=8 python scripts/run_bench.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Make `import src...` work when run as a plain script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.bench import demo_priority, run_batched, run_naive  # noqa: E402
from src.engine import BatchingEngine  # noqa: E402
from src.model import CausalLMWrapper  # noqa: E402


async def main() -> None:
    load_dotenv()

    n = int(os.environ.get("N", "64"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "32"))
    max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", "8"))
    max_wait_ms = float(os.environ.get("MAX_WAIT_MS", "20"))

    wrapper = CausalLMWrapper()
    prompts = [f"Prompt number {i}: tell me a short fact." for i in range(n)]

    print(f"Model: {wrapper.model_id} on device: {wrapper.device}")
    print(f"N={n} max_new_tokens={max_new_tokens} "
          f"max_batch_size={max_batch_size} max_wait_ms={max_wait_ms}")

    print("Running naive sequential baseline...")
    naive = run_naive(wrapper.generate, prompts, max_new_tokens)

    print("Running batched engine...")
    engine = BatchingEngine(
        generate_fn=wrapper.generate,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
    )
    await engine.start()
    try:
        batched = await run_batched(engine, prompts, max_new_tokens)
        priority = await demo_priority(engine, n_low=max_batch_size * 3, max_new_tokens=max_new_tokens)
    finally:
        await engine.stop()

    speedup = (
        batched["throughput_req_per_s"] / naive["throughput_req_per_s"]
        if naive["throughput_req_per_s"] > 0
        else 0.0
    )

    result = {
        "config": {
            "model_id": wrapper.model_id,
            "device": wrapper.device,
            "n_requests": n,
            "max_new_tokens": max_new_tokens,
            "max_batch_size": max_batch_size,
            "max_wait_ms": max_wait_ms,
        },
        "naive": naive,
        "batched": batched,
        "throughput_speedup_x": speedup,
        "priority_demo": priority,
    }

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "bench.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
