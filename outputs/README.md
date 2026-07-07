# outputs/

Benchmark artifacts land here. Running `python scripts/run_bench.py` writes
`bench.json`, which contains the measured naive-vs-batched throughput/latency
numbers, the computed speedup, and the priority-demo result.

Everything in this folder except this README is gitignored, so committed results
never contain machine-specific numbers.
