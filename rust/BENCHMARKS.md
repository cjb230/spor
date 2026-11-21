# M4 Pro measurements

These are saved historical results for greedy/random workloads. They do not
measure the deterministic or Monte Carlo strategies or tournament recording.
Rerun the comparison below to measure the current checkout on your hardware.

Measured on 2026-11-16 on the target MacBook Pro: arm64, 14 logical CPUs.
Rust 1.100.0; compiler native CPU detection reports `apple-m4`.
Release profile: optimization level 3, fat LTO, one codegen unit,
`-C target-cpu=native`. The executable is a native ARM64 Mach-O binary.

Each result is the median of three sequential runs, measured externally with a
monotonic wall clock, including process startup. Deck and bot seed are both 1;
starting strategies alternate seats. No tests or compiler were running during
these measurements. Python REST used Uvicorn with access logging disabled.

| Workload | Games/run | Workers | Median wall time | Games/second |
|---|---:|---:|---:|---:|
| Python bots → Python REST server | 100 | 1 client loop | 4.6794 s | 21.4 |
| Same Python bots → Rust REST server | 100 | 1 client loop | 4.0167 s | 24.9 |
| Python engine and bots directly | 100 | 1 | 0.6290 s | 159.0 |
| Rust engine and bots directly | 100,000 | 1 | 3.6284 s | 27,560 |
| Rust engine and bots directly | 100,000 | 4 | 0.9547 s | 104,747 |
| Rust engine and bots directly | 100,000 | 8 | 0.4955 s | 201,802 |
| Rust engine and bots directly | 100,000 | 14 | 0.3445 s | 290,284 |

The 100-game workloads all returned greedy 95 wins, random 5 wins, no draws.
Every 100,000-game native run returned greedy 96,196 wins, random 3,804 wins, no
draws. Explicit seeds reproduce outcomes across worker counts.

The REST comparison shows roughly 1.16× end-to-end throughput with the unchanged
Python clients. It measures client code, bots, HTTP, JSON and server together,
not isolated server throughput. Native simulation eliminates HTTP/JSON and uses
compact player views without event histories; the direct Python baseline retains
Pydantic views and histories. The direct rows use different game counts and
include startup costs, so they should not be read as a controlled rules-only
language speedup. Hardware load, frequency, temperature and workload change results.

The Python [MLP player](../docs/NEURAL.md) uses the compatible REST server. The
native throughput figure applies to built-in greedy/random self-play; it is not
a promised training rate for Python agents over REST.

Reproduce using `python3 -m benchmarks.compare` with both servers running as
described in [README.md](README.md#reproducing-measurements). Full timing samples
are in [m4-pro-results.json](../benchmarks/m4-pro-results.json).
