"""Measure direct simulation separately from Python clients over real REST."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parent.parent


def measure(name, command, games, repeats):
    print(f"Measuring {name}...", file=sys.stderr, flush=True)
    samples = []
    tallies = []
    for _ in range(repeats):
        started = perf_counter()
        result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        samples.append(perf_counter() - started)
        tallies.append([line for line in result.stdout.splitlines() if " wins (" in line or line.startswith("Draws:")])
    if not tallies[0] or any(tally != tallies[0] for tally in tallies):
        raise RuntimeError(f"{name} did not produce consistent seeded tallies")
    median = statistics.median(samples)
    return {"workload": name, "games": games, "wall_seconds": samples,
            "median_seconds": median, "games_per_second": games / median, "tallies": tallies[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-url", default="http://127.0.0.1:18080")
    parser.add_argument("--rust-url", default="http://127.0.0.1:18081")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--native-games", type=int, default=100000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8, 14])
    args = parser.parse_args()
    if min(args.games, args.native_games, args.repeats, *args.threads) < 1:
        parser.error("games, repeats and thread counts must be positive")
    results = []
    seeds = ["--seed", "1", "--bot-seed", "1"]
    for name, url in [("python-rest", args.python_url), ("rust-rest", args.rust_url)]:
        command = [sys.executable, "-m", "schotten.bot", "series", "--url", url,
                   "--games", str(args.games), "--quiet", *seeds]
        results.append(measure(name, command, args.games, args.repeats))
    command = [sys.executable, "-m", "benchmarks.python_series", "--games", str(args.games), *seeds]
    results.append(measure("python-direct", command, args.games, args.repeats))
    if any(result["tallies"] != results[0]["tallies"] for result in results[1:]):
        raise RuntimeError("Python and Rust REST/direct workloads produced different seeded tallies")
    for threads in args.threads:
        command = [str(ROOT / "rust/target/release/schotten-native"), "series",
                   "--games", str(args.native_games), "--threads", str(threads), "--quiet", *seeds]
        results.append(measure(f"rust-native-{threads}-threads", command, args.native_games, args.repeats))
    print(json.dumps({"machine": platform.machine(), "logical_cpus": os.cpu_count(),
                      "repeats": args.repeats, "seeds": {"deck": 1, "bots": 1},
                      "timing": "external wall clock including process startup",
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()
