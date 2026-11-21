"""Run one bounded, recorded Monte Carlo matchup experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--agents", nargs=2, required=True)
    parser.add_argument("--agent-seeds", nargs=2, type=int, required=True)
    parser.add_argument("--deck-seed", type=int, required=True)
    parser.add_argument("--games", type=int, default=100)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    binary = root / "rust/target/release/schotten-native"
    output = root / "benchmarks/mc-comparison-2026-11-19"
    output.mkdir(exist_ok=True)
    destination = output / f"{args.name}.json"
    if destination.exists():
        parser.error(f"result already exists: {destination}")
    command = [str(binary), "tournament", "--agents", *args.agents,
               "--games-per-pair", str(args.games), "--deck-seed", str(args.deck_seed),
               "--agent-seeds", *map(str, args.agent_seeds), "--record-games", "0", "--threads", "14"]
    started = perf_counter()
    progress_path = output / f"{args.name}.progress.log"
    with progress_path.open("x") as errors:
        process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=errors, text=True)
        while True:
            remaining = 1100 - (perf_counter() - started)
            if remaining <= 0:
                process.kill()
                stdout, _ = process.communicate()
                break
            try:
                stdout, _ = process.communicate(timeout=min(15, remaining))
                break
            except subprocess.TimeoutExpired:
                updates = progress_path.read_text().splitlines()
                if updates:
                    print(f"{args.name}: {updates[-1]}", flush=True)
    stderr = progress_path.read_text()
    elapsed = perf_counter() - started
    rows = []
    for line in stdout.splitlines():
        match = re.fullmatch(r"(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%", line)
        if match:
            name, strategy, wins, played, losses, draws, win_percent = match.groups()
            rows.append(dict(name=name, strategy=strategy, wins=int(wins), played=int(played),
                             losses=int(losses), draws=int(draws), win_percent=float(win_percent)))
    result = dict(command=command, deck_seed=args.deck_seed, agent_seeds=args.agent_seeds,
                  games=args.games, elapsed_seconds=elapsed, exit_code=process.returncode,
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  standings=rows, stdout=stdout, stderr=stderr)
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    if process.returncode or len(rows) != 2 or sum(row["wins"] for row in rows) + rows[0]["draws"] != args.games:
        raise RuntimeError(f"experiment failed; see {destination}")
    print(json.dumps({"name": args.name, "seconds": round(elapsed, 2), "standings": rows}), flush=True)


if __name__ == "__main__":
    main()
