"""Run one command with its own durable log and explicit timeout."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1150)
    parser.add_argument("--tail", type=int, default=3000)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.tail < 0:
        parser.error("timeout must be positive and finite; tail must be nonnegative")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.log.open("x", encoding="utf-8") as handle:
        header = {"command": command, "cwd": os.getcwd(),
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "timeout_seconds": args.timeout}
        handle.write(json.dumps(header) + "\n\n")
        handle.flush()
        print(f"Logging command to {args.log}", flush=True)
        try:
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = process.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                status = 124 if isinstance(error, subprocess.TimeoutExpired) else 130
        except OSError as error:
            handle.write(f"Command could not start: {error}\n")
            status = 127
        handle.write("\n" + json.dumps({"exit_status": status,
                     "wall_seconds": time.monotonic() - started,
                     "finished_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    print(f"{args.log}: exit {status}, {time.monotonic() - started:.2f}s", flush=True)
    if args.tail:
        print(args.log.read_text()[-args.tail:], flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
