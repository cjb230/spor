"""Generate exact train/validation sample counts from native MCB teacher games.

The native worker accepts one JSON request per line. Its response is SPGEN001,
two little-endian u64s (next split-local game counter, row count), then 200-byte
rows: feature bits[135], raw card counts[2], mask bits[61], action u16 LE.
Only the optional native trace mode constructs JSON views, for parity tests.

Each split owns seed + 2*counter + split_index. Native requests finish whole
games and discard surplus samples from the final game. A committed shard's
cursor therefore always resumes on a fresh deck. Samples are not guaranteed
to have unique board positions. Memory use is bounded by one shard plus one
1024-row native response, regardless of the total requested sample count.
"""
from contextlib import AbstractContextManager
import json
import math
import os
from pathlib import Path
import re
import selectors
import struct
import subprocess
import tempfile
import time

import numpy as np

from .stream_data import (Progress, append_shard, create_dataset, digest,
                          exclusive_lock, finish_dataset, open_manifest)


REQUEST_ROWS = 1024
RESPONSE_TIMEOUT = 3600
EXIT_TIMEOUT = 5
ROW_SIZE = 200
PROTOCOL_VERSION = 1


def _integer(value, name, minimum, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _strategy(value):
    return isinstance(value, str) and (
        value in ("random", "greedy", "builder", "tactician", "planner")
        or re.fullmatch(r"(?:mcg|mcb|mcp)(?::[0-9]+)?", value) is not None
    )


class _Worker(AbstractContextManager):
    """A persistent native process; bounded reads handle partial binary responses."""
    def __init__(self, path, threads):
        self.path, self.threads = path, threads

    def __enter__(self):
        self.errors = tempfile.TemporaryFile()
        self.selector = selectors.DefaultSelector()
        try:
            self.process = subprocess.Popen([str(self.path)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self.errors,
                env=os.environ | {"RAYON_NUM_THREADS": str(self.threads)})
            self.selector.register(self.process.stdout, selectors.EVENT_READ)
        except BaseException:
            self.errors.close()
            self.selector.close()
            raise
        return self

    def _error(self, message):
        self.errors.seek(0, os.SEEK_END)
        size = self.errors.tell()
        self.errors.seek(max(0, size - 4096))
        detail = self.errors.read().decode("utf-8", errors="replace").strip()
        return RuntimeError(f"{message}: {detail}" if detail else message)

    def _read(self, size, deadline):
        result = bytearray(size)
        position = 0
        while position < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise self._error("native generator response timed out")
            chunk = os.read(self.process.stdout.fileno(), min(size - position, 65536))
            if not chunk:
                raise self._error(f"native generator EOF (exit {self.process.poll()})")
            result[position:position + len(chunk)] = chunk
            position += len(chunk)
        return result

    def request(self, command):
        try:
            self.process.stdin.write(json.dumps(command).encode() + b"\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise self._error("native generator closed its input") from error
        deadline = time.monotonic() + RESPONSE_TIMEOUT
        header = self._read(24, deadline)
        if header[:8] != b"SPGEN001":
            raise self._error("invalid native generator binary header")
        cursor, count = struct.unpack("<QQ", header[8:])
        if count != command["count"] or not command["start"] < cursor <= 2**63 - 1:
            raise self._error("invalid native generator row count or cursor")
        data = self._read(count * ROW_SIZE, deadline)
        return cursor, np.frombuffer(data, dtype=np.uint8).reshape(count, ROW_SIZE)

    def _terminate(self):
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=EXIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=EXIT_TIMEOUT)

    def __exit__(self, exception_type, exception, traceback):
        try:
            if exception_type is not None:
                self._terminate()
            else:
                self.process.stdin.close()
                try:
                    status = self.process.wait(timeout=EXIT_TIMEOUT)
                except subprocess.TimeoutExpired as error:
                    self._terminate()
                    raise self._error("native generator shutdown timed out") from error
                if status != 0:
                    raise self._error(f"native generator exited with status {status}")
        finally:
            self.process.stdin.close()
            self.process.stdout.close()
            self.errors.close()
            self.selector.close()


def _arrays(rows):
    return dict(packed=rows[:, :135], counts=rows[:, 135:137],
                mask=rows[:, 137:198],
                actions=rows[:, 198:200].copy().view("<u2").reshape(-1))


def _resume(root, config, progress):
    manifest = open_manifest(root)
    if manifest.get("config") != config:
        raise ValueError("resume configuration or native binary changed")
    cursor = manifest.get("cursor")
    if not isinstance(cursor, dict) or set(cursor) - {"train", "validation"}:
        raise ValueError("invalid generation cursor")
    for split, key in (("train", "examples"), ("validation", "validation_examples")):
        count = manifest["counts"][split]
        _integer(count, f"{split} count", 0, config[key])
        counter = cursor.get(split, 0)
        _integer(counter, f"{split} cursor", 0)
        if bool(count) != bool(counter):
            raise ValueError("generation counts disagree with cursor")
    total = sum(manifest["counts"].values())
    checked = 0
    progress.update(0, total, detail="checking committed shard hashes")
    for entry in manifest["shards"]:
        if digest(root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"shard integrity hash mismatch: {entry['path']}")
        checked += entry["count"]
        progress.update(checked, total, detail=f"checked {entry['path']}")
    if manifest["complete"] and manifest["counts"] != {
            "train": config["examples"], "validation": config["validation_examples"]}:
        raise ValueError("completed dataset has incorrect sample counts")
    return manifest


def generate(output, examples, teacher="mcb:16", validation_examples=100000, seed=1001,
             opponents=("builder", "tactician", "planner", "mcb:16"), shard_size=65536,
             threads=None, worker=Path("rust/target/release/examples/neural_generate"),
             progress_interval=1, resume=False):
    """Write compressed shards with exact sample counts and deterministic decks.

    Resume requires identical generation options and worker bytes. Orphaned
    shards are ignored and atomically replaced by the shared dataset writer.
    Threads are recorded in the config although native results do not depend on
    thread count. Progress interval and worker path may change on resume.
    """
    _integer(examples, "examples", 1)
    _integer(validation_examples, "validation_examples", 0)
    _integer(seed, "seed", 0, 2**63 - 2)
    _integer(shard_size, "shard_size", 1)
    if threads is None:
        threads = os.cpu_count() or 1
    _integer(threads, "threads", 1)
    if (not isinstance(progress_interval, (int, float)) or isinstance(progress_interval, bool)
            or not math.isfinite(progress_interval) or progress_interval <= 0):
        raise ValueError("progress_interval must be positive and finite")
    if not _strategy(teacher) or not (teacher == "mcb" or teacher.startswith("mcb:")):
        raise ValueError("teacher must be an MCB strategy")
    if (not isinstance(opponents, (list, tuple)) or not opponents
            or not all(_strategy(opponent) for opponent in opponents)):
        raise ValueError("opponents must be a nonempty sequence of native strategies")
    worker = Path(worker).resolve()
    config = dict(examples=examples, validation_examples=validation_examples, teacher=teacher,
                  seed=seed, opponents=list(opponents), shard_size=shard_size, threads=threads,
                  worker_sha256=digest(worker), protocol_version=PROTOCOL_VERSION)
    root = Path(output)
    with exclusive_lock(root):
        if resume:
            with Progress("Verifying", interval=progress_interval) as verification:
                manifest = _resume(root, config, verification)
        else:
            manifest = create_dataset(root, config)
        if manifest["complete"]:
            return manifest
        cursors = {"train": 0, "validation": 0} | manifest.get("cursor", {})
        total = examples + validation_examples
        with Progress(f"Generating ({threads} threads)", interval=progress_interval) as progress, _Worker(worker, threads) as process:
            for split, target in (("train", examples), ("validation", validation_examples)):
                while manifest["counts"][split] < target:
                    n = min(shard_size, target - manifest["counts"][split])
                    rows = np.empty((n, ROW_SIZE), dtype=np.uint8)
                    filled = 0
                    while filled < n:
                        completed = sum(manifest["counts"].values()) + filled
                        progress.update(completed, total, detail=f"{split}: native active; {filled:,}/{n:,} in shard")
                        command = dict(op="generate", seed=seed, split=split, start=cursors[split],
                                       count=min(REQUEST_ROWS, n - filled), teacher=teacher,
                                       opponents=list(opponents))
                        counter, batch = process.request(command)
                        rows[filled:filled + len(batch)] = batch
                        filled += len(batch)
                        cursors[split] = counter
                        progress.update(sum(manifest["counts"].values()) + filled, total,
                                        detail=f"{split}: {filled:,}/{n:,} in shard")
                    progress.update(sum(manifest["counts"].values()) + n, total,
                                    detail=f"{split}: compressing shard")
                    manifest = append_shard(root, split, _arrays(rows), cursor=dict(cursors))
                    del rows, batch
            progress.update(total, total, detail="all shards committed")
        # A failed worker shutdown leaves a resumable, incomplete manifest.
        return finish_dataset(root)
