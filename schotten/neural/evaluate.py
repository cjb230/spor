"""Evaluate a frozen MLP on a scheduled game slice through a native JSON-lines worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time

import torch

from ..schemas import GameView
from .encoding import decode_action, encode_view, legal_mask
from .model import load_checkpoint, masked_logits

RESPONSE_TIMEOUT = 120.0
EXIT_TIMEOUT = 60.0


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _integer(value, name, minimum, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum or 'unbounded'}")


def _schedule(path):
    contents = Path(path).read_bytes()
    manifest = json.loads(contents)
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int:
        raise ValueError("schedule must be an object with schema_version 1")
    if manifest["schema_version"] != 1 or manifest.get("opponent") != "mcb:16":
        raise ValueError("schedule requires schema_version 1 and opponent mcb:16")
    games = manifest.get("games")
    if not isinstance(games, list) or not games:
        raise ValueError("schedule games must be a nonempty list")
    for index, game in enumerate(games):
        if not isinstance(game, dict) or set(game) != {"seed", "bot_seed", "mlp_seat"}:
            raise ValueError(f"schedule game {index} requires seed, bot_seed and mlp_seat")
        for field in ("seed", "bot_seed"):
            _integer(game[field], f"game {index} {field}", -(2**63), 2**63 - 1)
        _integer(game["mlp_seat"], f"game {index} mlp_seat", 0, 1)
    return games, hashlib.sha256(contents).hexdigest()


class _Worker:
    def __init__(self, path, threads):
        self.path = Path(path).resolve()
        self.threads = threads
        self.buffer = b""

    def __enter__(self):
        self.process = subprocess.Popen(
            [str(self.path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            env=os.environ | {"RAYON_NUM_THREADS": str(self.threads)},
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        return self

    def request(self, request):
        try:
            self.process.stdin.write(json.dumps(request).encode("utf-8") + b"\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError("native worker closed its input unexpectedly") from error
        deadline = time.monotonic() + RESPONSE_TIMEOUT
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise RuntimeError(f"native worker response timed out after {RESPONSE_TIMEOUT:g}s")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(f"native worker EOF before a response (exit {self.process.poll()})")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        try:
            response = json.loads(line)
        except (ValueError, UnicodeError) as error:
            raise RuntimeError("native worker returned invalid JSON") from error
        if not isinstance(response, dict):
            raise RuntimeError("native worker response must be a JSON object")
        if "error" in response:
            raise RuntimeError(f"native worker error: {response['error']}")
        return response

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
                    raise RuntimeError("native worker shutdown timed out") from error
                if status != 0:
                    raise RuntimeError(f"native worker exited with status {status}")
        finally:
            self.selector.close()
            self.process.stdin.close()
            self.process.stdout.close()


def _snapshots(response, specifications, previous=None):
    snapshots = response.get("games")
    if not isinstance(snapshots, list) or len(snapshots) != len(specifications):
        raise RuntimeError("native worker must return all game snapshots")
    views = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            raise RuntimeError("native worker snapshot must be an object")
        identity = snapshot.get("id")
        if type(identity) is not int or not 0 <= identity < len(specifications) or identity in views:
            raise RuntimeError("native worker returned invalid or duplicate game id")
        raw = snapshot.get("view")
        if not isinstance(raw, dict) or type(raw.get("version")) is not int or raw["version"] < 1:
            raise RuntimeError("native worker returned invalid game version")
        view = GameView.model_validate(raw)
        if type(raw.get("you")) is not int or view.you != specifications[identity]["mlp_seat"]:
            raise RuntimeError("native worker returned the wrong player seat")
        if view.phase == "your_turn":
            if view.status != "in_progress" or view.turn != view.you or view.winner is not None:
                raise RuntimeError("native worker returned inconsistent active game phase")
        elif view.phase == "game_over":
            if view.status not in ("finished", "forfeit") or view.turn is not None or view.win_reason is None:
                raise RuntimeError("native worker returned inconsistent terminal game phase")
        else:
            raise RuntimeError(f"native worker returned unexpected game phase {view.phase}")
        if previous is not None:
            old = previous[identity]
            if old.phase == "game_over":
                if view != old:
                    raise RuntimeError("native worker changed a finished game")
            elif view.version - old.version not in (1, 2):
                raise RuntimeError("native worker returned a stale or inconsistent game version")
        views[identity] = view
    return views


def _publish(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def evaluate(*, checkpoint, schedule, worker, output, start=0, count,
             batch_size=64, threads=8):
    """Run CPU argmax inference; inference time includes encoding, masking and decoding.

    Turns exclude the game's initial join: native versions advance once per turn.
    """
    started = time.monotonic()
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    specifications, schedule_hash = _schedule(schedule)
    _integer(start, "start", 0, len(specifications) - 1)
    _integer(count, "count", 1, len(specifications) - start)
    _integer(batch_size, "batch_size", 1)
    _integer(threads, "threads", 1)
    checkpoint_hash, worker_hash = _sha256(checkpoint), _sha256(worker)
    torch.set_num_threads(1)
    model, _ = load_checkpoint(checkpoint, "cpu")
    model.eval().requires_grad_(False)
    if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise ValueError("checkpoint parameters must be finite")
    results = []
    inference_seconds = 0.0
    decisions = 0
    with _Worker(worker, threads) as process, torch.inference_mode():
        for offset in range(start, start + count, batch_size):
            batch = specifications[offset:min(offset + batch_size, start + count)]
            views = _snapshots(process.request({"op": "new", "games": batch}), batch)
            while active := sorted(identity for identity, view in views.items() if view.phase == "your_turn"):
                inference_started = time.monotonic()
                features = torch.tensor([encode_view(views[identity]) for identity in active], dtype=torch.float32)
                mask = torch.tensor([legal_mask(views[identity]) for identity in active], dtype=torch.bool)
                logits = model(features)
                if not torch.isfinite(logits).all().item():
                    raise ValueError("model produced nonfinite action scores")
                choices = masked_logits(logits, mask).argmax(dim=-1).tolist()
                actions = []
                for identity, choice in zip(active, choices):
                    play = decode_action(choice)
                    actions.append({"id": identity, "version": views[identity].version,
                                    "play": play.model_dump(mode="json") if play is not None else None})
                inference_seconds += time.monotonic() - inference_started
                decisions += len(active)
                views = _snapshots(process.request({"op": "turn", "actions": actions}), batch, views)
            for identity, specification in enumerate(batch):
                view = views[identity]
                results.append({"schedule_index": offset + identity, **specification,
                                "winner": {"you": "mlp", "them": "mcb:16", None: None}[view.winner],
                                "reason": view.win_reason, "turns": view.version - 1})
            print(f"\r{len(results)} / {count} games played", end="", flush=True)
    print(flush=True)
    report = {"schema_version": 1, "checkpoint_sha256": checkpoint_hash,
              "worker_sha256": worker_hash, "schedule_sha256": schedule_hash,
              "start": start, "count": count, "elapsed_seconds": time.monotonic() - started,
              "inference_seconds": inference_seconds, "inference_decisions": decisions,
              "games": results, "wins": sum(game["winner"] == "mlp" for game in results),
              "losses": sum(game["winner"] == "mcb:16" for game in results),
              "draws": sum(game["winner"] is None for game in results)}
    _publish(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "schedule", "worker", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        evaluate(**vars(args))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"evaluation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
