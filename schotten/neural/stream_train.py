"""Exact epoch targets with bounded decoding and resumable shard minibatches.

Only a compact shard and its row permutation are retained on the host. CPU
restarts reproduce the exact AdamW updates; MPS may have numerical variation.
Epochs are total targets, so a completed run can be extended without repeating
its earlier passes. Each shard's final short minibatch is included.
"""
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path
from time import monotonic
from uuid import uuid4

import numpy as np
import torch
from torch.nn import functional

from . import stream_data
from .model import load_checkpoint, masked_logits, save_checkpoint, select_device
from .train import _finite_number, _validate_options


def _publish_best(source, destination):
    """Atomically refresh the convenience copy from an immutable best version."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".best-", delete=False) as handle:
            temporary = Path(handle.name)
            with source.open("rb") as original:
                shutil.copyfileobj(original, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _order(entries, seed, epoch):
    order = list(range(len(entries)))
    random.Random(seed + epoch).shuffle(order)
    return order


def _rows(count, seed, epoch, shard):
    # Independent per-shard generators let resume seek without opening earlier
    # shards or constructing their permutations. Epoch numbering starts at one.
    return np.random.default_rng(np.random.SeedSequence(
        [(seed + epoch) % (1 << 64), shard]
    )).permutation(count)


def _check_progress(progress, epoch, entries, batch_size, seed):
    if not isinstance(progress, dict) or progress.get("version") != 1:
        raise ValueError("incompatible stream progress")
    for field in ("completed_epochs", "shard_position", "batch_offset", "updates",
                  "epoch_correct", "epoch_examples"):
        if type(progress.get(field)) is not int or progress[field] < 0:
            raise ValueError(f"invalid stream progress {field}")
    completed = progress["completed_epochs"]
    position, offset = progress["shard_position"], progress["batch_offset"]
    if type(epoch) is not int or epoch != completed or position >= len(entries):
        raise ValueError("invalid stream epoch or shard cursor")
    ordered = [entries[i] for i in _order(entries, seed, completed + 1)]
    if offset >= ordered[position]["count"] or offset % batch_size:
        raise ValueError("invalid stream minibatch cursor")
    seen = sum(entry["count"] for entry in ordered[:position]) + offset
    updates = completed * sum(math.ceil(e["count"] / batch_size) for e in entries)
    updates += sum(math.ceil(e["count"] / batch_size) for e in ordered[:position]) + offset // batch_size
    if (progress["epoch_examples"] != seen or progress["epoch_correct"] > seen
            or progress["updates"] != updates):
        raise ValueError("invalid stream progress totals")
    for field in ("epoch_loss_sum", "cumulative_seconds", "best_val_loss"):
        _finite_number(progress.get(field), f"stream progress {field}")
        if progress[field] < 0:
            raise ValueError(f"invalid stream progress {field}")
    if seen == 0 and (progress["epoch_loss_sum"] or progress["epoch_correct"]):
        raise ValueError("invalid empty epoch progress")


def train_stream(dataset, initial, output, epochs=3, resume=None, batch_size=512,
                 learning_rate=.001, seed=1, device="auto", progress_interval=1,
                 checkpoint_interval=60):
    """Train to a total epoch target; atomically save last and best in output.

    Resume verifies the completed dataset, frozen initial checkpoint, device,
    optimizer configuration, cursor, and best checkpoint hash before writing.
    The last checkpoint is saved before the first update, periodically between
    minibatches, and after each full validation. Interruptions resume from the
    last durable minibatch boundary. No augmentation or early stopping is used.
    """
    _validate_options(epochs, batch_size, learning_rate, .2, seed, (256, 256))
    for name, value in (("progress_interval", progress_interval),
                        ("checkpoint_interval", checkpoint_interval)):
        _finite_number(value, name)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    initial, output = Path(initial), Path(output)
    reserved = [output / name for name in ("last.pt", "best.pt", "metrics.jsonl", ".writer.lock")]
    if initial.resolve() in {p.resolve() for p in reserved} or initial.resolve() == output.resolve():
        raise ValueError("initial and output paths must be distinct")
    if resume is not None and Path(resume).resolve() == initial.resolve():
        raise ValueError("resume must be a streaming checkpoint, not initial")
    if isinstance(dataset, (str, Path)):
        dataset = stream_data.Dataset(dataset)
    if not dataset.manifest.get("complete"):
        raise ValueError("dataset generation must complete before training")
    entries = {split: list(dataset.entries(split)) for split in ("train", "validation")}
    sizes = {split: dataset.count(split) for split in entries}
    for split in entries:
        if (not entries[split] or sizes[split] <= 0
                or any(type(e["count"]) is not int or e["count"] <= 0 for e in entries[split])
                or sum(e["count"] for e in entries[split]) != sizes[split]):
            raise ValueError(f"invalid or empty {split} split")
    selected_device = select_device(device)
    identity = {
        "dataset_fingerprint": dataset.fingerprint,
        "initial_sha256": stream_data.digest(initial),
        "training_config": {"batch_size": batch_size, "learning_rate": float(learning_rate),
                            "seed": seed, "device": str(selected_device), "hidden": [256, 256],
                            "optimizer": "AdamW", "weight_decay": .01},
    }
    with stream_data.exclusive_lock(output):
        return _train_locked(dataset, initial, output, epochs, resume, batch_size,
                             learning_rate, seed, selected_device, progress_interval,
                             checkpoint_interval, entries, sizes, identity)


def _train_locked(dataset, initial, output, epochs, resume, batch_size, learning_rate,
                  seed, device, progress_interval, checkpoint_interval, entries, sizes, identity):
    last, best, metrics_path = (output / name for name in ("last.pt", "best.pt", "metrics.jsonl"))
    if resume is None and any(path.exists() for path in (last, best, metrics_path)):
        raise ValueError("training output already exists; use resume or a new output directory")
    model, payload = load_checkpoint(initial if resume is None else resume, device)
    if model.hidden != (256, 256):
        raise ValueError("initial/resume model must have hidden widths 256, 256")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    if resume is None:
        progress = dict(version=1, completed_epochs=0, shard_position=0, batch_offset=0,
                        updates=0, epoch_loss_sum=0., epoch_correct=0, epoch_examples=0,
                        cumulative_seconds=0., best_val_loss=None)
        metrics, best_hash, best_file = {}, None, None
    else:
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("missing streaming resume metadata")
        for key, expected in identity.items():
            if metadata.get(key) != expected:
                raise ValueError(f"resume {key} is incompatible")
        progress = metadata.get("stream_progress")
        _check_progress(progress, payload.get("epoch"), entries["train"], batch_size, seed)
        if (progress["completed_epochs"] > epochs
                or (progress["completed_epochs"] == epochs and progress["epoch_examples"])):
            raise ValueError("epochs target precedes completed or partial epoch work")
        if not payload.get("optimizer_state"):
            raise ValueError("resume checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer_state"])
        if any(g["lr"] != learning_rate or g["weight_decay"] != .01 for g in optimizer.param_groups):
            raise ValueError("resume optimizer settings are incompatible")
        best_hash = metadata.get("best_checkpoint_sha256")
        best_file = metadata.get("best_checkpoint_file")
        if (not isinstance(best_file, str) or Path(best_file).name != best_file
                or not best_file.startswith("best-") or not best_file.endswith(".pt")):
            raise ValueError("resume best checkpoint filename is incompatible")
        best_version = output / best_file
        if not best_version.exists() or not best_hash or stream_data.digest(best_version) != best_hash:
            raise ValueError("resume best checkpoint is missing or incompatible")
        metrics = metadata.get("metrics")
        if (not isinstance(metrics, dict) or type(metrics.get("updates")) is not int
                or not 0 <= metrics["updates"] <= progress["updates"]):
            raise ValueError("resume metrics are incompatible")
        # A kill after publishing newer best weights but before committing last
        # must not invalidate the previous last checkpoint's best identity.
        if not best.exists() or stream_data.digest(best) != best_hash:
            _publish_best(best_version, best)

    prior_seconds, started = progress["cumulative_seconds"], monotonic()
    last_saved = started

    def save(path):
        progress["cumulative_seconds"] = prior_seconds + monotonic() - started
        model.eval()
        save_checkpoint(path, model, optimizer=optimizer, epoch=progress["completed_epochs"],
                        metadata={**identity, "untrained": progress["updates"] == 0,
                                  "stream_progress": dict(progress), "metrics": dict(metrics),
                                  "best_checkpoint_sha256": best_hash,
                                  "best_checkpoint_file": best_file})

    def record(event):
        row = {**metrics, "event": event, "best_val_loss": progress["best_val_loss"],
               "cumulative_seconds": prior_seconds + monotonic() - started}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
        return row

    def validate(event, train_loss=None, train_accuracy=None):
        nonlocal metrics, best_hash, best_file
        total_loss, correct, seen = 0., 0, 0
        model.eval()
        with stream_data.Progress("Validation", interval=progress_interval) as reporter, torch.no_grad():
            reporter.update(0, sizes["validation"], f"epoch {progress['completed_epochs']}")
            for entry in entries["validation"]:
                arrays = dataset.load(entry)
                for offset in range(0, entry["count"], batch_size):
                    indices = np.arange(offset, min(offset + batch_size, entry["count"]))
                    features, masks, actions = (
                        torch.from_numpy(a).to(device) for a in stream_data.unpack_batch(arrays, indices)
                    )
                    logits = masked_logits(model(features), masks)
                    loss = functional.cross_entropy(logits, actions).item()
                    if not math.isfinite(loss):
                        raise ValueError("nonfinite validation loss")
                    total_loss += loss * len(indices)
                    correct += (logits.argmax(-1) == actions).sum().item()
                    seen += len(indices)
                    reporter.update(seen, sizes["validation"], f"epoch {progress['completed_epochs']}")
                del arrays
        metrics = dict(epoch=progress["completed_epochs"], updates=progress["updates"],
                       train_loss=train_loss, train_accuracy=train_accuracy,
                       val_loss=total_loss / seen, val_accuracy=correct / seen,
                       train_examples=sizes["train"], val_examples=sizes["validation"])
        if progress["best_val_loss"] is None or metrics["val_loss"] < progress["best_val_loss"]:
            progress["best_val_loss"] = metrics["val_loss"]
            # Retain prior versions: last may still reference one if interrupted
            # between this publication and its next atomic checkpoint commit.
            best_file = f"best-{progress['updates']:012d}-{uuid4().hex}.pt"
            best_version = output / best_file
            save(best_version)
            best_hash = stream_data.digest(best_version)
            _publish_best(best_version, best)
        record(event)

    if resume is None:
        validate("initial")
        save(last)
        last_saved = monotonic()

    # Start a separate heartbeat for each phase so validation and training never
    # write competing carriage-return lines. Counts advance across epochs.
    remaining = (epochs - progress["completed_epochs"]) * sizes["train"] - progress["epoch_examples"]
    processed = 0
    while progress["completed_epochs"] < epochs:
        epoch = progress["completed_epochs"] + 1
        order = _order(entries["train"], seed, epoch)
        with stream_data.Progress(f"Training ({device})", interval=progress_interval) as reporter:
            reporter.update(processed, remaining, f"epoch {epoch}/{epochs}")
            while progress["shard_position"] < len(order):
                shard = order[progress["shard_position"]]
                entry = entries["train"][shard]
                arrays = dataset.load(entry)
                rows = _rows(entry["count"], seed, epoch, shard)
                while progress["batch_offset"] < entry["count"]:
                    offset = progress["batch_offset"]
                    indices = rows[offset:offset + batch_size]
                    features, masks, actions = (
                        torch.from_numpy(a).to(device) for a in stream_data.unpack_batch(arrays, indices)
                    )
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    logits = masked_logits(model(features), masks)
                    loss = functional.cross_entropy(logits, actions)
                    loss_value = loss.item()
                    if not math.isfinite(loss_value):
                        raise ValueError("nonfinite training loss")
                    correct = (logits.argmax(-1) == actions).sum().item()
                    loss.backward()
                    optimizer.step()
                    if device.type == "mps":
                        torch.mps.synchronize()
                    progress["updates"] += 1
                    progress["batch_offset"] += len(indices)
                    progress["epoch_examples"] += len(indices)
                    progress["epoch_loss_sum"] += loss_value * len(indices)
                    progress["epoch_correct"] += correct
                    processed += len(indices)
                    reporter.update(processed, remaining,
                                    f"epoch {epoch}/{epochs}, shard {progress['shard_position'] + 1}/{len(order)}")
                    if progress["batch_offset"] == entry["count"]:
                        progress["shard_position"] += 1
                        progress["batch_offset"] = 0
                        break
                    if monotonic() - last_saved >= checkpoint_interval:
                        save(last)
                        last_saved = monotonic()
                del arrays, rows
                if progress["shard_position"] == len(order):
                    break
                if monotonic() - last_saved >= checkpoint_interval:
                    save(last)
                    last_saved = monotonic()
            reporter.update(processed, remaining, f"epoch {epoch}/{epochs}, validating")
        train_loss = progress["epoch_loss_sum"] / sizes["train"]
        train_accuracy = progress["epoch_correct"] / sizes["train"]
        progress.update(completed_epochs=epoch, shard_position=0, batch_offset=0,
                        epoch_examples=0, epoch_loss_sum=0., epoch_correct=0)
        validate("epoch", train_loss, train_accuracy)
        save(last)
        last_saved = monotonic()
    return record("final")
