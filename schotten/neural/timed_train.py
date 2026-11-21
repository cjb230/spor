"""Time-budgeted MCB:16 imitation, with resumable minibatches and best weights.

The budget starts after data preparation and optimizer restoration. It includes
validation and checkpoint work, and is checked between training batches. Final
validation and an atomic save may extend the budget. CPU resumes preserve the
exact update sequence; different devices can introduce numerical differences.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from time import monotonic

import torch
from torch.nn import functional

from .data import load_examples, split_examples
from .model import load_checkpoint, masked_logits, save_checkpoint, select_device
from .train import _dataset, _epoch, _finite_number, _tensors, _validate_options


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_progress(progress, completed_epochs, training_size, batch_size, seed):
    if not isinstance(progress, dict) or progress.get("version") != 1:
        raise ValueError("resume checkpoint has incompatible timed progress")
    integers = ("epoch_index", "completed_epochs", "next_batch_position", "updates", "epoch_correct")
    if any(type(progress.get(field)) is not int or progress[field] < 0 for field in integers):
        raise ValueError("resume checkpoint has invalid timed progress")
    position = progress["next_batch_position"]
    if (
        type(completed_epochs) is not int
        or completed_epochs != progress["completed_epochs"]
        or completed_epochs != progress["epoch_index"]
        or position >= training_size
        or position % batch_size
        or progress.get("shuffle_seed") != seed + completed_epochs + 1
        or progress["epoch_correct"] > position
        or progress["updates"] != completed_epochs * math.ceil(training_size / batch_size) + position // batch_size
    ):
        raise ValueError("resume checkpoint has invalid epoch progress or batch cursor")
    for field in ("epoch_loss_sum", "cumulative_seconds", "best_val_loss"):
        value = progress.get(field)
        _finite_number(value, f"resume progress {field}")
        if value < 0:
            raise ValueError(f"resume progress {field} must be nonnegative")
    if position == 0 and (progress["epoch_loss_sum"] != 0 or progress["epoch_correct"] != 0):
        raise ValueError("resume checkpoint has invalid epoch progress totals")


def train(
    recordings,
    initial,
    output,
    best,
    seconds=600,
    resume=None,
    device="mps",
    seed=1,
    batch_size=512,
    learning_rate=0.001,
    validation_fraction=0.2,
):
    """Continue the frozen 1082-256-256-487 policy for another wall-time budget.

    Resume requires the same initial file, examples, split, optimizer settings
    and best checkpoint. The latter must remain available, including when its
    weights are still the initial baseline. Training never stops early. The last
    checkpoint holds the next example offset within its deterministically
    shuffled epoch and the optimizer state. Elapsed checkpoint metadata is a
    wall-clock snapshot immediately before the atomic write.
    """
    _finite_number(seconds, "seconds")
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    _validate_options(1, batch_size, learning_rate, validation_fraction, seed, (256, 256))
    if validation_fraction == 0:
        raise ValueError("validation_fraction must be positive to select best weights")
    initial, output, best = (Path(path) for path in (initial, output, best))
    if len({path.resolve() for path in (initial, output, best)}) != 3:
        raise ValueError("initial, output and best paths must be distinct")
    if resume is not None and Path(resume).resolve() == initial.resolve():
        raise ValueError("resume must be a timed checkpoint, not the frozen initial checkpoint")
    if resume is None and (output.exists() or best.exists()):
        raise ValueError("output or best already exists; use resume or new paths")
    metrics_path = output.with_suffix(".metrics.jsonl")
    if metrics_path.resolve() in {initial.resolve(), best.resolve(), output.resolve()}:
        raise ValueError("metrics and checkpoint paths must be distinct")
    if resume is None and metrics_path.exists():
        raise ValueError("metrics already exist; use resume or a new output path")
    selected_device = select_device(device)
    initial_hash = _sha256(initial)
    print("Preparing and fingerprinting MCB:16 examples (outside training budget)...", flush=True)
    examples, fingerprint = _dataset(load_examples(recordings, ["mcb:16"]))
    training, validation = split_examples(examples, validation_fraction, seed)
    if not training or not validation:
        raise ValueError("training and validation splits must both contain examples")
    training_size, validation_size = len(training), len(validation)
    split = {
        "training_groups": sorted({example.group for example in training}),
        "validation_groups": sorted({example.group for example in validation}),
    }
    config = {
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "validation_fraction": float(validation_fraction),
        "seed": seed,
        "hidden": [256, 256],
        "teachers": ["mcb:16"],
        "optimizer": "AdamW",
        "weight_decay": 0.01,
    }
    identity = {
        "dataset_fingerprint": fingerprint,
        "training_config": config,
        "split": split,
        "initial_sha256": initial_hash,
    }
    model, payload = load_checkpoint(initial if resume is None else resume, selected_device)
    if model.hidden != (256, 256):
        raise ValueError("initial/resume model must have hidden widths 256, 256")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    if resume is None:
        progress = {
            "version": 1,
            "epoch_index": 0,
            "completed_epochs": 0,
            "next_batch_position": 0,
            "shuffle_seed": seed + 1,
            "updates": 0,
            "epoch_loss_sum": 0.0,
            "epoch_correct": 0,
            "cumulative_seconds": 0.0,
            "best_val_loss": None,
        }
        best_hash = None
        metrics = {}
    else:
        metadata = payload.get("metadata", {})
        for field, expected in identity.items():
            if metadata.get(field) != expected:
                raise ValueError(f"resume {field} is incompatible")
        progress = metadata.get("timed_progress")
        _validate_progress(progress, payload.get("epoch"), training_size, batch_size, seed)
        if not payload.get("optimizer_state"):
            raise ValueError("resume checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer_state"])
        if any(group["lr"] != learning_rate or group["weight_decay"] != 0.01 for group in optimizer.param_groups):
            raise ValueError("resume optimizer settings are incompatible")
        best_hash = metadata.get("best_checkpoint_sha256")
        if not best.exists() or not best_hash or _sha256(best) != best_hash:
            raise ValueError("resume best checkpoint is missing or incompatible")
        metrics = metadata.get("metrics", {})
        if (
            not isinstance(metrics, dict)
            or type(metrics.get("updates")) is not int
            or not 0 <= metrics["updates"] <= progress["updates"]
        ):
            raise ValueError("resume metrics do not match timed progress")
    training_tensors = _tensors(training)
    validation_tensors = _tensors(validation)
    del examples, training, validation
    validation_indices = list(range(validation_size))
    output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Device: {selected_device}; training examples: {training_size}; "
        f"validation examples: {validation_size}; budget: {seconds:g}s; "
        f"completed epochs: {progress['completed_epochs']}; "
        f"next batch position: {progress['next_batch_position']}",
        flush=True,
    )
    prior_seconds = progress["cumulative_seconds"]
    started = monotonic()
    last_saved = last_reported = started

    def elapsed():
        return monotonic() - started

    def save(path):
        progress["cumulative_seconds"] = prior_seconds + elapsed()
        model.eval()
        save_checkpoint(
            path,
            model,
            optimizer=optimizer,
            epoch=progress["completed_epochs"],
            metadata={
                **identity,
                "untrained": progress["updates"] == 0,
                "timed_progress": dict(progress),
                "best_checkpoint_sha256": best_hash,
                "metrics": dict(metrics),
            },
        )

    def record(event):
        row = {
            **metrics,
            "event": event,
            "epoch_index": progress["epoch_index"],
            "next_batch_position": progress["next_batch_position"],
            "elapsed_seconds": elapsed(),
            "cumulative_seconds": prior_seconds + elapsed(),
            "best_val_loss": progress["best_val_loss"],
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        return row

    def validate(event, train_loss=None, train_accuracy=None):
        nonlocal metrics, best_hash
        val_loss, val_accuracy = _epoch(
            model, validation_tensors, validation_indices, batch_size, selected_device
        )
        metrics = {
            "epoch": progress["completed_epochs"],
            "updates": progress["updates"],
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "train_examples": training_size,
            "val_examples": validation_size,
        }
        if progress["best_val_loss"] is None or val_loss < progress["best_val_loss"]:
            progress["best_val_loss"] = val_loss
            save(best)
            best_hash = _sha256(best)
        record(event)
        print(
            f"{event.capitalize()}: epoch={progress['completed_epochs']} "
            f"updates={progress['updates']} val_loss={val_loss:.6f} "
            f"val_accuracy={val_accuracy:.2%} best_val_loss={progress['best_val_loss']:.6f} "
            f"elapsed={elapsed():.1f}s/{seconds:g}s",
            flush=True,
        )

    if resume is None:
        validate("initial")
    indices = None
    while elapsed() < seconds:
        if indices is None:
            indices = list(range(training_size))
            random.Random(progress["shuffle_seed"]).shuffle(indices)
        position = progress["next_batch_position"]
        batch = indices[position : position + batch_size]
        features, masks, actions = (values[batch].to(selected_device) for values in training_tensors)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = masked_logits(model(features), masks)
        loss = functional.cross_entropy(logits, actions)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise ValueError("nonfinite training loss")
        correct = (logits.argmax(dim=-1) == actions).sum().item()
        loss.backward()
        optimizer.step()
        if selected_device.type == "mps":
            torch.mps.synchronize()
        progress["updates"] += 1
        progress["next_batch_position"] += len(batch)
        progress["epoch_loss_sum"] += loss_value * len(batch)
        progress["epoch_correct"] += correct
        if progress["next_batch_position"] == training_size:
            train_loss = progress["epoch_loss_sum"] / training_size
            train_accuracy = progress["epoch_correct"] / training_size
            progress["completed_epochs"] += 1
            progress["epoch_index"] += 1
            progress["next_batch_position"] = 0
            progress["shuffle_seed"] += 1
            progress["epoch_loss_sum"] = 0.0
            progress["epoch_correct"] = 0
            indices = None
            validate("epoch", train_loss, train_accuracy)
            save(output)
            last_saved = monotonic()
        elif monotonic() - last_saved >= 60:
            save(output)
            last_saved = monotonic()
        if monotonic() - last_reported >= 30:
            print(
                f"Progress: elapsed={elapsed():.1f}s/{seconds:g}s "
                f"epoch={progress['epoch_index']} "
                f"position={progress['next_batch_position']}/{training_size} "
                f"updates={progress['updates']}",
                flush=True,
            )
            last_reported = monotonic()
    if metrics.get("updates") != progress["updates"] or progress["next_batch_position"]:
        position = progress["next_batch_position"]
        validate(
            "partial",
            progress["epoch_loss_sum"] / position if position else None,
            progress["epoch_correct"] / position if position else None,
        )
    save(output)
    result = record("final")
    print(f"Saved last: {output}; best: {best}; elapsed={elapsed():.1f}s", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, nargs="+", required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="mps")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    options = parser.parse_args(argv)
    try:
        train(**vars(options))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
