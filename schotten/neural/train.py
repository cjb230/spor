"""Supervised policy training with reproducible, resumable epoch checkpoints."""

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Sequence

import torch
from torch.nn import functional

from .data import Example, load_examples, split_examples
from .encoding import ACTION_SIZE, INPUT_SIZE
from .model import Policy, load_checkpoint, masked_logits, save_checkpoint, select_device


def _finite_number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_options(epochs, batch_size, learning_rate, validation_fraction, seed, hidden):
    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    _finite_number(learning_rate, "learning_rate")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    _finite_number(validation_fraction, "validation_fraction")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if (
        not isinstance(hidden, (list, tuple))
        or len(hidden) != 2
        or any(type(width) is not int or width <= 0 for width in hidden)
    ):
        raise ValueError("hidden must contain two positive integer widths")


def _dataset(examples):
    if not examples:
        raise ValueError("no training examples found")
    rows = []
    for example in examples:
        if len(example.features) != INPUT_SIZE or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in example.features
        ):
            raise ValueError(f"examples must have {INPUT_SIZE} finite features")
        if len(example.mask) != ACTION_SIZE or any(type(value) is not bool for value in example.mask):
            raise ValueError(f"examples must have {ACTION_SIZE} boolean action flags")
        if (
            type(example.action) is not int
            or not 0 <= example.action < ACTION_SIZE
            or not example.mask[example.action]
        ):
            raise ValueError("each example must target a legal action")
        if not isinstance(example.group, str) or not example.group:
            raise ValueError("each example must identify its deck group")
        content = json.dumps(
            [example.features, example.mask, example.action, example.group],
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        rows.append((hashlib.sha256(content).digest(), example))
    rows.sort(key=lambda row: row[0])
    fingerprint = hashlib.sha256(b"schotten-mlp-dataset-v1\0")
    for digest, _ in rows:
        fingerprint.update(digest)
    return [example for _, example in rows], fingerprint.hexdigest()


def _tensors(examples: Sequence[Example]):
    features = torch.tensor([example.features for example in examples], dtype=torch.float32)
    if not torch.isfinite(features).all().item():
        raise ValueError("features must be representable as finite float32 numbers")
    masks = torch.tensor([example.mask for example in examples], dtype=torch.bool)
    actions = torch.tensor([example.action for example in examples], dtype=torch.long)
    return features, masks, actions


def _epoch(model, tensors, indices, batch_size, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss = 0.0
    correct = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            features, masks, actions = (values[batch].to(device) for values in tensors)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = masked_logits(model(features), masks)
            loss = functional.cross_entropy(logits, actions)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise ValueError("nonfinite loss; check training features and learning_rate")
            total_loss += loss_value * len(batch)
            correct += (logits.argmax(dim=-1) == actions).sum().item()
            if optimizer is not None:
                loss.backward()
                optimizer.step()
    return total_loss / len(indices), correct / len(indices)


def train(
    recordings: Sequence[Path],
    output: Path,
    teachers: Sequence[str],
    epochs=10,
    batch_size=256,
    learning_rate=0.001,
    validation_fraction=0.2,
    seed=1,
    device="auto",
    resume=None,
    hidden=(256, 256),
) -> dict:
    """Imitate recorded actions and return the last epoch's loss and accuracy.

    Epochs are additional completed epochs, including when resuming. Resume
    requires identical dataset contents, teacher selection, split seed/fraction,
    hidden widths, batch size and learning rate; learning-rate overrides are
    rejected. Paths and source names are excluded from dataset identity. Each
    epoch is shuffled using seed + its absolute epoch number. CPU execution
    resumes exactly; switching devices can introduce numerical differences.

    Accuracy is a fraction in [0, 1]. Training metrics aggregate predictions
    made while updating; validation metrics evaluate the resulting model.
    A zero validation fraction disables validation and returns None for its
    metrics. Checkpoints are atomically replaced after every completed epoch.
    """
    _validate_options(epochs, batch_size, learning_rate, validation_fraction, seed, hidden)
    if isinstance(teachers, str) or not teachers or any(
        not isinstance(teacher, str) or not teacher.strip() for teacher in teachers
    ):
        raise ValueError("at least one nonempty teacher strategy is required")
    teachers = sorted(set(teachers))
    examples, fingerprint = _dataset(load_examples(recordings, teachers))
    training, validation = split_examples(examples, validation_fraction, seed)
    if not training:
        raise ValueError("training split contains no examples")
    split = {
        "training_groups": sorted({example.group for example in training}),
        "validation_groups": sorted({example.group for example in validation}),
    }
    config = {
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "validation_fraction": float(validation_fraction),
        "seed": seed,
        "hidden": list(hidden),
        "teachers": teachers,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
    }
    selected_device = select_device(device)
    torch.manual_seed(seed % (1 << 64))
    completed_epochs = 0
    payload = None
    if resume is None:
        model = Policy(hidden).to(selected_device)
    else:
        model, payload = load_checkpoint(resume, selected_device)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("dataset_fingerprint") != fingerprint:
            raise ValueError("resume dataset fingerprint does not match the training examples")
        if metadata.get("training_config") != config or model.hidden != tuple(hidden):
            raise ValueError("resume training configuration is incompatible")
        if metadata.get("split") != split:
            raise ValueError("resume dataset split is incompatible")
        completed_epochs = payload.get("epoch")
        if type(completed_epochs) is not int or completed_epochs < 0:
            raise ValueError("resume checkpoint has an invalid completed epoch")
        if not payload.get("optimizer_state"):
            raise ValueError("resume checkpoint has no optimizer state")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
        if any(group["lr"] != learning_rate for group in optimizer.param_groups):
            raise ValueError("resume optimizer learning_rate is incompatible")
    training_tensors = _tensors(training)
    validation_tensors = _tensors(validation) if validation else None
    metadata = {
        "untrained": False,
        "dataset_fingerprint": fingerprint,
        "training_config": config,
        "split": split,
    }
    for absolute_epoch in range(completed_epochs + 1, completed_epochs + epochs + 1):
        indices = list(range(len(training)))
        random.Random(seed + absolute_epoch).shuffle(indices)
        train_loss, train_accuracy = _epoch(
            model, training_tensors, indices, batch_size, selected_device, optimizer
        )
        val_loss, val_accuracy = None, None
        if validation_tensors is not None:
            val_loss, val_accuracy = _epoch(
                model,
                validation_tensors,
                list(range(len(validation))),
                batch_size,
                selected_device,
            )
        metrics = {
            "epoch": absolute_epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "train_examples": len(training),
            "val_examples": len(validation),
        }
        model.eval()
        save_checkpoint(
            output, model, optimizer=optimizer, epoch=absolute_epoch, metadata={**metadata, "metrics": metrics}
        )
        validation_text = (
            f"val_loss={val_loss:.6f} val_accuracy={val_accuracy:.2%}"
            if val_loss is not None
            else "val_loss=n/a val_accuracy=n/a"
        )
        print(
            f"Epoch {absolute_epoch}: train_loss={train_loss:.6f} "
            f"train_accuracy={train_accuracy:.2%} {validation_text}",
            flush=True,
        )
    return metrics
