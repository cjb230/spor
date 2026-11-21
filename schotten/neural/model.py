"""MLP policy, device selection and portable versioned checkpoints."""
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

from .encoding import ACTION_SIZE, ENCODING_VERSION, INPUT_SIZE

CHECKPOINT_VERSION = 1


class Policy(nn.Module):
    def __init__(self, hidden=(256, 256)):
        super().__init__()
        if len(hidden) != 2 or any(type(width) is not int or width < 1 for width in hidden):
            raise ValueError("hidden must contain two positive integer widths")
        self.hidden = tuple(hidden)
        self.layers = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], ACTION_SIZE),
        )

    def forward(self, features):
        return self.layers(features)


def select_device(name="auto") -> torch.device:
    if name not in ("auto", "cpu", "mps"):
        raise ValueError("device must be auto, cpu or mps")
    available = torch.backends.mps.is_available()
    if name == "mps" and not available:
        raise ValueError("MPS is unavailable; use --device cpu or a compatible Apple Silicon PyTorch installation")
    return torch.device("mps" if available and name != "cpu" else "cpu")


def masked_logits(logits, mask):
    if logits.shape != mask.shape or logits.shape[-1] != ACTION_SIZE:
        raise ValueError("logits and mask must have matching (..., 487) shapes")
    if mask.dtype != torch.bool or not mask.any(dim=-1).all().item():
        raise ValueError("each decision must have at least one legal action and a boolean mask")
    return logits.masked_fill(~mask, -torch.inf)


def save_checkpoint(path, model, optimizer=None, epoch=0, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_VERSION,
        "name": "mlp",
        "encoding_version": ENCODING_VERSION,
        "input_size": INPUT_SIZE,
        "action_size": ACTION_SIZE,
        "hidden": list(model.hidden),
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metadata": metadata or {},
        "torch_version": str(torch.__version__),
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"schema_version": CHECKPOINT_VERSION, "name": "mlp",
                "encoding_version": ENCODING_VERSION, "input_size": INPUT_SIZE,
                "action_size": ACTION_SIZE}
    if not isinstance(payload, dict):
        raise ValueError("invalid MLP checkpoint")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"incompatible checkpoint {key}; expected {value!r}")
    model = Policy(payload["hidden"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, payload
