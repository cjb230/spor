"""Neural placement decisions with deterministic claims and the existing REST runner."""
import torch

from ..bot import Bot
from .encoding import decode_action, encode_view, legal_mask
from .model import load_checkpoint, masked_logits, select_device


class NeuralBot(Bot):
    def __init__(self, client, checkpoint, device="auto", verbose=False, name="mlp"):
        super().__init__(client, verbose=verbose, name=name)
        self.strategy = "mlp"
        self.device = select_device(device)
        self.model, self.checkpoint = load_checkpoint(checkpoint, self.device)

    def choose(self, view):
        features = torch.tensor([encode_view(view)], dtype=torch.float32, device=self.device)
        mask = torch.tensor([legal_mask(view)], dtype=torch.bool, device=self.device)
        with torch.inference_mode():
            action = masked_logits(self.model(features), mask).argmax(dim=-1).item()
        play = decode_action(action)
        return play, self.provable_stones(view, play)
