# MLP agent design

> Historical design/development record. Later changes may supersede the original
> scope, limits and verification counts. See the [current documentation](../../README.md)
> for supported commands and behavior.

The first neural strategy is named `mlp`. It is a Python PyTorch player using
the existing REST API, with no Rust neural inference dependency.

Encode only the acting player's observation: 54 hand-presence flags; nine stones
with two 54-card presence vectors (972 flags); nine three-way ownership vectors
(27 flags); nine three-way first-completion vectors (27 flags); deck count / 42
and opponent hand count / 6. Total: 1,082 float32 inputs. Categories are ordered
none, you, them. Stone order is significant; card order within a hand or formation
is not. Unknown cards are derivable. Hidden hands, future draws and seeds never
enter the model. History-dependent inference is outside this initial model.

Use Linear(1082,256), ReLU, Linear(256,256), ReLU, Linear(256,487): 468,199 parameters.
Action 9*card_id+stone chooses a placement; 486 passes. Card IDs follow Rust's
red/blue/green/yellow/purple/brown order, each with values 1–9. Mask illegal moves
before loss, softmax or argmax. Terminal observations have no legal actions.
Claims use the existing deterministic proof helper; the network learns placements.

A fixed action head is the first baseline because it has stable action meanings
and simple batching. Hand-slot outputs depend on hand ordering; shared candidate
scoring may improve reuse but is deferred. A value head, symmetry augmentation,
transformers and reinforcement learning are future experiments.

Initial training is supervised imitation of selected strategies in native
tournament recordings. Replay each recording through the Python reference engine,
using its revealed initial deal only to reconstruct correct player observations.
Take examples before each selected player's action. Split by deck seed so games
sharing deals cannot cross training and validation. Fail on inconsistent or illegal
records. Train with masked cross entropy and AdamW; report loss and move accuracy.

Save versioned checkpoints with model and optimizer state, architecture/encoding
metadata, completed epoch, training options and dataset identity. Resume checks
compatibility and preserves the split and optimizer state. Select CPU or MPS;
auto selects MPS when available. Untrained initialization is explicitly separate
from trained checkpoints. Validate learning and save/reload behavior on CPU, and
exercise MPS when available. Existing non-ML commands must not import torch.
