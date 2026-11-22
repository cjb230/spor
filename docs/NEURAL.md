# The first neural player: `mlp`

`mlp` is a Python PyTorch policy that uses the existing REST game API. It can
play against another REST client on either seat. It is separate from the eight
native Rust strategies: the native tournament command and browser opponent
selector do not load Python checkpoints.

## Model and representation

The default architecture is **1082 → 256 → ReLU → 256 → ReLU → 487**, with
468,199 parameters. More precisely, each hidden linear layer is followed by ReLU;
the output layer returns raw logits. Inputs are float32, always relative to the
acting player:

| Input | Width |
|---|---:|
| Own hand: one presence flag per card | 54 |
| Nine stones × two sides × 54 card-presence flags | 972 |
| Stone ownership: none / you / them | 27 |
| First completed formation: none / you / them | 27 |
| Deck count / 42 and opponent hand count / 6 | 2 |
| **Total** | **1082** |

Card IDs follow Rust: red, blue, green, yellow, purple, brown; within each color,
values 1 through 9. A placement has action ID `9 * card_id + stone_index`, with
stone indexes 0–8. Action 486 passes. Illegal actions are masked before training
loss or move selection; pass is legal only when there are no legal placements.
Evaluation uses argmax, with the lowest action ID breaking equal-score ties.
Provable claims use the existing deterministic proof routine, not a neural head.

The encoding ignores arbitrary order within hands and formations while retaining
stone location and first-completion tie breakers. Unknown cards can be inferred
from the hand and board. Actual opponent hands, seeds, future draws, winners,
and recording metadata never enter the model. The current observation does not
include move history, so the model cannot infer hidden cards from historical
opponent behavior beyond what is present in the current position.

This fixed action head was chosen for simplicity and stable action meanings.
Smaller hand-slot heads depend on ordering; shared candidate scoring is a future
comparison. Wider networks may represent more complex policies but require more
computation and data. A value head, symmetry augmentation, transformer encoders,
and reinforcement learning are deferred until the imitation baseline is measured.

## Install

Use the project's Python environment, then install optional dependencies:

```sh
pyenv version
python -c 'import sys; print(sys.executable)'
python -m pip install -r requirements-ml.txt
python -m schotten.neural --help
```

This checkout's `.python-version` selects `spor`. The executable should be in that
environment; do not borrow another project's interpreter for training or tests.
The implementation was verified in `spor` with Python 3.15.2, PyTorch 2.15.0 and
NumPy 2.5.5. Both PyTorch and NumPy are included in the optional ML requirements.

Existing Python/Rust bots and servers do not require PyTorch. Training and play
accept `--device auto`, `cpu`, or `mps`. Auto chooses the Apple GPU when MPS is
available, otherwise CPU. Single-position inference may be faster on CPU; measure
both rather than assuming GPU inference always wins.

## Generate teacher games

Run a native tournament and retain complete recordings:

```sh
./rust/target/release/schotten-native tournament \
  --agents mcb:20 builder --games-per-pair 200 \
  --deck-seed 10001 --agent-seed 20001 \
  --record-games all --record-dir models/mlp-data
```

The command prints the generated recording directory. Either that directory or
its parent can be passed to training. Use a fresh data directory for unrelated
experiments; directories are searched recursively for `game-*.json` files.

The loader replays each game through the Python reference engine and checks its
deal, sequential turns, legal moves, claims, and terminal outcome. It extracts
the selected teacher's observation **before** each action. Both winning and losing
teacher games contribute examples. `--teachers mcb` matches every MCB sampling
count; `--teachers mcb:20` matches only that exact strategy. You can select several
teachers, but imitation of weaker policies can change what the model learns.

## Train and save weights

```sh
python -m schotten.neural train \
  --recordings models/mlp-data --teachers mcb \
  --output models/mlp-v1.pt --epochs 20 --device auto
```

Training uses AdamW and masked cross entropy. It prints training/validation loss
and teacher-action accuracy each epoch, and atomically replaces the output
checkpoint after each completed epoch. Keep distinct output filenames to preserve
different runs. The default learning rate is 0.001 and batch size is 256;
`--hidden 128 128` or `--hidden 512 512` selects another two-layer width.

Validation holds out complete deck-seed groups, including mirrored games and
games from different pairings with the same deal. It needs at least two distinct
decks. `--validation-fraction 0` explicitly disables validation for a tiny
experiment. Data is currently loaded into memory, so begin with hundreds of games
before scaling to a large corpus.

Checkpoints contain architecture and encoding versions, weights, optimizer state,
completed epoch, and training metadata. They load on CPU even if trained on MPS.
Inspect or continue a run with:

```sh
python -m schotten.neural inspect models/mlp-v1.pt
python -m schotten.neural train \
  --recordings models/mlp-data --teachers mcb \
  --resume models/mlp-v1.pt --output models/mlp-v2.pt \
  --epochs 20 --device auto
```

Here `--epochs 20` means 20 **additional** epochs. Resume validates dataset and
training compatibility; use the same split seed, widths, batch size and optimizer
settings. Changing datasets starts a new experiment rather than silently resuming
with a different validation split. Saved weights do not improve during normal play.
Switching CPU/MPS devices when resuming is supported, but can introduce numerical
differences; exact resume comparisons require the same device and software stack.

For a connectivity test without training, explicitly initialize random weights:

```sh
python -m schotten.neural init --output models/mlp-untrained.pt --seed 1
```

An initialized checkpoint is not a trained player. Likewise, a short smoke-test
training run verifies the machinery, not competitive strength. Teacher-move
accuracy is not a win-rate measurement; evaluate on held-out games before making
strength claims. Imitation provides a starting policy; outcome-based training or
self-play will be needed to optimize beyond matching the teacher.

## Play through REST

Start the Rust server normally. In one terminal, create a game:

```sh
python -m schotten.neural create \
  --checkpoint models/mlp-v1.pt --device cpu --seed 50001
```

It prints a game ID and waits for a second player. In another terminal:

```sh
python -m schotten.bot join GAME_ID --strategy greedy
```

The TUI can join that game instead. To use the neural player as seat 1, first
create a game using another client, then run:

```sh
python -m schotten.neural join GAME_ID --checkpoint models/mlp-v1.pt
```

Pass `--url http://host:8000` to select a server. The checkpoint is loaded locally;
the server continues to receive ordinary versioned REST turns.

## Controlled before/after experiments

For tens of millions of examples, use the disk-backed generation and training
commands in [Large MLP imitation experiments](STREAMING_EXPERIMENTS.md). The old
recording-based `train` command still loads its dataset into memory.

The [first completed MCB:16 comparison](../benchmarks/mlp-comparison-2026-11-21/RESULTS.md)
measured 20,000 games per checkpoint. Wins increased from 0.080% to 0.485%,
but the fixed teacher corpus overfit heavily; the validation-selected weights
came from epoch 3 rather than the end of the 20-minute training budget.

The first MCB:16 comparison uses a fresh policy, a frozen paired schedule, separate
teacher recordings, and a validation-selected checkpoint after 20 minutes of
training. Experiment artifacts and separate per-command logs live under
`models/mlp-experiment-2026-11-21/`. The dated experiment plan records the exact
protocol; `manifest.json` freezes seeds, initialization and code hashes.

Build the native batched evaluation worker from `rust/`:

```sh
cargo build --release --locked --example neural_worker
```

`python -m schotten.neural.evaluate` exchanges JSON-lines messages with this
worker. The Python process runs CPU policy inference; Rust owns the game, claim
proofs and MCB:16. Each worker snapshot exposes only the neural player's view.
Evaluation uses fixed argmax choices and independently seeded MCB decisions.
This bulk evaluation path is separate from the ordinary REST player.

A schedule is a JSON object with `schema_version: 1`, `opponent: "mcb:16"` and
`games`, an array of `{seed, bot_seed, mlp_seat}` objects. Pair each deck in both
seat orientations, keeping its MCB seed attached to the opponent. For example,
from the repository root, evaluate a slice of a previously frozen schedule:

```sh
python benchmarks/log_command.py --log RUN/logs/before-00000.log -- \
  python -m schotten.neural.evaluate \
  --checkpoint RUN/before.pt --schedule RUN/schedule.json \
  --worker rust/target/release/examples/neural_worker \
  --output RUN/before/00000.json --start 0 --count 1000
```

Replace `RUN` with the experiment directory. Reports include the complete game
schedule slice, outcomes, checkpoint/worker/schedule hashes and timing totals.
Existing reports and command logs are never silently overwritten.

The timed trainer continues from the frozen initialization, and can resume at an
exact minibatch position rather than restarting a partial epoch:

```sh
python benchmarks/log_command.py --log RUN/logs/train-1.log -- \
  python -m schotten.neural.timed_train \
  --recordings RUN/teacher-games --initial RUN/before.pt \
  --output RUN/last.pt --best RUN/best.pt --seconds 600 --device mps
```

Run another segment with the same options plus `--resume RUN/last.pt` and a new
command log. This trainer selects exactly `mcb:16` teacher decisions and requires
the default 256–256 architecture. Its wall-time budget includes optimization,
validation and checkpoint writes, but excludes loading/fingerprinting data.
Budget checks occur between batches; final validation/saving can slightly exceed
the requested time. Both last and best checkpoints are retained, including an
initial-weight best checkpoint if training fails to improve validation loss.

The frozen schedule generator and paired-result summarizer are available as
`python -m benchmarks.mlp_experiment`. Summary confidence intervals resample
whole deck pairs, retaining the dependence between mirrored and before/after
games. These intervals concern the fixed policy and opponent; one training run
does not measure variability across neural initialization or training seeds.
