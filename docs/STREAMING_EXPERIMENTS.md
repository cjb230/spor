# Large MLP imitation experiments

The `generate` and `train-stream` commands keep only a shard and decoded batch
in memory. Generation runs the native Rust teacher; training uses the existing
PyTorch MLP and writes ordinary portable `.pt` checkpoints. Run from the repository
root in the local `spor` Python environment.

These experiments compare 40 million training samples presented three times
(120 million presentations) with 125 million samples presented once. Both use
the same initialization, batch size 512, learning rate 0.001 and AdamW weight
decay 0.01. They do not add augmentation or stop early. Samples come from fresh
games; this is not a promise that every observed position is globally unique.

## Setup

Build the generator, then create a shared initialization once. Each command below
has its own log; `tee` preserves live progress. `pipefail` propagates failures
through the logging pipeline. The old `benchmarks/log_command.py` wrapper buffers
console output until completion and has a 1,150-second default timeout, so it is
not used for these long-running commands.

The generator is already built in this checkout, so the build step can be
skipped here. On another checkout it requires an installed Rust toolchain.

```sh
mkdir -p models/large-mlp/logs
set -o pipefail

python -c 'import sys, torch; print(sys.executable); print(torch.__version__); print("MPS:", torch.backends.mps.is_available())' \
  2>&1 | tee models/large-mlp/logs/environment.log

(cd rust && cargo build --release --locked --example neural_generate) \
  2>&1 | tee models/large-mlp/logs/build-generator.log

python -m schotten.neural init \
  --output models/large-mlp/shared-initial.pt --seed 20261121 \
  2>&1 | tee models/large-mlp/logs/initialize.log
```

This development session used a cached Rust toolchain because `cargo` is not
on its default PATH. To rebuild with that cache on this machine, use this in
place of the standard build command above:

```sh
(cd rust && env CARGO_HOME=/private/tmp/spor-cargo \
  RUSTUP_HOME=/private/tmp/spor-rustup RUSTUP_TOOLCHAIN=stable \
  /private/tmp/spor-cargo/bin/cargo build --release --locked --example neural_generate) \
  2>&1 | tee models/large-mlp/logs/build-generator-cached.log
```

Do not rerun initialization once either experiment has started. Install the
optional dependencies from `requirements-ml.txt` if needed. The generation
commands print examples generated, throughput, elapsed time and ETA on a single
line refreshed approximately every second, including while native workers run.

## Experiment A: 40 million examples, three passes

```sh
python -m schotten.neural generate \
  --teacher mcb:16 --examples 40000000 --validation-examples 100000 \
  --output datasets/mcb16-40m --seed 1001 \
  --progress-interval 1 \
  2>&1 | tee models/large-mlp/logs/generate-40m.log

python -m schotten.neural train-stream \
  --dataset datasets/mcb16-40m \
  --initial models/large-mlp/shared-initial.pt \
  --epochs 3 --batch-size 512 --learning-rate 0.001 --seed 1 \
  --output models/large-mlp/40m-three-passes --device mps \
  --progress-interval 1 --checkpoint-interval 60 \
  2>&1 | tee models/large-mlp/logs/train-40m-three-passes.log
```

## Experiment B: 125 million examples, one pass

```sh
python -m schotten.neural generate \
  --teacher mcb:16 --examples 125000000 --validation-examples 100000 \
  --output datasets/mcb16-125m --seed 1002 \
  --progress-interval 1 \
  2>&1 | tee models/large-mlp/logs/generate-125m.log

python -m schotten.neural train-stream \
  --dataset datasets/mcb16-125m \
  --initial models/large-mlp/shared-initial.pt \
  --epochs 1 --batch-size 512 --learning-rate 0.001 --seed 1 \
  --output models/large-mlp/125m-one-pass --device mps \
  --progress-interval 1 --checkpoint-interval 60 \
  2>&1 | tee models/large-mlp/logs/train-125m-one-pass.log
```

Each generation command creates an additional 100,000 validation examples from
separate games. Those examples are excluded from the 40/125-million training
counts. Opponents default to Builder, Tactician, Planner and MCB:16; override
with `--opponents ...` if desired. `--shard-size` defaults to 65,536 examples.
Generation is likely to dominate total runtime; the progress rate provides a
machine-specific estimate. Run a small corpus in a different directory first
if you want a timing estimate before starting.

Generation defaults to all available CPU cores. Set `--threads 10` or
`--threads 14` explicitly to compare this Mac's performance cores with all cores.
The native worker keeps a bounded queue of independent games, returns them in
deterministic counter order, and retains completed future games across requests.
Changing thread count does not change native decisions or example order, although
the generation manifest requires the same configured count when resuming.

A short M4 Pro smoke run before the queue optimization, with eight native
worker threads and the default opponent mix,
generated 2,048 training plus 128 validation samples in 20.31 seconds (about
107 samples/second). If sustained, that corresponds to roughly 4.3 days for
40 million samples and 13.5 days for 125 million, excluding training. This is
an extrapolation from a short run, not a sustained throughput benchmark; use
the live rate/ETA to refine it. Subsequent queue and Builder optimizations now
measure **1,314–1,360 samples/second** over 32,768-example runs with 14 threads,
projecting to **8.2–8.5 hours for 40M** and **25.5–26.4 hours for 125M**.
The subsequent completed 40M-training + 100K-validation run took **34,500
seconds (9 hours 35 minutes)** as reported by the user, averaging about
**1,162 generated examples/second**. That sustained rate suggests approximately
**30 hours for 125M**, excluding training, and supersedes the short-run timing
extrapolations above. The completed corpus occupies about **916 MB** in 613
shards. See [Generation performance](GENERATION_PERFORMANCE.md) and the
[Builder report](../benchmarks/builder-optimization-2026-11-22/RESULTS.md) for
the controlled comparisons, raw results and limitations. The requested example
counts and MCB:16 teacher strength are preserved.

## Interrupt and resume

Generation resumes by repeating the same command with `--resume`, retaining all
committed shards. A partly generated shard may need to be regenerated. Use a new
log filename for every invocation, for example `generate-40m-resume-01.log`.
Changing generation options or the native binary is rejected on resume.

Training periodically saves `last.pt`, including weights, optimizer and exact
shuffled batch position. To resume experiment A, keep **`--epochs 3`**:

```sh
python -m schotten.neural train-stream \
  --dataset datasets/mcb16-40m \
  --initial models/large-mlp/shared-initial.pt \
  --epochs 3 --batch-size 512 --learning-rate 0.001 --seed 1 \
  --output models/large-mlp/40m-three-passes --device mps \
  --resume models/large-mlp/40m-three-passes/last.pt \
  --progress-interval 1 --checkpoint-interval 60 \
  2>&1 | tee models/large-mlp/logs/train-40m-resume-01.log
```

Unlike the older `train` command, `train-stream --epochs` is the **total target**,
not additional epochs. Completed runs do not train extra passes when resumed.
Dataset identity, initial weights and optimizer settings must match. A hard
termination loses work since the most recent successful checkpoint; resume
replays that work from the checkpoint. Exact numerical resume is tested on CPU;
MPS and software/device changes can introduce numerical differences.

## Results and storage

Training retains `last.pt` for the final requested pass and `best.pt` for the
lowest measured validation loss, plus JSON-lines metrics. Teacher-action
accuracy measures imitation, not playing strength. Evaluate both experiments
against a frozen opponent schedule before claiming one is a stronger player.
Keep the versioned `best-*.pt` files alongside `last.pt`: resume checks the
immutable best version referenced by that checkpoint, then repairs `best.pt`
if a previous interruption left the convenience copy incomplete or newer.

The packed representation uses 200 bytes per example before compression:
135 observation-bit bytes, two card counts, 61 legal-mask bytes and a two-byte
action label. That is about 8 GB for 40 million and 25 GB for 125 million,
before compression, plus validation and temporary shard files. A 4,096-example
MCB:16 pilot compressed to 94,997 bytes (23.2 bytes per example), projecting to
about **2.9 GB / 2.7 GiB for 125 million examples**. This is a small-sample
estimate for the current compressed NPZ format, not a guaranteed final size.
Allow 5–10 GB of free space for the corpus, validation, logs and working files;
separate experiments or extra corpus copies require additional space. Shards include
hashes and are decoded in bounded batches. Shard order and rows within shards
are shuffled, rather than constructing a 125-million-row global permutation.

Dataset and model directories are ignored by Git. Keep the manifest, shards,
checkpoints and logs together for reproducibility; a manifest alone does not
contain the examples or weights.

Further generation optimization ideas are recorded in
[Generation performance](GENERATION_PERFORMANCE.md).
