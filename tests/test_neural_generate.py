"""Native stream protocol and shared-dataset generation regressions."""
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from schotten.game import Game
from schotten.neural.encoding import action_id, encode_view, legal_mask
from schotten.schemas import CreateGameRequest, GameView, Play, TurnRequest


WORKER = Path(os.environ.get("NEURAL_GENERATE_WORKER", "rust/target/release/examples/neural_generate")).resolve()
NOW = datetime(2026, 11, 21, tzinfo=timezone.utc)


def request(**changes):
    return dict(op="generate", seed=1001, split="train", start=0, count=113,
                teacher="mcb:0", opponents=["mcb:0"], trace=False) | changes


def native(command, threads=1):
    if not WORKER.is_file():
        pytest.skip("build rust example neural_generate to run native parity tests")
    return subprocess.run([str(WORKER)], input=json.dumps(command).encode() + b"\n",
                          capture_output=True, timeout=90,
                          env=os.environ | {"RAYON_NUM_THREADS": str(threads)})


def unpack_response(data):
    assert data[:8] == b"SPGEN001"
    next_game, count = struct.unpack("<QQ", data[8:24])
    assert len(data) == 24 + count * 200
    rows = np.frombuffer(data[24:], dtype=np.uint8).reshape(count, 200)
    return next_game, dict(packed=rows[:, :135], counts=rows[:, 135:137],
                           mask=rows[:, 137:198],
                           actions=rows[:, 198:200].copy().view("<u2").reshape(-1))


@pytest.mark.parametrize("teacher,count", [("mcb:0", 113), ("mcb:16", 64)])
def test_native_encoding_matches_python_pre_action_views_and_replay(teacher, count):
    result = native(request(trace=True, teacher=teacher, opponents=[teacher], count=count))
    assert result.returncode == 0, result.stderr.decode()
    document = json.loads(result.stdout)
    assert len(document["examples"]) == count
    games = {}
    seats = set()
    for example in document["examples"]:
        seed = example["deck_seed"]
        if seed not in games:
            game = Game("parity", CreateGameRequest(seed=seed), NOW)
            game.join(NOW)
            games[seed] = game
        game = games[seed]
        view = GameView.model_validate(example["view"])
        expected = game.view(game.turn)
        assert encode_view(view) == encode_view(expected)
        assert legal_mask(view) == legal_mask(expected)
        assert "opponent_hand" not in example["view"]
        assert "draw_order" not in example["view"]
        seats.add(view.you)
        play = None if example["play"] is None else Play.model_validate(example["play"])
        row = np.asarray(example["row"], dtype=np.uint8)
        features = np.unpackbits(row[:135], bitorder="little").tolist()
        assert features == encode_view(expected)[:1080]
        assert row[135:137].tolist() == [view.deck_count, view.opponent_hand_count]
        assert np.unpackbits(row[137:198], bitorder="little")[:487].tolist() == legal_mask(expected)
        label = int.from_bytes(row[198:200].tobytes(), "little")
        assert label == action_id(play)
        assert legal_mask(expected)[label]
        game.play_turn(game.turn, TurnRequest(version=game.version, play=play,
                                             claims=example["claims"]), NOW)
    assert seats == {0, 1}
    assert any(game.over for game in games.values())


def test_native_binary_matches_trace_and_is_independent_of_threads():
    command = request(count=57, opponents=["builder", "mcb:0", "tactician"])
    first = native(command)
    assert first.returncode == 0, first.stderr.decode()
    assert native(command, threads=3).stdout == first.stdout
    cursor, arrays = unpack_response(first.stdout)
    traced = json.loads(native(command | {"trace": True}).stdout)
    assert cursor == traced["next_game"] > 0
    rows = np.array([item["row"] for item in traced["examples"]], dtype=np.uint8)
    assert np.array_equal(arrays["packed"], rows[:, :135])
    assert np.array_equal(arrays["actions"], rows[:, 198:200].copy().view("<u2").ravel())


def test_native_split_decks_are_disjoint_and_cursor_skips_truncated_game():
    train = json.loads(native(request(trace=True, count=1)).stdout)
    validation = json.loads(native(request(trace=True, count=1, split="validation")).stdout)
    following = json.loads(native(request(trace=True, count=1, start=train["next_game"])).stdout)
    seeds = [doc["examples"][0]["deck_seed"] for doc in (train, validation, following)]
    assert seeds == [1001, 1002, 1003]
    assert train["next_game"] == 1


@pytest.mark.parametrize("changes", [{"count": 0}, {"count": 65537}, {"split": "bad"},
    {"opponents": []}, {"teacher": "builder"}, {"teacher": "mcb:bad"},
    {"seed": -1}, {"seed": 2**63 - 1, "start": 1}])
def test_native_rejects_bad_requests(changes):
    result = native(request(**changes))
    assert result.returncode != 0
    assert not result.stdout
    assert result.stderr


def generator():
    return importlib.import_module("schotten.neural.generate")


def test_python_exact_counts_and_compressed_shards(tmp_path):
    if not WORKER.is_file():
        pytest.skip("native generator is not built")
    root = tmp_path / "dataset"
    result = generator().generate(root, 73, teacher="mcb:0", validation_examples=11,
                                 opponents=("mcb:0",), shard_size=31, threads=2, worker=WORKER)
    assert result["complete"]
    assert result["counts"] == {"train": 73, "validation": 11}
    shards = list(root.glob("**/*.npz"))
    assert len(shards) == 4
    for path in shards:
        with np.load(path, allow_pickle=False) as arrays:
            assert set(arrays.files) == {"packed", "counts", "mask", "actions"}
            n = len(arrays["actions"])
            assert n <= 31
            mask = np.unpackbits(arrays["mask"], axis=1, bitorder="little")[:, :487]
            assert mask[np.arange(n), arrays["actions"]].all()
    assert generator().generate(root, 73, teacher="mcb:0", validation_examples=11,
        opponents=("mcb:0",), shard_size=31, threads=2, worker=WORKER, resume=True) == result


@pytest.mark.parametrize("changes", [{"examples": 0}, {"validation_examples": -1},
    {"seed": -1}, {"shard_size": 0}, {"threads": 0}, {"opponents": ()},
    {"teacher": "greedy"}, {"examples": True}, {"progress_interval": 0}])
def test_python_rejects_bad_arguments_before_creating_output(tmp_path, changes):
    options = dict(output=tmp_path / "new", examples=3, validation_examples=0, worker=WORKER) | changes
    with pytest.raises(ValueError):
        generator().generate(**options)
    assert not options["output"].exists()


@pytest.fixture
def synthetic_worker(tmp_path):
    """Protocol-only worker: isolates orchestration failures from the game engine."""
    path = tmp_path / "worker"
    path.write_text(f"#!{sys.executable}\n" + '''import json, os, struct, sys, time
mode = os.environ.get("GENERATE_TEST_MODE", "normal")
for line in sys.stdin:
    r = json.loads(line)
    if mode == "slow":
        time.sleep(.25)
    if mode == "hang":
        time.sleep(30)
    if mode == "error":
        print("deliberate failure", file=sys.stderr, flush=True)
        sys.exit(7)
    n = r["count"]
    header = b"SPGEN001" + struct.pack("<QQ", r["start"] + n, n)
    row = bytearray(200)
    row[135:137] = bytes([42, 6])
    row[197] = 64
    row[198:200] = struct.pack("<H", 486)
    if mode == "header":
        header = b"BADMAGIC" + header[8:]
    if mode == "count":
        header = header[:16] + struct.pack("<Q", n + 1)
    if mode == "cursor":
        header = header[:8] + struct.pack("<Q", r["start"]) + header[16:]
    if mode == "illegal":
        row[197] = 0
    if mode == "partial":
        sys.stdout.buffer.write(header[:4])
        sys.stdout.buffer.flush()
        time.sleep(30)
    sys.stdout.buffer.write(header + row * n)
    sys.stdout.buffer.flush()
if mode == "bad_exit":
    sys.exit(9)
''')
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("after_commit", [False, True])
def test_resume_replays_uncommitted_work_and_matches_fresh_shards(tmp_path, monkeypatch, after_commit):
    if not WORKER.is_file():
        pytest.skip("native generator is not built")
    module = generator()
    options = dict(examples=49, validation_examples=9, teacher="mcb:0", opponents=("mcb:0",),
                   shard_size=17, threads=2, worker=WORKER)
    root = tmp_path / "resumed"
    original = module.append_shard

    def interrupt(*args, **kwargs):
        if after_commit:
            original(*args, **kwargs)
        raise RuntimeError("interrupted")

    monkeypatch.setattr(module, "append_shard", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        module.generate(root, **options)
    monkeypatch.setattr(module, "append_shard", original)
    manifest = module.open_manifest(root)
    # Simulate a shard rename that succeeded just before a manifest commit failed.
    orphan = root / f'train-{len(manifest["shards"]):06d}.npz'
    orphan.write_bytes(b"uncommitted orphan")
    resumed = module.generate(root, resume=True, **options)
    fresh_root = tmp_path / "fresh"
    fresh = module.generate(fresh_root, **options)
    assert resumed == fresh
    assert [p.read_bytes() for p in sorted(root.glob("*.npz"))] == [
        p.read_bytes() for p in sorted(fresh_root.glob("*.npz"))]


def test_resume_rejects_changed_options_binary_and_corrupt_shards(tmp_path, synthetic_worker):
    module = generator()
    root = tmp_path / "dataset"
    options = dict(examples=3, validation_examples=1, worker=synthetic_worker, threads=8)
    original = module.generate(root, **options)
    for change in (dict(seed=1002), dict(examples=4), dict(teacher="mcb:8"),
                   dict(shard_size=10), dict(opponents=("builder",)), dict(threads=1)):
        with pytest.raises(ValueError, match="configuration"):
            module.generate(root, resume=True, **(options | change))
    contents = synthetic_worker.read_bytes()
    synthetic_worker.write_bytes(contents + b"\n# changed\n")
    with pytest.raises(ValueError, match="binary"):
        module.generate(root, resume=True, **options)
    synthetic_worker.write_bytes(contents)
    (root / original["shards"][0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        module.generate(root, resume=True, **options)


def test_resume_integrity_scan_reports_progress(tmp_path, synthetic_worker, monkeypatch, capsys):
    import time
    module = generator()
    root = tmp_path / 'dataset'
    options = dict(examples=3, validation_examples=1, worker=synthetic_worker)
    module.generate(root, **options)
    capsys.readouterr()
    original = module.digest

    def slow_digest(path):
        if Path(path).suffix == '.npz':
            time.sleep(.06)
        return original(path)

    monkeypatch.setattr(module, 'digest', slow_digest)
    module.generate(root, resume=True, progress_interval=.01, **options)
    output = capsys.readouterr().err
    assert 'Verifying' in output
    assert output.count('\r') >= 3


def test_default_threads_uses_available_cpu_count(tmp_path, synthetic_worker, monkeypatch):
    module = generator()
    monkeypatch.setattr(module.os, 'cpu_count', lambda: 14)
    manifest = module.generate(tmp_path / 'automatic', 3, validation_examples=1,
                               worker=synthetic_worker)
    assert manifest['config']['threads'] == 14


@pytest.mark.parametrize('threads', [1, 3, 14])
def test_persistent_requests_preserve_pre_queue_rows_and_cursors(threads):
    """Golden hashes captured from the original eight-game worker before replacement."""
    import hashlib
    if not WORKER.is_file():
        pytest.skip('native generator is not built')
    base = dict(op='generate', seed=3001, split='train', teacher='mcb:0',
                opponents=['builder', 'mcb:0'])
    changes = [dict(count=1, start=0), dict(count=73), dict(count=19),
               dict(count=13, start=0, seed=4001),
               dict(count=3, start=0, seed=4001, opponents=['mcb:0']),
               dict(count=13, start=0, split='validation'), dict(count=1, start=0)]
    expected = [
        (1, 'a649602239a271d8ba503a1deddb9ba1f1084562379a4bb71faa9f7fec6add9d'),
        (6, 'c1f61d9b424902e84edbe8ede52fe1d17cb7c9ac00e4e8f500dd1a18c1f4c614'),
        (7, '825f0233fe111d2abcde7e0b7c2d8004a2867247cb9a0cd1de57fe2db0820248'),
        (1, 'a60444a8f93c1fb1ffa7571a6423bba5115b2ad6dfe02cbdd5c2ba400d21103c'),
        (1, 'dc952ecadd9ee6ba2adf13532dd6cbbfeebb1e25d6c3657f16fc011ac6c8e27a'),
        (1, 'dcf44fe0801c80b121139ee89510602c2117c343f7fe97096aaca3bc3dd18be4'),
        (1, 'a649602239a271d8ba503a1deddb9ba1f1084562379a4bb71faa9f7fec6add9d'),
    ]
    cursor = 0
    with generator()._Worker(WORKER, threads) as worker:
        for change, (next_game, checksum) in zip(changes, expected):
            cursor, rows = worker.request(base | dict(start=cursor) | change)
            assert cursor == next_game
            assert hashlib.sha256(rows.tobytes()).hexdigest() == checksum


def test_exclusive_writer_and_existing_dataset_are_protected(tmp_path, synthetic_worker):
    module = generator()
    root = tmp_path / "dataset"
    with module.exclusive_lock(root):
        with pytest.raises(ValueError, match="another writer"):
            module.generate(root, 3, validation_examples=1, worker=synthetic_worker)
    first = module.generate(root, 3, validation_examples=1, worker=synthetic_worker)
    with pytest.raises(ValueError, match="resume"):
        module.generate(root, 3, validation_examples=1, worker=synthetic_worker)
    assert module.open_manifest(root) == first


def test_heartbeat_and_counts_advance_between_requests_before_shard_commit(
        tmp_path, synthetic_worker, monkeypatch, capsys):
    module = generator()
    monkeypatch.setenv("GENERATE_TEST_MODE", "slow")
    monkeypatch.setattr(module, "REQUEST_ROWS", 2)
    original = module.append_shard
    observed = []

    def append(*args, **kwargs):
        observed.append(capsys.readouterr().err)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "append_shard", append)
    module.generate(tmp_path / "dataset", 6, validation_examples=0, worker=synthetic_worker,
                    shard_size=6, progress_interval=.02)
    assert len(observed) == 1
    assert "0 / 6" in observed[0] and "2 / 6" in observed[0] and "4 / 6" in observed[0]
    assert "native active" in observed[0] and "ETA" in observed[0]
    assert observed[0].count("\r") >= 10


@pytest.mark.parametrize("mode,message", [("error", "deliberate failure"), ("header", "header"),
    ("count", "count"), ("cursor", "cursor"), ("illegal", "illegal"),
    ("bad_exit", "9"), ("hang", "timed out"), ("partial", "timed out")])
def test_worker_failures_leave_resumable_manifest_and_no_live_process(
        tmp_path, synthetic_worker, monkeypatch, mode, message):
    module = generator()
    monkeypatch.setenv("GENERATE_TEST_MODE", mode)
    monkeypatch.setattr(module, "RESPONSE_TIMEOUT", .5 if mode in ("hang", "partial") else 5)
    monkeypatch.setattr(module, "EXIT_TIMEOUT", 1)
    processes = []
    original = module.subprocess.Popen

    def popen(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    root = tmp_path / "dataset"
    with pytest.raises((RuntimeError, ValueError), match=message):
        module.generate(root, 3, validation_examples=1, worker=synthetic_worker)
    assert not module.open_manifest(root)["complete"]
    assert processes and all(process.poll() is not None for process in processes)
