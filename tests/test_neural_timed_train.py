import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from schotten.neural.data import Example
from schotten.neural.model import Policy, load_checkpoint, save_checkpoint


class Clock:
    def __init__(self):
        self.seconds = 0.0

    def __call__(self):
        return self.seconds


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def experiment(monkeypatch, tmp_path):
    from schotten.neural import timed_train

    examples = []
    for group in range(8):
        for action, signal in ((0, -1.0), (9, 1.0)):
            features = [0.0] * 1082
            features[0] = signal
            mask = [False] * 487
            mask[0] = mask[9] = True
            examples.append(Example(features, mask, action, str(group), "fixture.json"))
    clock = Clock()
    monkeypatch.setattr(timed_train, "monotonic", clock)
    monkeypatch.setattr(timed_train, "load_examples", lambda paths, teachers: examples)
    real_step = torch.optim.AdamW.step

    def step(optimizer, *args, **kwargs):
        result = real_step(optimizer, *args, **kwargs)
        clock.seconds += 1.0
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    torch.manual_seed(13)
    initial = tmp_path / "before.pt"
    save_checkpoint(initial, Policy(), metadata={"untrained": True})

    def run(**overrides):
        options = {
            "recordings": [Path("fixture.json")],
            "initial": initial,
            "output": tmp_path / "last.pt",
            "best": tmp_path / "best.pt",
            "seconds": 1.5,
            "batch_size": 3,
            "learning_rate": 0.001,
            "validation_fraction": 0.25,
            "seed": 7,
            "device": "cpu",
        }
        options.update(overrides)
        return timed_train.train(**options)

    return timed_train, run, examples, clock, initial


def assert_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_equal(left, right)
    else:
        assert first == second


def test_initial_validation_and_best_use_frozen_weights(experiment, monkeypatch, tmp_path):
    timed, run, _, clock, initial = experiment
    original = initial.read_bytes()
    real_epoch = timed._epoch

    def validate(*args, **kwargs):
        result = real_epoch(*args, **kwargs)
        clock.seconds += 2
        return result

    monkeypatch.setattr(timed, "_epoch", validate)
    metrics = run(seconds=1)
    _, before = load_checkpoint(initial)
    _, last = load_checkpoint(tmp_path / "last.pt")
    _, best = load_checkpoint(tmp_path / "best.pt")
    assert_equal(before["model_state"], last["model_state"])
    assert_equal(before["model_state"], best["model_state"])
    assert metrics["updates"] == 0
    assert best["epoch"] == 0
    assert initial.read_bytes() == original
    assert last["metadata"]["initial_sha256"] == hashlib.sha256(original).hexdigest()
    assert metrics["cumulative_seconds"] >= 2


def test_budget_stops_between_batches_and_saves_partial_epoch(experiment, tmp_path, capsys):
    _, run, _, _, _ = experiment
    metrics = run()
    _, payload = load_checkpoint(tmp_path / "last.pt")
    progress = payload["metadata"]["timed_progress"]
    assert payload["epoch"] == progress["completed_epochs"] == 0
    assert progress["epoch_index"] == 0
    assert progress["next_batch_position"] == 6
    assert progress["updates"] == metrics["updates"] == 2
    assert 1.5 <= progress["cumulative_seconds"] <= 2.5
    assert payload["optimizer_state"]["state"]
    assert metrics["val_loss"] > 0
    json.dumps(progress, allow_nan=False)
    assert "cpu" in capsys.readouterr().out


@pytest.mark.parametrize("batch_size,completed", [(3, 1), (5, 2)])
def test_middle_epoch_resume_matches_uninterrupted_adamw_exactly(experiment, tmp_path, batch_size, completed):
    _, run, _, _, _ = experiment
    run(output=tmp_path / "full.pt", best=tmp_path / "full-best.pt", seconds=5.5, batch_size=batch_size)
    run(seconds=1.5, batch_size=batch_size)
    run(resume=tmp_path / "last.pt", seconds=3.5, batch_size=batch_size)
    _, full = load_checkpoint(tmp_path / "full.pt")
    _, resumed = load_checkpoint(tmp_path / "last.pt")
    assert full["epoch"] == resumed["epoch"] == completed
    assert_equal(full["model_state"], resumed["model_state"])
    assert_equal(full["optimizer_state"], resumed["optimizer_state"])
    for field in ("epoch_index", "next_batch_position", "updates", "epoch_loss_sum", "epoch_correct"):
        assert full["metadata"]["timed_progress"][field] == resumed["metadata"]["timed_progress"][field]
    assert resumed["metadata"]["timed_progress"]["cumulative_seconds"] == 6
    assert full["metadata"]["metrics"]["val_loss"] == resumed["metadata"]["metrics"]["val_loss"]
    lines = [json.loads(line) for line in (tmp_path / "last.metrics.jsonl").read_text().splitlines()]
    assert sum(line["event"] == "initial" for line in lines) == 1
    assert sum(line["event"] == "final" for line in lines) == 2


@pytest.mark.parametrize("change", [
    {"batch_size": 4}, {"learning_rate": 0.002}, {"seed": 8},
    {"validation_fraction": 0.5},
])
def test_resume_rejects_settings_changes_before_overwriting(experiment, tmp_path, change):
    _, run, _, _, _ = experiment
    run()
    original = (tmp_path / "last.pt").read_bytes()
    with pytest.raises(ValueError, match="(?i)(config|incompatib)"):
        run(resume=tmp_path / "last.pt", **change)
    assert (tmp_path / "last.pt").read_bytes() == original


@pytest.mark.parametrize("change", ["dataset", "initial", "optimizer", "cursor", "best"])
def test_resume_rejects_changed_identity_or_missing_state(experiment, tmp_path, change):
    _, run, examples, _, initial = experiment
    run()
    path = tmp_path / "last.pt"
    if change == "dataset":
        examples[0] = replace(examples[0], action=9)
    elif change == "initial":
        save_checkpoint(initial, Policy())
    elif change == "best":
        (tmp_path / "best.pt").unlink()
    else:
        payload = torch.load(path, weights_only=True)
        if change == "optimizer":
            payload["optimizer_state"] = None
        else:
            payload["metadata"]["timed_progress"]["next_batch_position"] = 1
        torch.save(payload, path)
    original = path.read_bytes()
    with pytest.raises((ValueError, FileNotFoundError), match="(?i)(dataset|initial|optimizer|progress|cursor|best)"):
        run(resume=path)
    assert path.read_bytes() == original


def test_periodic_last_checkpoint_and_progress_during_long_epoch(experiment, monkeypatch, tmp_path, capsys):
    timed, run, _, clock, _ = experiment
    real_step = torch.optim.AdamW.step
    real_save = timed.save_checkpoint
    saved = []

    def slow_step(optimizer, *args, **kwargs):
        result = real_step(optimizer, *args, **kwargs)
        clock.seconds += 34
        return result

    def save(path, *args, **kwargs):
        real_save(path, *args, **kwargs)
        if Path(path).name == "last.pt":
            saved.append(kwargs["metadata"]["timed_progress"]["updates"])

    monkeypatch.setattr(torch.optim.AdamW, "step", slow_step)
    monkeypatch.setattr(timed, "save_checkpoint", save)
    run(seconds=80)
    assert saved == [2, 3]
    assert "elapsed" in capsys.readouterr().out.lower()


def test_periodic_checkpoint_resumes_even_before_next_validation(experiment, monkeypatch, tmp_path):
    timed, run, _, clock, _ = experiment
    real_step = torch.optim.AdamW.step
    real_save = timed.save_checkpoint
    snapshot = {}

    def slow_step(optimizer, *args, **kwargs):
        result = real_step(optimizer, *args, **kwargs)
        clock.seconds += 34
        return result

    def save(path, *args, **kwargs):
        real_save(path, *args, **kwargs)
        if Path(path).name == "last.pt" and not snapshot:
            snapshot["last"] = Path(path).read_bytes()
            snapshot["best"] = (tmp_path / "best.pt").read_bytes()

    monkeypatch.setattr(torch.optim.AdamW, "step", slow_step)
    monkeypatch.setattr(timed, "save_checkpoint", save)
    run(seconds=80)
    _, uninterrupted = load_checkpoint(tmp_path / "last.pt")
    (tmp_path / "last.pt").write_bytes(snapshot["last"])
    (tmp_path / "best.pt").write_bytes(snapshot["best"])
    _, periodic = load_checkpoint(tmp_path / "last.pt")
    assert periodic["metadata"]["metrics"]["updates"] == 0
    assert periodic["metadata"]["timed_progress"]["updates"] == 2
    run(resume=tmp_path / "last.pt", seconds=10)
    _, resumed = load_checkpoint(tmp_path / "last.pt")
    assert_equal(uninterrupted["model_state"], resumed["model_state"])
    assert_equal(uninterrupted["optimizer_state"], resumed["optimizer_state"])


def test_final_partial_epoch_can_provide_best_weights(experiment, tmp_path):
    _, run, _, _, _ = experiment
    metrics = run(seconds=9.5, learning_rate=0.01)
    _, best = load_checkpoint(tmp_path / "best.pt")
    _, last = load_checkpoint(tmp_path / "last.pt")
    assert metrics["next_batch_position"] == 6
    assert metrics["val_loss"] < 0.1
    assert best["metadata"]["metrics"]["updates"] == 10
    assert_equal(best["model_state"], last["model_state"])


def test_validation_and_checkpoint_time_count_toward_budget(experiment, monkeypatch):
    timed, run, _, clock, _ = experiment
    real_epoch = timed._epoch
    real_save = timed.save_checkpoint

    def validate(*args, **kwargs):
        result = real_epoch(*args, **kwargs)
        clock.seconds += 2
        return result

    def save(*args, **kwargs):
        real_save(*args, **kwargs)
        clock.seconds += 1

    monkeypatch.setattr(timed, "_epoch", validate)
    monkeypatch.setattr(timed, "save_checkpoint", save)
    metrics = run(seconds=2.5)
    assert metrics["updates"] == 0
    assert metrics["cumulative_seconds"] == 4


def test_best_keeps_initial_baseline_when_validation_does_not_improve(experiment, monkeypatch, tmp_path):
    timed, run, _, _, initial = experiment
    real_epoch = timed._epoch
    calls = 0

    def worsening_validation(*args, **kwargs):
        nonlocal calls
        real_epoch(*args, **kwargs)
        calls += 1
        return float(calls), 0.0

    monkeypatch.setattr(timed, "_epoch", worsening_validation)
    run()
    _, before = load_checkpoint(initial)
    _, best = load_checkpoint(tmp_path / "best.pt")
    _, last = load_checkpoint(tmp_path / "last.pt")
    assert_equal(before["model_state"], best["model_state"])
    assert last["metadata"]["timed_progress"]["best_val_loss"] == 1
    assert last["metadata"]["metrics"]["val_loss"] == 2


@pytest.mark.parametrize("change", [
    {"seconds": 0}, {"seconds": -1}, {"seconds": float("nan")},
    {"seconds": float("inf")}, {"seconds": True}, {"batch_size": 0},
    {"learning_rate": 0}, {"validation_fraction": 0}, {"validation_fraction": 1},
])
def test_invalid_budget_or_settings_do_not_create_checkpoints(experiment, tmp_path, change):
    _, run, _, _, _ = experiment
    with pytest.raises(ValueError):
        run(**change)
    assert not (tmp_path / "last.pt").exists()
    assert not (tmp_path / "best.pt").exists()


def test_frozen_initial_and_best_paths_cannot_be_overwritten(experiment, tmp_path):
    _, run, _, _, initial = experiment
    original = initial.read_bytes()
    with pytest.raises(ValueError, match="(?i)(distinct|path|initial)"):
        run(output=initial)
    with pytest.raises(ValueError, match="(?i)(distinct|path)"):
        run(best=tmp_path / "last.pt")
    assert initial.read_bytes() == original


def test_cli_exposes_experiment_options(experiment, monkeypatch, tmp_path):
    timed, _, _, _, _ = experiment
    received = {}

    def train(**kwargs):
        received.update(kwargs)

    monkeypatch.setattr(timed, "train", train)
    assert timed.main([
        "--recordings", "corpus", "--initial", "before.pt", "--output", "last.pt",
        "--best", "best.pt", "--seconds", "600", "--resume", "last.pt", "--device", "mps",
        "--seed", "1", "--batch-size", "512", "--learning-rate", "0.001",
        "--validation-fraction", "0.2",
    ]) == 0
    assert received["seconds"] == 600
    assert received["initial"] == Path("before.pt")
    assert received["recordings"] == [Path("corpus")]
    assert received["resume"] == Path("last.pt")
    assert received["device"] == "mps"
