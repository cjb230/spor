"""Small CPU tests for the streaming trainer's update and restart contract."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from schotten.neural.model import Policy, load_checkpoint, save_checkpoint
from schotten.neural import stream_data


class PackedDataset(stream_data.Dataset):
    """Real compact disk shards, with unique bits identifying individual rows."""

    def __init__(self, root):
        self.decoded = []
        self.visits = []
        stream_data.create_dataset(root, {"examples": 8, "validation_examples": 5})
        for split, sizes in (("train", (5, 3)), ("validation", (3, 2))):
            seen = 0
            for shard, count in enumerate(sizes):
                features = np.zeros((count, 1082), dtype=np.float32)
                features[:, 0] = np.arange(count) % 2
                base = 100 if split == "train" else 200
                features[np.arange(count), base + seen + np.arange(count)] = 1
                features[:, -2:] = [1., .5]
                mask = np.zeros((count, 487), dtype=bool)
                mask[:, [0, 9]] = True
                arrays = stream_data.pack_examples(features, mask, (features[:, 0] * 9).astype(np.int64))
                stream_data.append_shard(root, split, arrays, cursor={"shard": shard})
                seen += count
        stream_data.finish_dataset(root)
        super().__init__(root)


class QuietProgress:
    def __init__(self, label, interval=1):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def update(self, completed, total, detail=""):
        pass

    def close(self):
        pass


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    dataset = PackedDataset(tmp_path / "dataset")
    unpack = stream_data.unpack_batch

    def tracked_unpack(arrays, indices):
        result = unpack(arrays, indices)
        dataset.decoded.append(len(indices))
        for row in result[0]:
            dataset.visits.extend(np.flatnonzero(row[100:108]).tolist())
        return result

    monkeypatch.setattr(stream_data, "unpack_batch", tracked_unpack)
    monkeypatch.setattr(stream_data, "Progress", QuietProgress)
    trainer = importlib.import_module("schotten.neural.stream_train")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(13)
    initial = tmp_path / "initial.pt"
    save_checkpoint(initial, Policy(), metadata={"untrained": True})

    def run(**changes):
        options = dict(dataset=dataset, initial=initial, output=tmp_path / "run", epochs=3,
                       batch_size=2, seed=7, device="cpu", checkpoint_interval=1)
        options.update(changes)
        return trainer.train_stream(**options)

    yield trainer, dataset, initial, run
    torch.set_num_threads(previous)


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_state(a, b)
    else:
        assert left == right


def test_total_epochs_stream_batches_and_preserve_initial(experiment, tmp_path):
    _, dataset, initial, run = experiment
    original = initial.read_bytes()
    result = run(epochs=1)
    _, first = load_checkpoint(tmp_path / "run/last.pt")
    assert first["epoch"] == result["epoch"] == 1
    assert result["updates"] == 5  # ceil(5/2) + ceil(3/2), no dropped tails
    assert result["train_examples"] == 8
    assert result["val_examples"] == 5
    run(resume=tmp_path / "run/last.pt", epochs=3)
    _, final = load_checkpoint(tmp_path / "run/last.pt")
    assert final["epoch"] == 3
    assert final["metadata"]["stream_progress"]["updates"] == 15
    assert initial.read_bytes() == original
    assert max(dataset.decoded) <= 2
    rows = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows if row["event"] == "epoch"] == [1, 2, 3]
    assert sum(row["event"] == "initial" for row in rows) == 1
    assert (tmp_path / "run/best.pt").exists()
    assert [dataset.visits.count(index) for index in range(8)] == [3] * 8


@pytest.mark.parametrize("stop_update", [1, 3, 4, 5, 7])
def test_periodic_minibatch_resume_is_exact(experiment, tmp_path, monkeypatch, stop_update):
    trainer, _, _, run = experiment
    run(output=tmp_path / "full")
    clock = [0.0]
    monkeypatch.setattr(trainer, "monotonic", lambda: clock[0])
    real_step = torch.optim.AdamW.step
    real_save = trainer.save_checkpoint

    def step(optimizer, *args, **kwargs):
        result = real_step(optimizer, *args, **kwargs)
        clock[0] += 2
        return result

    def save(path, *args, **kwargs):
        real_save(path, *args, **kwargs)
        progress = kwargs["metadata"]["stream_progress"]
        if Path(path).name == "last.pt" and progress["updates"] == stop_update:
            raise KeyboardInterrupt

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    monkeypatch.setattr(trainer, "save_checkpoint", save)
    with pytest.raises(KeyboardInterrupt):
        run()
    monkeypatch.setattr(trainer, "save_checkpoint", real_save)
    run(resume=tmp_path / "run/last.pt")
    _, full = load_checkpoint(tmp_path / "full/last.pt")
    _, resumed = load_checkpoint(tmp_path / "run/last.pt")
    equal_state(full["model_state"], resumed["model_state"])
    equal_state(full["optimizer_state"], resumed["optimizer_state"])
    equal_state(full["metadata"]["metrics"], resumed["metadata"]["metrics"])
    for field in ("completed_epochs", "shard_position", "batch_offset", "updates"):
        assert full["metadata"]["stream_progress"][field] == resumed["metadata"]["stream_progress"][field]
    _, full_best = load_checkpoint(tmp_path / "full/best.pt")
    _, resume_best = load_checkpoint(tmp_path / "run/best.pt")
    equal_state(full_best["model_state"], resume_best["model_state"])


@pytest.mark.parametrize("change", ["dataset", "initial", "batch", "lr", "seed", "cursor", "optimizer", "best"])
def test_resume_rejects_incompatible_state_without_writes(experiment, tmp_path, change):
    _, dataset, initial, run = experiment
    run(epochs=1)
    last = tmp_path / "run/last.pt"
    options = {}
    if change == "dataset":
        dataset.fingerprint = "changed"
    elif change == "initial":
        save_checkpoint(initial, Policy())
    elif change in ("batch", "lr", "seed"):
        options[{"batch": "batch_size", "lr": "learning_rate", "seed": "seed"}[change]] = {"batch": 3, "lr": .02, "seed": 8}[change]
    elif change == "best":
        _, payload = load_checkpoint(last)
        (tmp_path / "run" / payload["metadata"]["best_checkpoint_file"]).unlink()
    else:
        payload = torch.load(last, weights_only=True)
        if change == "cursor":
            payload["metadata"]["stream_progress"]["batch_offset"] = 1
        else:
            payload["optimizer_state"] = None
        torch.save(payload, last)
    before = last.read_bytes()
    with pytest.raises(ValueError):
        run(resume=last, **options)
    assert last.read_bytes() == before


@pytest.mark.parametrize("changes", [{"epochs": 0}, {"epochs": True}, {"batch_size": 0},
                                      {"learning_rate": float("nan")}, {"progress_interval": 0},
                                      {"checkpoint_interval": float("inf")}])
def test_invalid_options_do_not_create_output(experiment, tmp_path, changes):
    _, _, _, run = experiment
    with pytest.raises(ValueError):
        run(**changes)
    assert not (tmp_path / "run").exists()


def test_initial_last_is_resumable_before_first_update(experiment, tmp_path, monkeypatch):
    trainer, _, initial, run = experiment
    real_save = trainer.save_checkpoint

    def save(path, *args, **kwargs):
        real_save(path, *args, **kwargs)
        if Path(path).name == "last.pt":
            raise KeyboardInterrupt

    monkeypatch.setattr(trainer, "save_checkpoint", save)
    with pytest.raises(KeyboardInterrupt):
        run(epochs=1)
    _, before = load_checkpoint(initial)
    _, last = load_checkpoint(tmp_path / "run/last.pt")
    _, best = load_checkpoint(tmp_path / "run/best.pt")
    equal_state(before["model_state"], last["model_state"])
    equal_state(before["model_state"], best["model_state"])
    assert last["metadata"]["stream_progress"]["updates"] == 0
    assert last["metadata"]["best_checkpoint_sha256"] == stream_data.digest(tmp_path / "run/best.pt")
    monkeypatch.setattr(trainer, "save_checkpoint", real_save)
    run(resume=tmp_path / "run/last.pt", epochs=1)
    _, final = load_checkpoint(tmp_path / "run/last.pt")
    assert final["epoch"] == 1


def test_completed_target_performs_no_extra_updates(experiment, tmp_path):
    _, dataset, _, run = experiment
    run(epochs=1)
    before = (tmp_path / "run/last.pt").read_bytes()
    visits = list(dataset.visits)
    result = run(resume=tmp_path / "run/last.pt", epochs=1)
    assert result["updates"] == 5
    assert dataset.visits == visits
    assert (tmp_path / "run/last.pt").read_bytes() == before


def test_output_lock_excludes_other_trainers(experiment, tmp_path):
    _, _, _, run = experiment
    with stream_data.exclusive_lock(tmp_path / "run"):
        with pytest.raises(ValueError, match="another writer"):
            run()
    assert not (tmp_path / "run/last.pt").exists()
    run(epochs=1)  # A failed lock attempt did not leave the output locked.


def test_best_is_minimum_full_validation_and_metrics_are_weighted(experiment, tmp_path):
    _, dataset, _, run = experiment
    run()
    rows = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    _, best = load_checkpoint(tmp_path / "run/best.pt")
    model, last = load_checkpoint(tmp_path / "run/last.pt")
    assert best["metadata"]["metrics"]["val_loss"] == min(row["val_loss"] for row in rows)
    losses, correct = [], 0
    with torch.no_grad():
        for entry in dataset.entries("validation"):
            arrays = dataset.load(entry)
            features, mask, labels = stream_data.unpack_batch(arrays, np.arange(entry["count"]))
            logits = model(torch.from_numpy(features)).masked_fill(~torch.from_numpy(mask), -torch.inf)
            losses.extend(torch.nn.functional.cross_entropy(logits, torch.from_numpy(labels), reduction="none").tolist())
            correct += (logits.argmax(-1) == torch.from_numpy(labels)).sum().item()
    assert last["metadata"]["metrics"]["val_loss"] == pytest.approx(sum(losses) / 5, rel=1e-6)
    assert last["metadata"]["metrics"]["val_accuracy"] == correct / 5


@pytest.mark.parametrize("change", ["incomplete", "empty", "corrupt_best", "earlier_target"])
def test_rejects_invalid_dataset_and_resume_target(experiment, tmp_path, change):
    _, dataset, _, run = experiment
    if change in ("incomplete", "empty"):
        if change == "incomplete":
            dataset.manifest["complete"] = False
        else:
            dataset.manifest["counts"]["validation"] = 0
        with pytest.raises(ValueError):
            run()
        assert not (tmp_path / "run").exists()
    else:
        run(epochs=3)
        if change == "corrupt_best":
            _, payload = load_checkpoint(tmp_path / "run/last.pt")
            (tmp_path / "run" / payload["metadata"]["best_checkpoint_file"]).write_bytes(b"corrupt")
        last = tmp_path / "run/last.pt"
        before = last.read_bytes()
        with pytest.raises(ValueError):
            run(resume=last, epochs=1 if change == "earlier_target" else 3)
        assert last.read_bytes() == before


def test_crash_after_best_write_keeps_previous_last_resumable(experiment, tmp_path, monkeypatch):
    trainer, _, _, run = experiment
    run(output=tmp_path / "full", epochs=1)
    real_save = trainer.save_checkpoint

    def save(path, *args, **kwargs):
        real_save(path, *args, **kwargs)
        if Path(path).name.startswith("best") and kwargs["epoch"] == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(trainer, "save_checkpoint", save)
    with pytest.raises(KeyboardInterrupt):
        run(epochs=1)
    _, previous = load_checkpoint(tmp_path / "run/last.pt")
    assert previous["epoch"] == 0
    monkeypatch.setattr(trainer, "save_checkpoint", real_save)
    run(resume=tmp_path / "run/last.pt", epochs=1)
    _, full = load_checkpoint(tmp_path / "full/last.pt")
    _, recovered = load_checkpoint(tmp_path / "run/last.pt")
    equal_state(full["model_state"], recovered["model_state"])
    equal_state(full["optimizer_state"], recovered["optimizer_state"])


@pytest.mark.parametrize("state", ["missing", "corrupt", "newer"])
def test_resume_restores_convenience_best_from_committed_version(experiment, tmp_path, state):
    _, _, initial, run = experiment
    run(epochs=1)
    best = tmp_path / "run/best.pt"
    expected = best.read_bytes()
    if state == "missing":
        best.unlink()
    elif state == "corrupt":
        best.write_bytes(b"broken convenience copy")
    else:
        best.write_bytes(initial.read_bytes())
    run(resume=tmp_path / "run/last.pt", epochs=1)
    assert best.read_bytes() == expected


def test_crash_after_best_publication_restores_previous_best(experiment, tmp_path, monkeypatch):
    trainer, _, _, run = experiment
    real_publish = trainer._publish_best
    publications = []

    def publish(source, destination):
        real_publish(source, destination)
        publications.append(destination.read_bytes())
        if len(publications) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(trainer, "_publish_best", publish)
    with pytest.raises(KeyboardInterrupt):
        run(epochs=1)
    assert len(publications) == 2
    assert publications[0] != publications[1]
    _, payload = load_checkpoint(tmp_path / "run/last.pt")
    assert payload["epoch"] == 0
    monkeypatch.setattr(trainer, "_publish_best", real_publish)
    # Stop immediately after recovery of the old convenience best, before the
    # next update; otherwise later validation could replace it again.
    def stop(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(torch.optim.AdamW, "step", stop)
    with pytest.raises(KeyboardInterrupt):
        run(resume=tmp_path / "run/last.pt", epochs=1)
    assert (tmp_path / "run/best.pt").read_bytes() == publications[0]


def test_progress_has_one_active_heartbeat(experiment, monkeypatch):
    _, _, _, run = experiment
    active = []
    peak = [0]

    class Progress(QuietProgress):
        def __init__(self, label, interval=1):
            self.label = label

        def __enter__(self):
            active.append(self)
            peak[0] = max(peak[0], len(active))
            return self

        def close(self):
            active.remove(self)

    monkeypatch.setattr(stream_data, "Progress", Progress)
    run(epochs=1)
    assert peak[0] == 1
    assert active == []


def test_total_target_cannot_precede_partial_epoch_work(experiment, tmp_path, monkeypatch):
    trainer, _, _, run = experiment
    clock = [0.]
    monkeypatch.setattr(trainer, "monotonic", lambda: clock[0])
    real_step, real_save = torch.optim.AdamW.step, trainer.save_checkpoint

    def step(optimizer, *args, **kwargs):
        result = real_step(optimizer, *args, **kwargs)
        clock[0] += 2
        return result

    def save(path, *args, **kwargs):
        real_save(path, *args, **kwargs)
        if Path(path).name == "last.pt" and kwargs["metadata"]["stream_progress"]["updates"] == 6:
            raise KeyboardInterrupt

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    monkeypatch.setattr(trainer, "save_checkpoint", save)
    with pytest.raises(KeyboardInterrupt):
        run()
    monkeypatch.setattr(trainer, "save_checkpoint", real_save)
    last = tmp_path / "run/last.pt"
    before = last.read_bytes()
    with pytest.raises(ValueError, match="target"):
        run(resume=last, epochs=1)
    assert last.read_bytes() == before
