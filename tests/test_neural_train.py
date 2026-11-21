from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from schotten.neural import train as training
from schotten.neural.data import Example
from schotten.neural.model import load_checkpoint, masked_logits, save_checkpoint


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def examples():
    result = []
    for group in range(8):
        for action, signal in ((0, -1.0), (9, 1.0)):
            features = [0.0] * 1082
            features[0] = signal
            mask = [False] * 487
            mask[0] = mask[9] = True
            result.append(Example(features, mask, action, str(group), "fixture.json"))
    return result


@pytest.fixture
def run_training(monkeypatch, examples, tmp_path):
    monkeypatch.setattr(training, "load_examples", lambda paths, teachers: examples)

    def run(**overrides):
        options = {
            "recordings": [Path("fixture.json")],
            "output": tmp_path / "weights.pt",
            "teachers": ["mcb:8"],
            "epochs": 4,
            "batch_size": 5,
            "learning_rate": 0.03,
            "validation_fraction": 0.25,
            "seed": 7,
            "device": "cpu",
            "hidden": (16, 16),
        }
        options.update(overrides)
        return training.train(**options)

    return run


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_nested_equal(left, right)
    else:
        assert first == second


def test_cpu_training_learns_and_saves_each_epoch(run_training, tmp_path, monkeypatch, capsys):
    saved_epochs = []
    real_save = training.save_checkpoint

    def save_and_read(path, model, **kwargs):
        real_save(path, model, **kwargs)
        _, payload = load_checkpoint(path)
        saved_epochs.append(payload["epoch"])

    monkeypatch.setattr(training, "save_checkpoint", save_and_read)
    metrics = run_training(epochs=12)
    assert metrics["epoch"] == 12
    assert metrics["train_loss"] < 0.1
    assert metrics["train_accuracy"] == 1.0
    assert metrics["val_loss"] < 0.1
    assert metrics["val_accuracy"] == 1.0
    assert saved_epochs == list(range(1, 13))
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 12
    assert all("train" in line.lower() and "val" in line.lower() for line in lines)
    model, payload = load_checkpoint(tmp_path / "weights.pt")
    assert model.hidden == (16, 16)
    assert payload["optimizer_state"]
    assert len(payload["metadata"]["dataset_fingerprint"]) == 64


def test_resumed_training_matches_uninterrupted_weights_and_optimizer(run_training, tmp_path):
    full_path = tmp_path / "full.pt"
    resumed_path = tmp_path / "resumed.pt"
    full_metrics = run_training(output=full_path, epochs=4)
    run_training(output=resumed_path, epochs=2)
    resumed_metrics = run_training(output=resumed_path, resume=resumed_path, epochs=2)
    assert resumed_metrics == full_metrics
    full_model, full_payload = load_checkpoint(full_path)
    resumed_model, resumed_payload = load_checkpoint(resumed_path)
    assert full_payload["epoch"] == resumed_payload["epoch"] == 4
    assert_nested_equal(full_model.state_dict(), resumed_model.state_dict())
    assert_nested_equal(full_payload["optimizer_state"], resumed_payload["optimizer_state"])


def test_checkpoint_roundtrip_preserves_predictions(run_training, examples, tmp_path):
    run_training()
    model, payload = load_checkpoint(tmp_path / "weights.pt")
    other_path = tmp_path / "copy.pt"
    save_checkpoint(other_path, model, epoch=payload["epoch"], metadata=payload["metadata"])
    restored, _ = load_checkpoint(other_path)
    features = torch.tensor([example.features for example in examples])
    masks = torch.tensor([example.mask for example in examples])
    with torch.no_grad():
        before = masked_logits(model(features), masks)
        after = masked_logits(restored(features), masks)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert set(after.argmax(dim=1).tolist()) == {0, 9}


@pytest.mark.parametrize(
    "changes",
    [
        {"learning_rate": 0.02},
        {"batch_size": 8},
        {"validation_fraction": 0.5},
        {"seed": 8},
        {"hidden": (32, 16)},
        {"teachers": ["planner"]},
    ],
)
def test_resume_rejects_incompatible_training_configuration(run_training, tmp_path, changes):
    run_training(epochs=1)
    path = tmp_path / "weights.pt"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="(?i)(resume|incompatib|config|hidden)"):
        run_training(resume=path, **changes)
    assert path.read_bytes() == original


@pytest.mark.parametrize("field", ["features", "mask", "action", "group"])
def test_resume_rejects_changed_training_contents(run_training, examples, tmp_path, field):
    run_training(epochs=1)
    path = tmp_path / "weights.pt"
    original = path.read_bytes()
    if field == "features":
        changed = list(examples[0].features)
        changed[1] = 1.0
    elif field == "mask":
        changed = list(examples[0].mask)
        changed[18] = True
    elif field == "action":
        changed = 9
    else:
        changed = "new-deal"
    examples[0] = replace(examples[0], **{field: changed})
    with pytest.raises(ValueError, match="(?i)(dataset|fingerprint)"):
        run_training(resume=path)
    assert path.read_bytes() == original


def test_resume_accepts_recordings_moved_to_another_directory(run_training, examples, tmp_path):
    run_training(epochs=1)
    for index, example in enumerate(examples):
        examples[index] = replace(example, source="moved/fixture.json")
    metrics = run_training(
        recordings=[Path("moved/fixture.json")], resume=tmp_path / "weights.pt", epochs=1
    )
    assert metrics["epoch"] == 2


def test_resume_requires_optimizer_state(run_training, tmp_path):
    run_training(epochs=1)
    path = tmp_path / "weights.pt"
    model, payload = load_checkpoint(path)
    save_checkpoint(path, model, epoch=payload["epoch"], metadata=payload["metadata"])
    with pytest.raises(ValueError, match="(?i)optimizer"):
        run_training(resume=path)


@pytest.mark.parametrize(
    "changes",
    [
        {"epochs": 0},
        {"epochs": float("nan")},
        {"epochs": 1.5},
        {"batch_size": 0},
        {"batch_size": float("inf")},
        {"batch_size": True},
        {"learning_rate": 0},
        {"learning_rate": -0.1},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"validation_fraction": -0.1},
        {"validation_fraction": 1},
        {"validation_fraction": float("nan")},
        {"validation_fraction": float("inf")},
        {"seed": float("nan")},
        {"hidden": (0, 16)},
        {"hidden": (16, float("nan"))},
    ],
)
def test_invalid_configuration_fails_before_creating_weights(run_training, tmp_path, changes):
    with pytest.raises(ValueError):
        run_training(**changes)
    assert not (tmp_path / "weights.pt").exists()


def test_training_without_validation_reports_missing_validation_metrics(run_training):
    metrics = run_training(validation_fraction=0, epochs=1)
    assert metrics["val_loss"] is None
    assert metrics["val_accuracy"] is None
    assert metrics["train_loss"] >= 0


def test_only_legal_action_has_zero_loss_and_perfect_accuracy(run_training, examples):
    for index, example in enumerate(examples):
        mask = [False] * 487
        mask[example.action] = True
        examples[index] = replace(example, mask=mask)
    metrics = run_training(learning_rate=1e-9, epochs=1)
    assert metrics["train_loss"] == metrics["val_loss"] == 0.0
    assert metrics["train_accuracy"] == metrics["val_accuracy"] == 1.0


def test_empty_dataset_is_rejected(run_training, examples):
    examples.clear()
    with pytest.raises(ValueError, match="(?i)(empty|no .*examples)"):
        run_training()


@pytest.mark.parametrize("problem", ["illegal_target", "no_legal_moves", "nonfinite_features"])
def test_malformed_examples_are_rejected(run_training, examples, problem):
    if problem == "illegal_target":
        examples[0] = replace(examples[0], action=486)
    elif problem == "no_legal_moves":
        examples[0] = replace(examples[0], mask=[False] * 487)
    else:
        features = list(examples[0].features)
        features[0] = float("nan")
        examples[0] = replace(examples[0], features=features)
    with pytest.raises(ValueError):
        run_training()
