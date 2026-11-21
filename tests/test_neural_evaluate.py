import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from schotten.neural.model import Policy, save_checkpoint


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    checkpoint = tmp_path / "before.pt"
    model = Policy((4, 4))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.layers[-1].bias[486] = 100
        model.layers[-1].bias[27] = 10
        model.layers[-1].bias[81] = 10
    save_checkpoint(checkpoint, model)
    games = [{"seed": index, "bot_seed": -(index + 10), "mlp_seat": index % 2}
             for index in range(7)]
    schedule = tmp_path / "schedule.json"
    schedule.write_text(json.dumps({"schema_version": 1, "opponent": "mcb:16", "games": games}))
    requests = tmp_path / "requests.jsonl"
    monkeypatch.setenv("EVAL_REQUESTS", str(requests))
    worker = tmp_path / "worker"
    worker.write_text(f"#!{sys.executable}\n" + '''import json, os, signal, sys, time
mode = os.environ.get("EVAL_MODE", "normal")
if mode == "ignore_terminate":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
snapshots = []
steps = {}
for line in sys.stdin:
    request = json.loads(line)
    with open(os.environ["EVAL_REQUESTS"], "a") as handle:
        handle.write(json.dumps({"request": request, "threads": os.environ.get("RAYON_NUM_THREADS")}) + "\\n")
    if mode == "error":
        print(json.dumps({"error": "deliberate worker failure"}), flush=True)
        continue
    if mode == "eof":
        sys.exit(7)
    if mode == "hang":
        time.sleep(30)
    if mode == "partial":
        sys.stdout.write('{"games":')
        sys.stdout.flush()
        time.sleep(30)
    if mode == "invalid_json":
        print("not json", flush=True)
        continue
    if mode == "output_race":
        with open(os.environ["EVAL_OUTPUT"], "w") as handle:
            handle.write("another evaluation")
    if request["op"] == "new":
        snapshots = []
        specifications = request["games"]
        steps = {}
        for index, spec in enumerate(specifications):
            steps[index] = 0
            snapshots.append({"id": index, "view": {
                "game_id": str(index), "version": 1 + spec["mlp_seat"],
                "status": "in_progress", "you": spec["mlp_seat"], "turn": spec["mlp_seat"],
                "phase": "your_turn", "hand": [] if mode == "pass" else [
                    {"color": "blue", "value": 1}, {"color": "red", "value": 4}],
                "opponent_hand_count": 6, "deck_count": 40,
                "stones": [{"index": stone, "yours": [], "theirs": []} for stone in range(9)],
                "unseen_cards": [], "winner": None, "win_reason": None}})
    else:
        active = [snapshot for snapshot in snapshots if snapshot["view"]["phase"] == "your_turn"]
        assert {action["id"] for action in request["actions"]} == {snapshot["id"] for snapshot in active}
        for action in request["actions"]:
            snapshot = snapshots[action["id"]]
            view = snapshot["view"]
            assert action["version"] == view["version"]
            assert "claims" not in action
            expected = None if mode == "pass" else {"card": {"color": "red", "value": 4}, "stone": 0}
            assert action["play"] == expected
            steps[action["id"]] += 1
            finished = steps[action["id"]] == 1 + action["id"] % 2
            outcome = specifications[action["id"]]["seed"] % 3
            view["version"] += 1 if finished and outcome == 0 else 2
            if finished:
                view.update(status="finished", phase="game_over", turn=None,
                            winner=["you", "them", None][outcome],
                            win_reason="stalemate" if outcome == 2 else "five_stones")
    response = {"games": snapshots}
    if mode == "duplicate_ids" and len(snapshots) > 1:
        response = {"games": [snapshots[0]] * len(snapshots)}
    if mode == "wrong_seat":
        snapshots[0]["view"]["you"] = 1 - snapshots[0]["view"]["you"]
    if mode == "wrong_phase":
        snapshots[0]["view"]["phase"] = "opponent_turn"
    print(json.dumps(response), flush=True)
if mode == "bad_exit":
    sys.exit(9)
if mode in ("exit_hang", "ignore_terminate"):
    time.sleep(30)
''')
    worker.chmod(0o755)
    return dict(checkpoint=checkpoint, schedule=schedule, worker=worker,
                output=tmp_path / "result.json", start=1, count=5, batch_size=2, threads=3), requests, games


def evaluator():
    from schotten.neural import evaluate
    return evaluate


def test_evaluate_batches_masks_actions_and_preserves_schedule(experiment, capsys, monkeypatch):
    options, requests_path, games = experiment
    module = evaluator()
    loaded = []
    original_loader = module.load_checkpoint

    def load_once(*args, **kwargs):
        loaded.append((args, kwargs))
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(module, "load_checkpoint", load_once)
    monkeypatch.setenv("RAYON_NUM_THREADS", "99")
    result = module.evaluate(**options)
    assert json.loads(options["output"].read_text()) == result
    assert len(loaded) == 1
    assert loaded[0][0][1] == "cpu"
    assert torch.get_num_threads() == 1
    assert result["schema_version"] == 1
    assert result["start"] == 1 and result["count"] == 5
    for name in ("checkpoint", "worker", "schedule"):
        assert result[f"{name}_sha256"] == hashlib.sha256(options[name].read_bytes()).hexdigest()
    assert (result["wins"], result["losses"], result["draws"]) == (1, 2, 2)
    assert result["elapsed_seconds"] >= result["inference_seconds"] > 0
    assert result["inference_decisions"] == 7
    for index, game in enumerate(result["games"], start=1):
        assert game["schedule_index"] == index
        assert {key: game[key] for key in ("seed", "bot_seed", "mlp_seat")} == games[index]
        assert game["winner"] == ["mlp", "mcb:16", None][index % 3]
        assert game["reason"] == ("stalemate" if index % 3 == 2 else "five_stones")
        local_id = (index - 1) % 2
        expected_turns = games[index]["mlp_seat"] + local_id * 2 + (1 if index % 3 == 0 else 2)
        assert game["turns"] == expected_turns
    requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
    assert all(request["threads"] == "3" for request in requests)
    new_requests = [entry["request"]["games"] for entry in requests if entry["request"]["op"] == "new"]
    assert new_requests == [games[1:3], games[3:5], games[5:6]]
    progress = capsys.readouterr().out
    assert "2 / 5 games played" in progress and "5 / 5 games played" in progress


def test_pass_and_single_game_chunk(experiment, monkeypatch):
    options, _, _ = experiment
    monkeypatch.setenv("EVAL_MODE", "pass")
    result = evaluator().evaluate(**(options | {"count": 1}))
    assert result["inference_decisions"] == 1
    assert result["losses"] == 1


@pytest.mark.parametrize("mode,message", [
    ("error", "deliberate worker failure"), ("eof", "EOF"),
    ("invalid_json", "JSON"), ("duplicate_ids", "id"),
    ("wrong_seat", "seat"), ("wrong_phase", "phase"), ("bad_exit", "9"),
])
def test_worker_failures_leave_no_report(experiment, monkeypatch, mode, message):
    options, _, _ = experiment
    monkeypatch.setenv("EVAL_MODE", mode)
    with pytest.raises((RuntimeError, ValueError), match=message):
        evaluator().evaluate(**options)
    assert not options["output"].exists()
    assert not list(options["output"].parent.glob(".result.json.*"))


@pytest.mark.parametrize("mode", ["hang", "partial", "exit_hang", "ignore_terminate"])
def test_worker_timeout_including_partial_line_and_shutdown(experiment, monkeypatch, mode):
    options, _, _ = experiment
    monkeypatch.setenv("EVAL_MODE", mode)
    module = evaluator()
    processes = []
    original_popen = module.subprocess.Popen

    def track_process(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(module.subprocess, "Popen", track_process)
    monkeypatch.setattr(module, "RESPONSE_TIMEOUT", 0.1)
    monkeypatch.setattr(module, "EXIT_TIMEOUT", 0.1)
    with pytest.raises(RuntimeError, match="timed out"):
        module.evaluate(**options)
    assert not options["output"].exists()
    assert processes and all(process.poll() is not None for process in processes)


@pytest.mark.parametrize("changes", [
    {"start": -1}, {"start": 7}, {"count": 0}, {"count": 7},
    {"batch_size": 0}, {"threads": 0}, {"start": True}, {"count": 1.5},
])
def test_invalid_ranges_fail_before_starting_worker(experiment, changes):
    options, requests, _ = experiment
    with pytest.raises(ValueError):
        evaluator().evaluate(**(options | changes))
    assert not requests.exists()


@pytest.mark.parametrize("mutation", [
    {"schema_version": 2}, {"schema_version": True}, {"opponent": "mcb:8"},
    {"games": []}, {"games": [{"seed": 2**63, "bot_seed": 1, "mlp_seat": 0}]},
    {"games": [{"seed": 1, "bot_seed": -(2**63) - 1, "mlp_seat": 0}]},
    {"games": [{"seed": True, "bot_seed": 1, "mlp_seat": 0}]},
    {"games": [{"seed": 1, "bot_seed": 2.5, "mlp_seat": 0}]},
    {"games": [{"seed": 1, "bot_seed": 1, "mlp_seat": True}]},
    {"games": [{"seed": 1, "bot_seed": 1, "mlp_seat": 2}]},
    {"games": [{"seed": 1, "mlp_seat": 0}]},
])
def test_invalid_manifests(experiment, mutation):
    options, requests, _ = experiment
    manifest = json.loads(options["schedule"].read_text()) | mutation
    options["schedule"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        evaluator().evaluate(**options)
    assert not requests.exists()


def test_signed_seed_extremes_are_not_rounded(experiment):
    options, requests, _ = experiment
    game = {"seed": -(2**63), "bot_seed": 2**63 - 1, "mlp_seat": 1}
    options["schedule"].write_text(json.dumps({"schema_version": 1, "opponent": "mcb:16", "games": [game]}))
    result = evaluator().evaluate(**(options | {"start": 0, "count": 1}))
    assert result["games"][0]["seed"] == -(2**63)
    assert json.loads(requests.read_text().splitlines()[0])["request"]["games"] == [game]


def test_existing_output_is_preserved_without_starting_worker(experiment):
    options, requests, _ = experiment
    options["output"].write_text("preserve me")
    with pytest.raises(FileExistsError):
        evaluator().evaluate(**options)
    assert options["output"].read_text() == "preserve me"
    assert not requests.exists()


def test_report_created_by_other_process_is_not_overwritten(experiment, monkeypatch):
    options, _, _ = experiment
    monkeypatch.setenv("EVAL_MODE", "output_race")
    monkeypatch.setenv("EVAL_OUTPUT", str(options["output"]))
    with pytest.raises(FileExistsError):
        evaluator().evaluate(**options)
    assert options["output"].read_text() == "another evaluation"
    assert not list(options["output"].parent.glob(".result.json.*"))


def test_failed_atomic_write_leaves_no_report(experiment, monkeypatch):
    options, _, _ = experiment

    def fail_fsync(descriptor):
        raise OSError("disk error")

    monkeypatch.setattr(evaluator().os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="disk error"):
        evaluator().evaluate(**options)
    assert not options["output"].exists()
    assert not list(options["output"].parent.glob(".result.json.*"))


def test_frozen_checkpoint_repeats_identical_decisions(experiment):
    options, _, _ = experiment
    before_hash = hashlib.sha256(options["checkpoint"].read_bytes()).hexdigest()
    first = evaluator().evaluate(**options)
    second = evaluator().evaluate(**(options | {"output": options["output"].with_name("repeat.json")}))
    assert first["games"] == second["games"]
    assert first["inference_decisions"] == second["inference_decisions"]
    assert hashlib.sha256(options["checkpoint"].read_bytes()).hexdigest() == before_hash


def test_nonfinite_checkpoint_is_rejected(experiment):
    options, requests, _ = experiment
    payload = torch.load(options["checkpoint"], weights_only=True)
    payload["model_state"]["layers.4.bias"][0] = float("nan")
    torch.save(payload, options["checkpoint"])
    with pytest.raises(ValueError, match="finite"):
        evaluator().evaluate(**options)
    assert not requests.exists()


def test_cli_uses_required_options(experiment):
    options, _, _ = experiment
    command = [sys.executable, "-m", "schotten.neural.evaluate"]
    for key, value in options.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert "5 / 5 games played" in completed.stdout
    assert json.loads(options["output"].read_text())["count"] == 5


@pytest.mark.skipif(not os.environ.get("SCHOTTEN_NEURAL_WORKER"), reason="native worker opt-in")
def test_native_worker_integration(experiment):
    options, _, _ = experiment
    result = evaluator().evaluate(**(options | {
        "worker": Path(os.environ["SCHOTTEN_NEURAL_WORKER"]), "start": 0, "count": 2,
    }))
    assert result["wins"] + result["losses"] + result["draws"] == 2
    assert all(0 < game["turns"] < 200 for game in result["games"])
