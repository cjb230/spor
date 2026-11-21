"""Freeze a neural-player experiment and summarize paired held-out results."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import secrets
import sys

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root, decks=10000, master_seed=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "schedule.json").exists():
        raise FileExistsError("evaluation schedule is already frozen")
    if type(decks) is not int or decks < 1:
        raise ValueError("decks must be positive")
    master_seed = secrets.randbits(128) if master_seed is None else master_seed
    generator = random.Random(master_seed)
    seeds = set()
    games = []
    while len(seeds) < decks:
        seed = generator.getrandbits(63)
        if seed in seeds or 10000 <= seed < 16000:
            continue
        seeds.add(seed)
        bot_seed = generator.getrandbits(63)
        games.extend({"seed": seed, "bot_seed": bot_seed, "mlp_seat": seat} for seat in (0, 1))
    schedule = {"schema_version": 1, "opponent": "mcb:16", "master_seed": str(master_seed), "games": games}
    (root / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    manifest = {"schema_version": 1, "decks": decks, "games_per_checkpoint": len(games),
                "master_seed": str(master_seed), "schedule_sha256": digest(root / "schedule.json"),
                "initialization_seed": 20261121, "opponent": "mcb:16",
                "teacher_deck_ranges": [[10000,11499],[11500,12999],[13000,14499],[14500,15999]],
                "teacher_opponents": ["builder", "tactician", "planner", "mcb:16"],
                "training_seconds": 1200, "training_seed": 1, "batch_size": 512,
                "learning_rate": 0.001, "selection": "minimum held-out validation cross entropy"}
    if (root / "before.pt").exists():
        manifest["before_sha256"] = digest(root / "before.pt")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return schedule


def summarize(schedule, before, after, resamples=10000, seed=20261121):
    def ordered(rows):
        indexed = {row["schedule_index"]: row for row in rows}
        if len(indexed) != len(rows) or set(indexed) != set(range(len(schedule))):
            raise ValueError("evaluation results have missing or duplicate scheduled games")
        result = []
        for index, spec in enumerate(schedule):
            row = indexed[index]
            if any(row[key] != spec[key] for key in ("seed", "bot_seed", "mlp_seat")):
                raise ValueError("results do not match the frozen evaluation schedule")
            if row["winner"] not in ("mlp", "mcb:16", None):
                raise ValueError("invalid winner")
            result.append(row)
        return result
    before, after = ordered(before), ordered(after)
    def stats(rows):
        wins = sum(row["winner"] == "mlp" for row in rows)
        draws = sum(row["winner"] is None for row in rows)
        return {"wins": wins, "draws": draws, "losses": len(rows)-wins-draws,
                "win_rate": wins/len(rows), "score": (wins+draws/2)/len(rows)}
    result = {"games_per_checkpoint": len(schedule), "before": stats(before), "after": stats(after)}
    result["by_seat"] = {str(seat): {"before": stats([row for row in before if row["mlp_seat"]==seat]),
                                   "after": stats([row for row in after if row["mlp_seat"]==seat])} for seat in (0,1)}
    def score(row):
        return 1.0 if row["winner"] == "mlp" else 0.5 if row["winner"] is None else 0.0
    groups = defaultdict(list)
    for old, new in zip(before, after):
        groups[old["seed"]].append((old, new))
    values = []
    for group in groups.values():
        if len(group) != 2 or {old["mlp_seat"] for old, _ in group} != {0,1}:
            raise ValueError("each evaluation deck must occur exactly once per seat")
        values.append([np.mean([score(old) for old,_ in group]),
                       np.mean([score(new) for _,new in group]),
                       np.mean([score(new)-score(old) for old,new in group])])
    values = np.asarray(values)
    generator = np.random.default_rng(seed)
    means = []
    for start in range(0, resamples, 100):
        samples = generator.integers(0, len(values), size=(min(100,resamples-start),len(values)))
        means.append(values[samples].mean(axis=1))
    interval = np.quantile(np.concatenate(means), [0.025,0.975], axis=0)
    result.update({"decks": len(groups), "score_improvement": result["after"]["score"]-result["before"]["score"],
                   "score_95_ci": {"before": interval[:,0].tolist(),"after":interval[:,1].tolist(),"improvement":interval[:,2].tolist()},
                   "bootstrap_resamples": resamples,
                   "loss_to_win": sum(old["winner"]=="mcb:16" and new["winner"]=="mlp" for old,new in zip(before,after)),
                   "win_to_loss": sum(old["winner"]=="mlp" and new["winner"]=="mcb:16" for old,new in zip(before,after))})
    return result


def report(root):
    root = Path(root)
    schedule = json.loads((root / "schedule.json").read_text())
    reports = {}
    for phase in ("before", "after"):
        reports[phase] = [json.loads(path.read_text()) for path in sorted((root / phase).glob("*.json"))]
        if not reports[phase]:
            raise ValueError(f"no {phase} evaluation reports")
    all_reports = reports["before"] + reports["after"]
    if {row["schedule_sha256"] for row in all_reports} != {digest(root / "schedule.json")}:
        raise ValueError("evaluation reports used a different schedule")
    if len({row["worker_sha256"] for row in all_reports}) != 1:
        raise ValueError("evaluation reports used different native workers")
    for phase, name in (("before","before.pt"),("after","after.pt")):
        if {row["checkpoint_sha256"] for row in reports[phase]} != {digest(root / name)}:
            raise ValueError(f"{phase} evaluation used a different checkpoint")
    result = summarize(schedule["games"], [game for row in reports["before"] for game in row["games"]],
                       [game for row in reports["after"] for game in row["games"]])
    result["timings"] = {phase: {"elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
                                "inference_seconds":sum(row["inference_seconds"] for row in rows),
                                "inference_decisions":sum(row["inference_decisions"] for row in rows)} for phase,rows in reports.items()}
    (root / "results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


def freeze(root, worker):
    import torch
    root=Path(root)
    path=root/"manifest.json"
    manifest=json.loads(path.read_text())
    if "worker_sha256" in manifest:
        raise ValueError("experiment code is already frozen")
    manifest.update({"worker_sha256":digest(worker),
                     "teacher_binary_sha256":digest("rust/target/release/schotten-native"),
                     "python_executable":sys.executable,"python_version":sys.version,
                     "torch_version":str(torch.__version__),"numpy_version":str(np.__version__),
                     "source_sha256":{str(path):digest(path) for path in [
                         Path("schotten/neural/encoding.py"),Path("schotten/neural/model.py"),
                         Path("schotten/neural/evaluate.py"),Path("schotten/neural/timed_train.py"),
                         Path("rust/examples/neural_worker.rs"),Path(__file__)]}})
    path.write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","freeze","report"))
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--decks",type=int,default=10000)
    parser.add_argument("--worker",type=Path,default=Path("rust/target/release/examples/neural_worker"))
    args=parser.parse_args()
    if args.command=="prepare":
        schedule=prepare(args.root,args.decks)
        print(f"Frozen {args.decks} decks, {len(schedule['games'])} games in {args.root}")
    elif args.command=="freeze":
        freeze(args.root,args.worker)
    else:
        report(args.root)


if __name__=="__main__":
    main()
