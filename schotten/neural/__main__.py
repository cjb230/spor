"""Train, inspect, or play the first neural policy."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m schotten.neural", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="save explicitly untrained initial weights")
    initialize.add_argument("--output", type=Path, required=True)
    initialize.add_argument("--seed", type=int, default=1)
    initialize.add_argument("--hidden", type=int, nargs=2, default=(256, 256))
    inspect = commands.add_parser("inspect", help="show checkpoint architecture and training metadata")
    inspect.add_argument("checkpoint", type=Path)
    trainer = commands.add_parser("train", help="imitate selected strategies from native recordings")
    trainer.add_argument("--recordings", type=Path, nargs="+", required=True)
    trainer.add_argument("--teachers", nargs="+", default=["mcb"])
    trainer.add_argument("--output", type=Path, required=True)
    trainer.add_argument("--resume", type=Path)
    trainer.add_argument("--epochs", type=int, default=10, help="additional epochs, including on resume")
    trainer.add_argument("--batch-size", type=int, default=256)
    trainer.add_argument("--learning-rate", type=float, default=0.001)
    trainer.add_argument("--validation-fraction", type=float, default=0.2)
    trainer.add_argument("--seed", type=int, default=1)
    trainer.add_argument("--hidden", type=int, nargs=2, default=(256, 256))
    trainer.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    generator = commands.add_parser("generate", help="generate compact disk-backed teacher examples")
    generator.add_argument("--output", type=Path, required=True)
    generator.add_argument("--examples", type=int, required=True, help="exact training row count")
    generator.add_argument("--validation-examples", type=int, default=100000,
                           help="additional examples from separate games")
    generator.add_argument("--teacher", default="mcb:16")
    generator.add_argument("--opponents", nargs="+", default=["builder", "tactician", "planner", "mcb:16"])
    generator.add_argument("--seed", type=int, default=1001)
    generator.add_argument("--shard-size", type=int, default=65536)
    generator.add_argument("--threads", type=int, default=None,
                           help="native worker threads (default: all available CPU cores)")
    generator.add_argument("--worker", type=Path, default=Path("rust/target/release/examples/neural_generate"))
    generator.add_argument("--progress-interval", type=float, default=1)
    generator.add_argument("--resume", action="store_true")
    streaming = commands.add_parser("train-stream", help="train on shards with resumable total epochs")
    streaming.add_argument("--dataset", type=Path, required=True)
    streaming.add_argument("--initial", type=Path, required=True)
    streaming.add_argument("--output", type=Path, required=True, help="directory for last.pt, best.pt and metrics")
    streaming.add_argument("--epochs", type=int, required=True, help="total target epochs, including on resume")
    streaming.add_argument("--batch-size", type=int, default=512)
    streaming.add_argument("--learning-rate", type=float, default=0.001)
    streaming.add_argument("--seed", type=int, default=1)
    streaming.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    streaming.add_argument("--resume", type=Path)
    streaming.add_argument("--progress-interval", type=float, default=1)
    streaming.add_argument("--checkpoint-interval", type=float, default=60)
    for command in ("create", "join"):
        player = commands.add_parser(command, help=f"{command} a REST game using saved weights")
        player.add_argument("--checkpoint", type=Path, required=True)
        player.add_argument("--url", default="http://localhost:8000")
        player.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
        if command == "join":
            player.add_argument("game_id")
        else:
            player.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            from .generate import generate
            generate(**{key: value for key, value in vars(args).items() if key != "command"})
            return 0
        if args.command == "train-stream":
            from .stream_train import train_stream
            train_stream(**{key: value for key, value in vars(args).items() if key != "command"})
            return 0
        import torch
        from .model import Policy, load_checkpoint, save_checkpoint
        if args.command == "init":
            torch.manual_seed(args.seed)
            model = Policy(args.hidden)
            save_checkpoint(args.output, model, metadata={"untrained": True, "seed": args.seed})
            print(f"Saved untrained mlp weights to {args.output}")
        elif args.command == "inspect":
            model, payload = load_checkpoint(args.checkpoint)
            information = {key: value for key, value in payload.items() if key not in ("model_state", "optimizer_state")}
            information["parameters"] = sum(parameter.numel() for parameter in model.parameters())
            print(json.dumps(information, indent=2))
        elif args.command == "train":
            from .train import train
            train(args.recordings, args.output, args.teachers, epochs=args.epochs,
                  batch_size=args.batch_size, learning_rate=args.learning_rate,
                  validation_fraction=args.validation_fraction, seed=args.seed,
                  device=args.device, resume=args.resume, hidden=tuple(args.hidden))
        else:
            from ..client import SchottenClient
            from .player import NeuralBot
            client = SchottenClient(args.url)
            try:
                bot = NeuralBot(client, args.checkpoint, device=args.device, verbose=True)
                if args.command == "create":
                    grant = client.create_game(seed=args.seed)
                    print(f"game {grant.game_id} created; waiting for an opponent", flush=True)
                else:
                    client.join(args.game_id)
                bot.run()
            finally:
                client.http.close()
        return 0
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        parser.exit(2, "PyTorch is optional. Install it with: python -m pip install -r requirements-ml.txt\n")
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
