import argparse
import json
from pathlib import Path

from .fem import Config, simulate
from .network import Predictor, train
from .pipeline import collect, compare, convergence, new_directory, write_json


def main():
    parser = argparse.ArgumentParser(description="Navier–Stokes / ANN / AMG research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "run", "compare", "convergence"):
        p = sub.add_parser(name)
        p.add_argument("--backend", choices=("petsc", "petsc-cuda", "pyamg"), default="petsc")
        p.add_argument("--output", type=Path, required=True)
        if name in ("run", "compare"):
            p.add_argument("--cells", type=int, default=12)
            p.add_argument("--dt", type=float, default=0.002)
            p.add_argument("--steps", type=int, default=5)
            p.add_argument("--viscosity", type=float, default=0.05)
            p.add_argument("--model", type=Path, required=(name == "compare"))
        if name in ("collect", "compare"):
            p.add_argument("--repeats", type=int, default=3)
        if name == "collect":
            p.add_argument("--profile", choices=("local", "smoke", "full"), default="local")
            p.add_argument("--steps", type=int, default=5)
    p = sub.add_parser("train")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    if args.command == "collect":
        result = collect(args.output, args.backend, args.repeats, args.steps, profile=args.profile)
    elif args.command == "train":
        if args.output.exists():
            raise FileExistsError(f"Model already exists: {args.output}")
        result = train(args.dataset, args.output, args.epochs)
    elif args.command == "convergence":
        result = convergence(args.output, args.backend)
    else:
        config = Config(cells=args.cells, dt=args.dt, steps=args.steps, viscosity=args.viscosity)
        if args.command == "compare":
            result = compare(config, args.model, args.output, args.backend, args.repeats)
        else:
            import numpy as np
            output = new_directory(args.output)
            selector = Predictor(args.model, args.backend) if args.model else None
            result, fields, _ = simulate(config, args.backend, selector)
            if selector:
                result["selections"] = selector.selections
            write_json(output / "run.json", result)
            np.savez_compressed(output / "fields.npz", **fields)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
