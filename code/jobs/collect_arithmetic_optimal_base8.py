import argparse
import os
import sys
import subprocess
from datetime import datetime
from typing import List, Optional

import torch


def add_code_to_sys_path(project_root: str) -> None:
    code_dir = os.path.join(project_root, "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)


def ensure_poetry_env(project_root: str):
    """Ensure Poetry venv is created and dependencies installed; re-exec under venv."""
    print("Setting up Poetry environment")
    cwd = os.getcwd()
    os.chdir(project_root)
    try:
        subprocess.run(["poetry", "config", "virtualenvs.in-project", "true", "--local"], check=True)
        print("Installing dependencies...")
        subprocess.run(["poetry", "install", "--no-interaction", "--no-root"], check=True)
        venv_path = os.path.join(project_root, ".venv")
        venv_python = os.path.join(venv_path, "bin", "python")
        if os.path.isfile(venv_python):
            if os.path.realpath(sys.executable) != os.path.realpath(venv_python):
                print(f"Re-executing with venv interpreter: {venv_python}")
                script_path = os.path.abspath(sys.argv[0])
                os.execv(venv_python, [venv_python, script_path] + sys.argv[1:])
        else:
            result = subprocess.run(["poetry", "env", "info", "--path"], capture_output=True, text=True, check=True)
            venv_path = result.stdout.strip()
            venv_python = os.path.join(venv_path, "bin", "python")
            if os.path.isfile(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
                print(f"Re-executing with Poetry venv interpreter: {venv_python}")
                script_path = os.path.abspath(sys.argv[0])
                os.execv(venv_python, [venv_python, script_path] + sys.argv[1:])
        print(f"Using Poetry environment at: {venv_path}")
        return venv_path
    finally:
        os.chdir(cwd)


def load_optimal_direction(direction_path: str) -> torch.Tensor:
    if not os.path.exists(direction_path):
        raise FileNotFoundError(f"Optimal direction not found at: {direction_path}")
    vec = torch.load(direction_path)
    if isinstance(vec, torch.Tensor):
        return vec.float().cpu()
    raise TypeError(f"Unsupported direction object type: {type(vec)}")


def build_intervention_vectors(
    direction: torch.Tensor,
    inject_layer: Optional[int] = None,
    zero_first_n: int = 3,
) -> List[torch.Tensor]:
    # Create 32-layer list of vectors
    hidden_dim = direction.shape[0]
    zeros_vec = torch.zeros(hidden_dim, dtype=direction.dtype)
    if inject_layer is None:
        # Apply the same direction to all layers, optionally zeroing the first N
        vectors = [direction.clone() for _ in range(32)]
        for i in range(min(zero_first_n, 32)):
            vectors[i] = zeros_vec.clone()
        return vectors
    else:
        if inject_layer < 0 or inject_layer >= 32:
            raise ValueError(f"inject_layer must be in [0, 31], got {inject_layer}")
        vectors = [zeros_vec.clone() for _ in range(32)]
        vectors[inject_layer] = direction.clone()
        return vectors


def main():
    parser = argparse.ArgumentParser(description="Collect base8 arithmetic activations with optimal direction applied.")
    parser.add_argument("--project_root", type=str, default="/gpfs/home5/ljilesen/intervention-experiment")
    parser.add_argument("--direction_path", type=str, default="/home/ljilesen/intervention-experiment/outputs/optimal_directions/arithmetic-base8/optimal_direction.pth")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--input_path", type=str, default="/home/ljilesen/intervention-experiment/inputs/arithmetic/data/base8.txt")
    parser.add_argument("--output_dir", type=str, default="/home/ljilesen/intervention-experiment/outputs/arithmetic/intervention/optimal_base8_eval/")
    parser.add_argument("--inject_layer", type=int, default=-1, help="If >=0, apply only at this layer; otherwise apply at all layers (zeroing first N).")
    parser.add_argument("--zero_first_n", type=int, default=3, help="When applying to all layers, set the first N layers to zero.")
    args = parser.parse_args()

    # Ensure Poetry env like cross-eval job; will re-exec under Poetry venv if needed
    try:
        ensure_poetry_env(args.project_root)
    except Exception as e:
        print(f"Warning: Poetry setup failed or Poetry not installed ({e}); continuing with current interpreter.")
    add_code_to_sys_path(args.project_root)

    # Imports that rely on sys.path
    from experiments.arithmetic import ArithmeticExperiment  # noqa: E402

    os.makedirs(args.output_dir, exist_ok=True)

    direction = load_optimal_direction(args.direction_path)
    inject_layer = None if args.inject_layer < 0 else int(args.inject_layer)
    intervention_vectors = build_intervention_vectors(direction, inject_layer=inject_layer, zero_first_n=args.zero_first_n)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out = os.path.join(
        args.output_dir,
        f"alpha_{args.alpha:.2f}_layer_{'all' if inject_layer is None else inject_layer}_{timestamp}",
    )
    os.makedirs(run_out, exist_ok=True)

    print("=== Collecting activations for Arithmetic base8 with optimal direction ===")
    print(f"Input file   : {args.input_path}")
    print(f"Output dir   : {run_out}")
    print(f"Alpha        : {args.alpha}")
    print(f"Inject layer : {'all (zero_first_n=' + str(args.zero_first_n) + ')' if inject_layer is None else inject_layer}")

    experiment = ArithmeticExperiment(
        input_path=args.input_path,
        output_path=run_out,
        chunk_size=args.chunk_size,
        chunk_id=args.chunk_id,
        base=8,
    )

    dataset = experiment.run_experiment(
        intervention_vectors=intervention_vectors,
        alpha=args.alpha,
        collect_activations=True,
        mode="counter_factual",
        save_activations=True,
    )

    print(f"Saved dataset with activations to: {run_out}")
    print(f"Columns: {dataset.column_names}")


if __name__ == "__main__":
    main()


