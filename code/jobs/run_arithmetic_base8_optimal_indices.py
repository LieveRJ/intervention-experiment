import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional, Tuple

import torch

# Alpha sweep from -0.25 to 0.25 in steps of 0.05 (inclusive)
ALPHAS = [-0.25, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25]


def ensure_poetry_env(project_root: str):
    """Ensure Poetry venv is created and dependencies installed; re-exec under venv."""
    print("Setting up Poetry environment")
    cwd = os.getcwd()
    os.chdir(project_root)
    try:
        subprocess.run(
            ["poetry", "config", "virtualenvs.in-project", "true", "--local"],
            check=True,
        )
        print("Installing dependencies...")
        subprocess.run(
            ["poetry", "install", "--no-interaction", "--no-root"], check=True
        )
        venv_path = os.path.join(project_root, ".venv")
        venv_python = os.path.join(venv_path, "bin", "python")
        if os.path.isfile(venv_python):
            if os.path.realpath(sys.executable) != os.path.realpath(venv_python):
                print(f"Re-executing with venv interpreter: {venv_python}")
                script_path = os.path.abspath(sys.argv[0])
                os.execv(venv_python, [venv_python, script_path] + sys.argv[1:])
        else:
            result = subprocess.run(
                ["poetry", "env", "info", "--path"],
                capture_output=True,
                text=True,
                check=True,
            )
            venv_path = result.stdout.strip()
            venv_python = os.path.join(venv_path, "bin", "python")
            if os.path.isfile(venv_python) and os.path.realpath(
                sys.executable
            ) != os.path.realpath(venv_python):
                print(f"Re-executing with Poetry venv interpreter: {venv_python}")
                script_path = os.path.abspath(sys.argv[0])
                os.execv(venv_python, [venv_python, script_path] + sys.argv[1:])
        print(f"Using Poetry environment at: {venv_path}")
        return venv_path
    finally:
        os.chdir(cwd)


def _format_intervention_vectors(layer_diff_means: torch.Tensor):
    intervention_vectors = [layer_diff_means] * 32
    intervention_vectors[:3] = torch.zeros(3, 4096)
    return intervention_vectors


def _load_optimal_direction() -> Tuple[torch.Tensor, str, Optional[int]]:
    """Load the optimal direction for arithmetic-base8; fallback to best layer from layerwise.

    Returns (direction_vector, source_kind, layer_idx_or_None).
    """
    optimal_path = "./outputs/optimal_directions/arithmetic-base8/optimal_direction.pth"
    layerwise_path = (
        "./outputs/arithmetic/tuning/counter_factual/diff_means_directions.pth"
    )

    if os.path.exists(optimal_path):
        vec = torch.load(optimal_path)
        return vec.cpu().float(), "optimal", None
    if os.path.exists(layerwise_path):
        layer_wise = torch.load(layerwise_path)
        norms = torch.linalg.norm(layer_wise, dim=1)
        best_layer = int(torch.argmax(norms).item())
        vec = layer_wise[best_layer].cpu().float()
        return vec, "layerwise_max_norm_fallback", best_layer
    raise FileNotFoundError(
        "Neither optimal nor layerwise directions found for arithmetic-base8"
    )


def eval_on_arithmetic(
    direction: torch.Tensor, alpha: float, chunk_size: int = 1000, chunk_id: int = 0
):
    from experiments.arithmetic import ArithmeticExperiment

    intervention_vectors = _format_intervention_vectors(direction)
    experiment = ArithmeticExperiment(
        input_path=f"./inputs/arithmetic/data/base8.txt",
        output_path=f"./outputs/arithmetic/intervention/eval_temp/",
        chunk_size=chunk_size,
        chunk_id=chunk_id,
        base=8,
    )
    dataset = experiment.run_experiment(
        intervention_vectors=intervention_vectors,
        alpha=alpha,
        collect_activations=False,
        mode="counter_factual",
    )
    (
        acc,
        correct_answer_indices,
        real_world_answer_indices,
        unparsable_answer_indices,
    ) = experiment.evaluate_llm_responses(dataset)
    return {
        "accuracy": float(acc),
        "counter_factual_indices": correct_answer_indices,
        "real_world_indices": real_world_answer_indices,
        "unparsable_indices": unparsable_answer_indices,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_root",
        type=str,
        default="/gpfs/home3/ljilesen/interaction-experiment",
    )
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument(
        "--out_path",
        type=str,
        default="./outputs/arithmetic-base8/tuning/counter_factual/optimal_sweep_indices.json",
    )
    args = parser.parse_args()

    ensure_poetry_env(args.project_root)
    sys.path.insert(0, os.path.join(args.project_root, "code"))

    direction, source_kind, layer_idx = _load_optimal_direction()

    indices_by_alpha = {}
    for alpha in ALPHAS:
        alpha_key = f"{alpha:.2f}"
        print(f"Evaluating arithmetic-base8 with alpha {alpha_key} using {source_kind}")
        res = eval_on_arithmetic(
            direction, alpha, chunk_size=args.chunk_size, chunk_id=args.chunk_id
        )
        indices_by_alpha[alpha_key] = {
            "counter_factual_indices": res["counter_factual_indices"],
            "real_world_indices": res["real_world_indices"],
            "unparsable_indices": res["unparsable_indices"],
            "accuracy": res["accuracy"],
        }

    manifest = {
        "created_at": datetime.now().isoformat(),
        "task": "arithmetic-base8",
        "alpha_sweep": ALPHAS,
        "direction_source": source_kind,
        "layer_index": layer_idx,
        "chunk_size": args.chunk_size,
        "chunk_id": args.chunk_id,
        "indices_by_alpha": indices_by_alpha,
    }

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved indices to {args.out_path}")


if __name__ == "__main__":
    main()
