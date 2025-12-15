import argparse
import json
import os
import random
import sys
from datetime import datetime

import torch


def ensure_poetry_env(project_root: str):
    """Ensure Poetry venv is created and switch to its interpreter if needed."""
    import subprocess

    cwd = os.getcwd()
    os.chdir(project_root)
    try:
        subprocess.run(
            ["poetry", "config", "virtualenvs.in-project", "true", "--local"],
            check=True,
        )
        subprocess.run(
            ["poetry", "install", "--no-interaction", "--no-root"], check=True
        )
        venv_path = os.path.join(project_root, ".venv")
        venv_python = os.path.join(venv_path, "bin", "python")
        if os.path.isfile(venv_python) and os.path.realpath(
            sys.executable
        ) != os.path.realpath(venv_python):
            script_path = os.path.abspath(sys.argv[0])
            os.execv(venv_python, [venv_python, script_path] + sys.argv[1:])
        # Already under venv interpreter; add site-packages
        pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        for base in ("lib", "lib64"):
            sp = os.path.join(venv_path, base, pyver, "site-packages")
            if os.path.isdir(sp) and sp not in sys.path:
                sys.path.insert(0, sp)
    finally:
        os.chdir(cwd)


def _format_activations(dataset):
    activations = dataset["residual_activations"]
    model_layers_num = 32
    mlp_dim_num = 4096
    layer_activation_dict = {
        i: torch.zeros(len(activations), mlp_dim_num) for i in range(model_layers_num)
    }
    for i in range(len(activations)):
        for j in range(model_layers_num):
            layer_activation_dict[j][i] = torch.tensor(activations[i][j])
    return layer_activation_dict


def _calculate_diff_means_directions(
    layer_activation_dict, first_indices, second_indices
):
    model_layers_num = 32
    mlp_dim_num = 4096
    candidate_directions = torch.zeros(
        (model_layers_num, mlp_dim_num), dtype=torch.float64, device="cuda"
    )
    for layer in range(model_layers_num):
        activations = layer_activation_dict[layer]
        a = activations[first_indices, :].to(torch.float64)
        b = activations[second_indices, :].to(torch.float64)
        candidate_directions[layer] = a.mean(dim=0) - b.mean(dim=0)
    return candidate_directions


def compute_gibberish_negative(project_root: str):
    from experiments.arithmetic import ArithmeticExperiment

    base = 8
    results_path = os.path.join(
        project_root, "outputs", "arithmetic", "tuning", "gibberish"
    )
    os.makedirs(os.path.join(results_path, "activations"), exist_ok=True)
    experiment = ArithmeticExperiment(
        input_path=os.path.join(
            project_root, "inputs", "arithmetic", "data", f"base{base}.txt"
        ),
        output_path=os.path.join(results_path, "activations"),
        chunk_size=1000,
        chunk_id=0,
        base=base,
    )
    dataset = experiment.run_experiment(
        collect_activations=True,
        mode="counter_factual",
        gibberish=True,
        save_activations=False,
    )
    _, correct_prediction_indices, _, _ = experiment.evaluate_llm_responses(dataset)
    incorrect_prediction_indices = [i for i in range(1000, 2000)]
    # Balance counts to original chunk size
    incorrect_prediction_indices = random.sample(
        incorrect_prediction_indices, max(0, 1000 - len(correct_prediction_indices))
    )
    layer_activation_dict = _format_activations(dataset)
    diff_means = _calculate_diff_means_directions(
        layer_activation_dict, correct_prediction_indices, incorrect_prediction_indices
    )
    out_path = os.path.join(results_path, "diff_means_directions.pth")
    torch.save(diff_means.cpu().float(), out_path)
    return out_path


def compute_exclusive_cf_negative(project_root: str):
    from experiments.arithmetic import ArithmeticExperiment

    base = 8
    results_path = os.path.join(
        project_root, "outputs", "arithmetic", "tuning", "exclusive"
    )
    os.makedirs(os.path.join(results_path, "activations"), exist_ok=True)
    experiment = ArithmeticExperiment(
        input_path=os.path.join(
            project_root, "inputs", "arithmetic", "data", f"base{base}.txt"
        ),
        output_path=os.path.join(results_path, "activations"),
        chunk_size=1000,
        chunk_id=0,
        base=base,
    )
    dataset = experiment.run_experiment(
        collect_activations=True,
        mode="counter_factual",
        save_activations=False,
    )
    _, correct_prediction_indices, real_world_prediction_indices, _ = (
        experiment.evaluate_llm_responses(dataset)
    )
    incorrect_prediction_indices = real_world_prediction_indices
    layer_activation_dict = _format_activations(dataset)
    diff_means = _calculate_diff_means_directions(
        layer_activation_dict, correct_prediction_indices, incorrect_prediction_indices
    )
    out_path = os.path.join(results_path, "diff_means_directions.pth")
    torch.save(diff_means.cpu().float(), out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_root",
        type=str,
        default="/gpfs/home3/ljilesen/interaction-experiment",
    )
    args = parser.parse_args()

    ensure_poetry_env(args.project_root)
    sys.path.insert(0, os.path.join(args.project_root, "code"))

    torch.manual_seed(8888)
    random.seed(8888)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(8888)
        torch.cuda.manual_seed_all(8888)

    results = {
        "gibberish": compute_gibberish_negative(args.project_root),
        "exclusive": compute_exclusive_cf_negative(args.project_root),
    }
    summary_path = os.path.join(
        args.project_root, "outputs", "arithmetic", "tuning", "variant_summary.json"
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {"results": results, "created_at": datetime.now().isoformat()}, f, indent=2
        )
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
